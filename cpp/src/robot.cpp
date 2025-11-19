#include "robot.hpp"

#include <array>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <cstring>

namespace robot {

Robot::Robot()
    : 
    last_action_len(8),
    last_action(last_action_len, 0.0f),  
    last_torque_ctrl(false),      

    kp(last_action_len, 0.0f),
    kd(last_action_len, 0.0f),
    gains_set(false),

    motor_ids{1u,2u,3u,4u,5u,6u,7u,8u},
    motor_pattern(last_action_len, "N"),      // 기본 pattern: "N"
    motor_err(last_action_len, "None"),       // 기본 err: "None"

    battery_voltage(""),
    battery_soc(""),

    max_disconn_count(5), 
    disconn_count(0),
    max_req_miss_count(5), 
    req_miss_count(0),

    cli()
{
    // Observation containers (pre-sized & reused)
    obs["dof_pos"] = std::vector<float>(6, 0.0f);   // 6개 관절 (바퀴 제외)
    obs["dof_vel"] = std::vector<float>(8, 0.0f);   // 8개 모터 속도 (바퀴 포함)
    obs["ang_vel"] = std::vector<float>(3, 0.0f);
    obs["proj_grav"] = std::vector<float>(3, 0.0f);
    obs["lin_vel"] = std::vector<float>(3, 0.0f);
    obs["height_map"] = std::vector<float>(144, 0.6128f);

    // Offsets & limits
    pos_offset = {
        {"left_hip", 0.0f}, {"right_hip", 0.0f},
        {"left_shoulder", 0.0f}, {"right_shoulder", 0.0f},
        {"left_leg", 0.0f}, {"right_leg", 0.0f},
    };
    rel_max_pos = {
        {"left_hip", 0.65f},   {"right_hip", 0.65f},
        {"left_shoulder", 1.5f}, {"right_shoulder", 1.5f},
        {"left_leg", 1.65f},  {"right_leg", 1.65f},
    };
    rel_min_pos = {
        {"left_hip", -0.67f},   {"right_hip", -0.67f},
        {"left_shoulder", -1.5f}, {"right_shoulder", -1.5f},
        {"left_leg", -0.9f}, {"right_leg", -0.9f},
    };
    joint_names = { // pos 인덱스 0..5에 해당하는 관절 이름
        "left_hip","right_hip","left_shoulder","right_shoulder","left_leg","right_leg",
    };

    wait(); // 보드 준비 대기
}

// ------- Wait All Nodes -------
void Robot::wait(std::int32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto retry_sleep = std::chrono::milliseconds(100);

    while (std::chrono::steady_clock::now() < deadline) {        
        bool started = cli.motor_start(motor_ids);
        if (!(started)) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        
        FxCliMap status = cli.status();
        auto [dis, emg, bat] = check_status(status, motor_ids);
        if (dis || emg) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        if(bat) estop("\033[31m[E-stop]\033[0m Battery low");

        FxCliMap mcu_obs = cli.req(motor_ids);
        parse_obs(mcu_obs);
        if(req_miss_count > 0){
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        auto& dof_pos = obs.at("dof_pos");
        for(int i=0; i<6; ++i){ // M1 .... M6
            last_action[i] = dof_pos[i];
        }
        for(int i=6; i<8; ++i){ // M7, M8
            last_action[i] = 0.0f;
        }
        last_torque_ctrl = false;
        return;
    }
    throw RobotInitError("Motor start timeout");
}

// ------- Set gains -------
void Robot::set_gains(const std::vector<float>& kp_, const std::vector<float>& kd_) {
    std::ostringstream oss;
    if (kp_.size() != last_action_len){
        oss << "\033[31mset_gains: kp length mismatch for the robot. "
            << "Expected " << last_action_len 
            << ", but Got " << kp_.size()
            << "\033[0m";
        throw RobotSetGainsError(oss.str());
    }
    if (kd_.size() != last_action_len){
        oss << "\033[31mset_gains: kd length mismatch for the robot. "
            << "Expected " << last_action_len 
            << ", but Got " << kd_.size()
            << "\033[0m";
        throw RobotSetGainsError(oss.str());
    }
    if (kp_[6] != 0.0f || kp_[7] != 0.0f){
        std::ostringstream oss;
        oss << "\033[31mset_gains: wheel motor kp must be zero for indices 6 and 7. But got "
            << "kp[6] = " << kp_[6]
            << ", kp[7] = " << kp_[7]
            << "\033[0m";
        throw RobotSetGainsError(oss.str());
    }
    for (std::size_t i = 0; i < kp_.size(); ++i) {
        if (kp_[i] < 0.0f) {
            std::ostringstream oss3;
            oss3 << "\033[31mset_gains: kp must be non-negative. But got "
                 << "kp[" << i << "] = " << kp_[i]
                 << "\033[0m";
            throw RobotSetGainsError(oss3.str());
        }
    }
    for (std::size_t i = 0; i < kd_.size(); ++i) {
        if (kd_[i] < 0.0f) {
            std::ostringstream oss4;
            oss4 << "\033[31mset_gains: kd must be non-negative. But got "
                 << "kd[" << i << "] = " << kd_[i]
                 << "\033[0m";
            throw RobotSetGainsError(oss4.str());
        }
    }
    kp = kp_; kd = kd_; gains_set = true;
}

// ------- Parse Observation -------
std::unordered_map<std::string, std::vector<float>>& Robot::parse_obs(const FxCliMap& mcu_obs) {
    // 1) ACK / REQ check
    if (auto it_ack = mcu_obs.find("ACK"); it_ack != mcu_obs.end()) {
        const auto& ack = it_ack->second;
        auto it = ack.find("REQ");
        if (!(it != ack.end() && it->second == "true")) {
            ++req_miss_count;
            return obs;
        }
    }
    else{
        ++req_miss_count;
        return obs;
    }

    // 2) Get references to observation vectors
    auto& dof_pos   = obs.at("dof_pos");   // size: 6
    auto& dof_vel   = obs.at("dof_vel");   // size: 8
    auto& ang_vel   = obs.at("ang_vel");   // size: 3
    auto& proj_grav = obs.at("proj_grav"); // size: 3

    // Helper: get motor map M1..M8
    auto get_motor = [&](int index)
        -> const std::unordered_map<std::string, std::string>*
    {
        std::string key = "M" + std::to_string(index); // "M1", "M2", ...
        auto it = mcu_obs.find(key);
        if (it == mcu_obs.end()) {
            return nullptr;
        }
        return &it->second;
    };

    // 3) Parse M1..M6 (leg joints: position + velocity)
    for (int i = 0; i < 6; ++i) {
        int mid = i + 1; // 1..6
        const auto* m = get_motor(mid);
        if (!m) {
            ++req_miss_count;
            return obs;
        }

        auto it_p = m->find("p");
        auto it_v = m->find("v");
        if (it_p == m->end() || it_v == m->end()) {
            ++req_miss_count;
            return obs;
        }
        if (it_p->second == "N" || it_v->second == "N") {
            ++req_miss_count;
            return obs;
        }
        float p = std::stof(it_p->second);
        float v = std::stof(it_v->second);

        // Apply joint offset
        const std::string& jname = joint_names[i];
        float off = pos_offset[jname];

        dof_pos[i] = p + off;
        dof_vel[i] = v;
    }
    // 4) Parse M7..M8 (wheel motors: velocity only)
    for (int i = 6; i < 8; ++i) {
        int mid = i + 1; // 7, 8
        const auto* m = get_motor(mid);
        if (!m) {
            ++req_miss_count;
            return obs;
        }

        auto it_v = m->find("v");
        if (it_v == m->end() || it_v->second == "N") {
            ++req_miss_count;
            return obs;
        }
        float v = std::stof(it_v->second);
        dof_vel[i] = v;
    }

    // 5) Parse IMU
    auto it_imu = mcu_obs.find("IMU");
    if (it_imu == mcu_obs.end()) {
        ++req_miss_count;
        return obs;
    }

    const auto& imu = it_imu->second;
    auto get_imu = [&](const char* key, float& out) -> bool {
        auto it = imu.find(key);
        if (it == imu.end() || it->second == "N")
            return false;
        out = std::stof(it->second);
        return true;
    };

    float gx, gy, gz, pgx, pgy, pgz;
    if (!get_imu("gx", gx)  ||
        !get_imu("gy", gy)  ||
        !get_imu("gz", gz)  ||
        !get_imu("pgx", pgx)||
        !get_imu("pgy", pgy)||
        !get_imu("pgz", pgz))
    {
        ++req_miss_count;
        return obs;
    }

    ang_vel[0]   = gx;
    ang_vel[1]   = gy;
    ang_vel[2]   = gz;
    proj_grav[0] = pgx;
    proj_grav[1] = pgy;
    proj_grav[2] = pgz;

    req_miss_count = 0;
    return obs;
}

// ------- Get Observation -------
std::unordered_map<std::string, std::vector<float>> Robot::get_obs() {
    do_action(last_action, last_torque_ctrl, false);
    FxCliMap mcu_obs = cli.req(motor_ids);
    auto& parsed = parse_obs(mcu_obs);
    return parsed;
}

// ------- Do Action -------
void Robot::do_action(const std::vector<float>& action, bool torque_ctrl, bool safe) {
    if (action.size() != last_action_len) {
        std::ostringstream oss;
        oss << "\033[31mdo_action: action length mismatch. "
            << "Expected " << last_action_len
            << ", but Got " << action.size()
            << "\033[0m";
        estop(oss.str());
    }

    std::vector<float> pos(last_action_len, 0.0f);
    std::vector<float> vel(last_action_len, 0.0f);
    std::vector<float> tau(last_action_len, 0.0f);
    std::vector<float> kp_(last_action_len, 0.0f);
    std::vector<float> kd_(last_action_len, 0.0f);

    if (torque_ctrl) {
        tau = action;
    }
    else {
        if (!gains_set)
        throw RobotSetGainsError("\033[31mRobot's kp and kd must be provided before do_action.\033[0m");
        //  - 0..5  : 다리 6관절 위치 제어
        //  - 6..7  : 바퀴 속도 제어
        for (size_t i = 0; i < last_action_len; ++i) {
            bool is_pos_idx = (i < 6);
            if (is_pos_idx) {
                // pos 인덱스 → _joint_names 인덱스 매핑
                const std::string& jname = joint_names[i];
                float off = pos_offset[jname];
                pos[i] = action[i] - off;
            } else {
                vel[i] = action[i]; // 바퀴는 속도 제어
            }
            kp_[i] = kp[i];
            kd_[i] = kd[i];
        }
    }

    auto slice = [](const std::vector<float>& v, size_t s, size_t e) {
        return std::vector<float>(v.begin()+s, v.begin()+e);
    };
    
    cli.operation_control(
        motor_ids,
        slice(pos, 0, 8), slice(vel, 0, 8),
        slice(kp_,  0, 8), slice(kd_,  0, 8),
        slice(tau, 0, 8)
    );

    last_action = action;
    last_torque_ctrl = torque_ctrl;
    
    if(safe){
        check_safety(0.175f, 0.261f, 6.28f, 8.16f);  // 10 deg. 15 deg. 2pi/s 2.6pi/s 
    }
}

// ------- Check Safety -------
void Robot::check_safety(float POS_SAFETY_MARGIN_RAD,
                         float POS_MARGIN_NEAR_LIMIT_RAD,
                         float VEL_THRESHOLD_NEAR_LIMIT_RAD_S,
                         float VEL_HARD_LIMIT_RAD_S) { 
   
    FxCliMap status = cli.status();
    auto [dis, emg, bat] = check_status(status, motor_ids); 

    bool disconn_flag = dis;
    bool emergency_flag = emg;
    bool battery_flag = bat;

    if (!disconn_flag) {
        disconn_count = 0;
    } else {
        ++disconn_count;
    }

    if (emergency_flag)
        estop("\033[31m[E-stop]\033[0m Emergency stop button was pressed", /*is_physical_estop=*/true);

    if (disconn_count >= max_disconn_count)
        estop("\033[31m[E-stop]\033[0m FxClient connection timeout");

    if (req_miss_count >= max_req_miss_count)
        estop("\033[31m[E-stop]\033[0m FxClient request timeout");

    if (battery_flag)
        estop("\033[31m[E-stop]\033[0m Battery low");

    // Positions: 6 joints (legs), Velocities: 8 joints (legs + wheels)
    const auto& joint_pos_vec = obs.at("dof_pos"); // size: 6
    const auto& joint_vel_vec = obs.at("dof_vel"); // size: 8
    const auto& proj_grav_vec = obs.at("proj_grav");

    for (size_t i = 0; i < joint_names.size(); ++i) {
        const std::string& joint_name = joint_names[i];

        const float joint_pos = joint_pos_vec.at(i);
        const float joint_vel = joint_vel_vec.at(i); // same index ordering assumed

        const float lower_safe_pos = rel_min_pos.at(joint_name) + POS_SAFETY_MARGIN_RAD;
        const float upper_safe_pos = rel_max_pos.at(joint_name) - POS_SAFETY_MARGIN_RAD;

        // 1) Position hard limit
        if (joint_pos < lower_safe_pos || joint_pos > upper_safe_pos) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "\033[31m[E-stop]\033[0m position limit exceeded on " << joint_name
                << " (pos=" << joint_pos << " rad, allowed ["
                << lower_safe_pos << ", " << upper_safe_pos << "])";
            estop(oss.str());
        }

        // 2) Global velocity hard limit
        if (std::fabs(joint_vel) > VEL_HARD_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "\033[31m[E-stop]\033[0m velocity limit exceeded on " << joint_name
                << " (vel=" << joint_vel << " rad/s, limit="
                << VEL_HARD_LIMIT_RAD_S << " rad/s)";
            estop(oss.str());
        }

        // 3) Too fast toward lower limit (리미트 근처 위치 + 음의 큰 속도)
        if (joint_pos <= lower_safe_pos + POS_MARGIN_NEAR_LIMIT_RAD &&
            joint_vel < -VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "\033[31m[E-stop]\033[0m excessive negative velocity near lower limit on "
                << joint_name
                << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
            estop(oss.str());
        }

        // 4) Too fast toward upper limit (리미트 근처 위치 + 양의 큰 속도)
        if (joint_pos >= upper_safe_pos - POS_MARGIN_NEAR_LIMIT_RAD &&
            joint_vel > VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "\033[31m[E-stop]\033[0m excessive positive velocity near upper limit on "
                << joint_name
                << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
            estop(oss.str());
        }
    }
    // 5) Fall detection
    const float pgz = proj_grav_vec[2];
    constexpr float kMinUprightFor60deg = -0.7f; // 45 deg.
    if(pgz > kMinUprightFor60deg){
        estop("\033[31m[E-stop]\033[0m robot posture is unstable");
    }
}

// Check STATUS Data
std::tuple<bool,bool,bool> Robot::check_status(const FxCliMap& status,
                                          const std::vector<uint8_t>& mids)
{
    bool disconn_flag   = false;
    bool emergency_flag = false;
    bool battery_flag = false;

    // 1) EMERGENCY 플래그 체크
    auto it_emg = status.find("EMERGENCY");
    if (it_emg != status.end()) {
        const auto& emg = it_emg->second;
        auto it_val = emg.find("value");
        if (it_val != emg.end() && it_val->second == "ON") {
            emergency_flag = true;
            return {disconn_flag, emergency_flag, battery_flag};
        }
    }
    // 2) ACK / STATUS 체크  
    auto it_ack = status.find("ACK");
    if (it_ack == status.end()) {
        disconn_flag = true;
        return {disconn_flag, emergency_flag, battery_flag};
    }
    const auto& ack = it_ack->second;
    auto it_status = ack.find("STATUS");
    if (it_status == ack.end() || it_status->second != "true") {
        disconn_flag = true;
    }
    // 3) 각 모터의 pattern & err 체크
    for (auto id : mids) {
        std::string key = "M" + std::to_string(id);  // "M1", "M2", ...
        auto it_m = status.find(key);

        const std::size_t idx = static_cast<std::size_t>(id - 1);

        if (it_m == status.end()) {
            if (idx < motor_pattern.size()) motor_pattern[idx] = "<missing>";
            if (idx < motor_err.size())     motor_err[idx]     = "<missing>";
            disconn_flag = true;
            break;
        }
        const auto& m = it_m->second;
        // pattern
        std::string pattern_str;
        auto it_pattern = m.find("pattern");
        if (it_pattern != m.end()) {
            pattern_str = it_pattern->second;
        } else {
            pattern_str = "<missing>";
            if (idx < motor_pattern.size()) motor_pattern[idx] = pattern_str;
            disconn_flag = true;
        }
        if (idx < motor_pattern.size())
            motor_pattern[idx] = pattern_str;
        // pattern 값이 "2"가 아니면 비정상
        if (pattern_str != "2") disconn_flag = true;
        // err
        std::string err_str;
        auto it_err = m.find("err");
        if (it_err != m.end()) {
            err_str = it_err->second;
        } else {
            err_str = "<missing>";
            if (idx < motor_err.size()) motor_err[idx] = err_str;
            disconn_flag = true;
        }
        if (idx < motor_err.size())
            motor_err[idx] = err_str;
        // err 값이 "None"이 아니면 비정상
        if (err_str != "None") disconn_flag = true;
    }

    // 4) Battery 정보 파싱 (V, SOC)
    auto it_batt = status.find("BATT");
    if (it_batt != status.end()) {
        const auto& batt = it_batt->second;
        auto it_v   = batt.find("V");
        auto it_soc = batt.find("SOC");
        if(it_v != batt.end()) battery_voltage = it_v->second;
        else disconn_flag = true;

        if(it_soc != batt.end()) battery_soc = it_soc->second;
        else disconn_flag = true;

        try {
            float soc_val = std::stof(battery_soc);
            if (soc_val < 2.0f) {  // 2% 미만이면 배터리 low로 처리
                battery_flag = true;
            }
        }
        catch (const std::exception&) {
            disconn_flag = true;
        }
    }
    else{
        disconn_flag = true;
    }

    return {disconn_flag, emergency_flag, battery_flag};
}

// ------- E-Stop API -------
[[noreturn]] void Robot::estop(const std::string& msg, bool is_physical_estop) {
    // debug
    auto print_motor_status = [&]() -> std::string {
        std::ostringstream oss;
        oss << "\n\n====== Last observed robot status ======\n";
        for (auto id : motor_ids) {
            std::size_t idx = static_cast<std::size_t>(id - 1);
            if (idx >= motor_pattern.size() || idx >= motor_err.size())
                continue;
            oss << "  M" << static_cast<int>(id)
                << " | pattern: " << motor_pattern[idx]
                << ", err: " << motor_err[idx] << "\n";
        }
        oss << "----------------------------------------\n";
        oss << "  Battery Voltage: " << battery_voltage << " V\n";
        oss << "  Battery SOC    : " << battery_soc     << " %\n";
        oss << "========================================\n";
        return oss.str();
    };
    const auto retry_ms = std::chrono::milliseconds(1);
    std::vector<float> estop_action(last_action_len, 0.0f);

    if(is_physical_estop){
        do_action(estop_action, true, false);
        cli.motor_estop(motor_ids); 
    }
    else{
        do_action(estop_action, true, false);
        for (;;) {
            bool ok = cli.motor_estop(motor_ids); 
            if (ok) break;
            else{
                do_action(estop_action, true, false);  // double check
                std::this_thread::sleep_for(retry_ms);
            }
        }
    }
    std::string estop_err_msg = msg.empty() ? "\033[31m[E-stop]\033[0m" : msg;
    estop_err_msg += print_motor_status();
    std::cout << estop_err_msg;
    throw RobotEStopError(estop_err_msg);
}
} // namespace robot
