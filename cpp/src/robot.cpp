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

    cli_disconn_timeout_ms(100),
    cli_disconn_duration_ms(0),
    cli_missed_req(0),
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
        {"left_hip", 0.6f},   {"right_hip", 0.6f},
        {"left_shoulder", 1.4f}, {"right_shoulder", 1.4f},
        {"left_leg", 1.8f},  {"right_leg", 1.8f},
    };
    rel_min_pos = {
        {"left_hip", -0.6f},   {"right_hip", -0.6f},
        {"left_shoulder", -1.4f}, {"right_shoulder", -1.4f},
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
        auto [dis, emg] = check_status(status, motor_ids);
        if (dis || emg) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }

        FxCliMap mcu_obs = cli.req(motor_ids);
        parse_obs(mcu_obs);
        if(cli_missed_req > 1){
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
    if (kp_.size() != last_action_len)
        throw RobotSetGainsError("kp length mismatch for the robot.");
    if (kd_.size() != last_action_len)
        throw RobotSetGainsError("kd length mismatch for the robot.");
    if (kp_[6] != 0.0f || kp_[7] != 0.0f)
        throw RobotSetGainsError("Wheel motor kp must be zero for indices 6 and 7.");
    for (float v : kp_) if (v < 0.0f) throw RobotSetGainsError("kp must be non-negative.");
    for (float v : kd_) if (v < 0.0f) throw RobotSetGainsError("kd must be non-negative.");
    kp = kp_; kd = kd_; gains_set = true;
}

// ------- Parse Observation -------
std::unordered_map<std::string, std::vector<float>>& Robot::parse_obs(const FxCliMap& mcu_obs) {
    // 1) ACK / REQ check
    if (auto it_ack = mcu_obs.find("ACK"); it_ack != mcu_obs.end()) {
        const auto& ack = it_ack->second;
        auto it = ack.find("REQ");
        if (!(it != ack.end() && it->second == "true")) {
            ++cli_missed_req;
            return obs;
        }
    }
    else{
        ++cli_missed_req;
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
            ++cli_missed_req;
            return obs;
        }

        auto it_p = m->find("p");
        auto it_v = m->find("v");
        if (it_p == m->end() || it_v == m->end()) {
            ++cli_missed_req;
            return obs;
        }
        if (it_p->second == "N" || it_v->second == "N") {
            ++cli_missed_req;
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
            ++cli_missed_req;
            return obs;
        }

        auto it_v = m->find("v");
        if (it_v == m->end() || it_v->second == "N") {
            ++cli_missed_req;
            return obs;
        }
        float v = std::stof(it_v->second);
        dof_vel[i] = v;
    }

    // 5) Parse IMU
    auto it_imu = mcu_obs.find("IMU");
    if (it_imu == mcu_obs.end()) {
        ++cli_missed_req;
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
        ++cli_missed_req;
        return obs;
    }

    ang_vel[0]   = gx;
    ang_vel[1]   = gy;
    ang_vel[2]   = gz;
    proj_grav[0] = pgx;
    proj_grav[1] = pgy;
    proj_grav[2] = pgz;

    cli_missed_req = 0;
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
    if (action.size() != last_action_len)
        estop("action length mismatch.");

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
        throw RobotSetGainsError("Robot's kp and kd must be provided before do_action.");
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
        check_safety(0.175f, 0.261f, 6.28f, 7.85f);  // 10 deg. 15 deg. 2pi/s 2.5pi/s 
    }
}

// Check STATUS Data
std::pair<bool, bool> Robot::check_status(const FxCliMap& status,
                                          const std::vector<uint8_t>& mids)
{
    bool disconn_flag   = false;
    bool emergency_flag = false;

    // 1) EMERGENCY 플래그 체크
    auto it_emg = status.find("EMERGENCY");
    if (it_emg != status.end()) {
        const auto& emg = it_emg->second;
        auto it_val = emg.find("value");
        if (it_val != emg.end() && it_val->second == "ON") {
            emergency_flag = true;
        }
    }
    // 2) ACK / STATUS 체크  
    auto it_ack = status.find("ACK");
    if (it_ack == status.end()) {
        disconn_flag = true;
        return {disconn_flag, emergency_flag};
    }
    const auto& ack = it_ack->second;
    auto it_status = ack.find("STATUS");
    if (it_status == ack.end() || it_status->second != "true") {
        disconn_flag = true;
        return {disconn_flag, emergency_flag};
    }
    // 3) 각 모터 M1..Mn pattern / err 체크
    for (auto id : mids) {
        std::string key = "M" + std::to_string(id);  // "M1", "M2", ...
        auto it_m = status.find(key);
        if (it_m == status.end()) {
            disconn_flag = true;
            break;
        }
        const auto& m = it_m->second;
        auto it_pattern = m.find("pattern");
        // pattern 없거나, "2"가 아니면 비정상
        if (it_pattern == m.end() || it_pattern->second != "2") {
            disconn_flag = true;
            break;
        }
        auto it_err = m.find("err");
        if (it_err != m.end() && it_err->second != "None") {
            disconn_flag = true;
            break;
        }
    }
    return {disconn_flag, emergency_flag};
}

// ------- Check Saftey -------
void Robot::check_safety(float POS_SAFETY_MARGIN_RAD,
                         float POS_MARGIN_NEAR_LIMIT_RAD,
                         float VEL_THRESHOLD_NEAR_LIMIT_RAD_S,
                         float VEL_HARD_LIMIT_RAD_S) { 

    FxCliMap status = cli.status();
    auto [dis, emg] = check_status(status, motor_ids); 

    bool disconn_flag = dis;
    bool emergency_flag = emg;

    constexpr int kCliPeriodMs = 20;

    if (!disconn_flag) cli_disconn_duration_ms = 0;
    else cli_disconn_duration_ms += kCliPeriodMs;
                       
    if (emergency_flag)
        throw RobotEStopError("E-stop: Emergency button triggered");  // TODO: change

    if (cli_disconn_duration_ms >= cli_disconn_timeout_ms)
        throw RobotEStopError("E-stop: FxClient connection timeout");

    if (cli_missed_req * kCliPeriodMs >= cli_disconn_timeout_ms)
        throw RobotEStopError("E-stop: Request miss timeout");

    // Positions: 6 joints (legs), Velocities: 8 joints (legs + wheels)
    const auto& joint_pos_vec = obs.at("dof_pos"); // size: 6
    const auto& joint_vel_vec = obs.at("dof_vel"); // size: 8

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
            oss << "E-stop: position limit exceeded on " << joint_name
                << " (pos=" << joint_pos << " rad, allowed ["
                << lower_safe_pos << ", " << upper_safe_pos << "])";
            estop(oss.str());
        }

        // 2) Global velocity hard limit
        if (std::fabs(joint_vel) > VEL_HARD_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "E-stop: velocity limit exceeded on " << joint_name
                << " (vel=" << joint_vel << " rad/s, limit="
                << VEL_HARD_LIMIT_RAD_S << " rad/s)";
            estop(oss.str());
        }

        // 3) Too fast toward lower limit (리미트 근처 위치 + 음의 큰 속도)
        if (joint_pos <= lower_safe_pos + POS_MARGIN_NEAR_LIMIT_RAD &&
            joint_vel < -VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "E-stop: excessive negative velocity near lower limit on "
                << joint_name
                << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
            estop(oss.str());
        }

        // 4) Too fast toward upper limit (리미트 근처 위치 + 양의 큰 속도)
        if (joint_pos >= upper_safe_pos - POS_MARGIN_NEAR_LIMIT_RAD &&
            joint_vel > VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "E-stop: excessive positive velocity near upper limit on "
                << joint_name
                << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
            estop(oss.str());
        }
    }
}

// ------- E-Stop API -------
[[noreturn]] void Robot::estop(const std::string& msg) {
    std::vector<float> estop_action(last_action_len, 0.0f);
    do_action(estop_action, true, false);

    const auto retry = std::chrono::milliseconds(1);
    for (;;) {
        bool ok = cli.motor_estop(motor_ids); 
        if (ok) break;
        else{
            do_action(estop_action, true, false);  // double check
            std::this_thread::sleep_for(retry);
        }
    }
    throw RobotEStopError(msg.empty() ? "E-stop triggered" : msg);
}
} // namespace robot
