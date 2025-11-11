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
    kp(last_action_len, 0.0f),
    kd(last_action_len, 0.0f),
    gains_set(false),
    motor_ids{1u,2u,3u,4u,5u,6u,7u,8u},
    cli_disconn_timeout_ms(200),
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

    patt_in_status_str.assign(last_action_len + 1, 0);
    emergency_in_status_str = 0;

    p_in_mcu_str.assign(last_action_len + 1, 0);
    v_in_mcu_str.assign(last_action_len + 1, 0);
    t_in_mcu_str.assign(last_action_len + 1, 0);
    gx_in_mcu_str  = 0;
    gy_in_mcu_str  = 0;
    gz_in_mcu_str  = 0;
    pgx_in_mcu_str = 0;
    pgy_in_mcu_str = 0;
    pgz_in_mcu_str = 0;

    // Offsets & limits
    pos_offset = {
        {"left_hip", 0.0f}, {"right_hip", 0.0f},
        {"left_shoulder", 0.0f}, {"right_shoulder", 0.0f},
        {"left_leg", 0.0f}, {"right_leg", 0.0f},
    };
    rel_max_pos = {
        {"left_hip", 3.14f}, {"right_hip", 3.14f},
        {"left_shoulder", 3.14f}, {"right_shoulder", 3.14f},
        {"left_leg", 3.14f}, {"right_leg", 3.14f},
    };
    rel_min_pos = {
        {"left_hip", -3.14f}, {"right_hip", -3.14f},
        {"left_shoulder", -3.14f}, {"right_shoulder", -3.14f},
        {"left_leg", -3.14f}, {"right_leg", -3.14f},
    };
    joint_names = { // pos 인덱스 0..5에 해당하는 관절 이름
        "left_hip","right_hip","left_shoulder","right_shoulder","left_leg","right_leg",
    };

    wait(); // 보드 준비 대기
}

// HW 준비 대기
void Robot::wait(std::int32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto retry_sleep = std::chrono::milliseconds(100);

    while (std::chrono::steady_clock::now() < deadline) {        
        bool started = cli.motor_start(motor_ids);
        if (!(started)) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        
        std::string status_str = cli.status();
        if (status_str.find("OK <STATUS>") == std::string::npos){
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }

        cache_status_values(status_str, patt_in_status_str, emergency_in_status_str);
        auto [dis, emg] = check_status(status_str, motor_ids);
        if (dis || emg) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }

        std::string mcu_str = cli.req(motor_ids);
        if(mcu_str.find("OK <REQ>") == std::string::npos){
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        else{
            // ─────────────────────────────────────────────
            // (A) 모터 값 검증
            // ─────────────────────────────────────────────
            bool motor_ok = true;
            const char* mid_keys[] = {"M1:","M2:","M3:","M4:","M5:","M6:","M7:","M8:"};
            for (const char* mid : mid_keys) {
                size_t mid_pos = mcu_str.find(mid);
                if (mid_pos == std::string::npos) { motor_ok=false; break; }
                for (const char* key : {"p:","v:","t:"}) {
                    size_t pos = mcu_str.find(key, mid_pos);
                    if (pos == std::string::npos) { motor_ok=false; break; }
                    if (mcu_str.substr(pos + std::strlen(key), 1) == "N") { motor_ok=false; break; }
                }
            }
            if(!motor_ok){
                std::this_thread::sleep_for(retry_sleep);
                continue;
            }
            // ─────────────────────────────────────────────
            // (B) IMU 값 검증
            // ─────────────────────────────────────────────
            std::size_t imu_pos = mcu_str.rfind("IMU");
            if (imu_pos == std::string::npos) {
                std::this_thread::sleep_for(retry_sleep);
                continue;
            }
            bool imu_ok = true;
            const char* imu_keys[] = {"r:", "p:", "y:", "gx:", "gy:", "gz:", "pgx:", "pgy:", "pgz:"};
            for (const char* key : imu_keys) {
                size_t pos = mcu_str.find(key, imu_pos);
                if (mcu_str.substr(pos + std::strlen(key), 1) == "N") {
                    imu_ok = false;
                    break;
                }
            }
            if (!imu_ok) {
                std::this_thread::sleep_for(retry_sleep);
                continue;
            }
        }
        cache_motor_values(mcu_str, p_in_mcu_str, v_in_mcu_str, t_in_mcu_str);
        cache_imu_values  (mcu_str, gx_in_mcu_str, gy_in_mcu_str, gz_in_mcu_str, pgx_in_mcu_str, pgy_in_mcu_str);
        cache_status_values(status_str, patt_in_status_str, emergency_in_status_str);
        return;
    }
    throw RobotInitError("Motor start timeout");
}

void Robot::cache_motor_values(const std::string& str,
                               std::vector<std::size_t>& ppos,
                               std::vector<std::size_t>& vpos,
                               std::vector<std::size_t>& tpos)
{
    std::fill(ppos.begin(), ppos.end(), 0);
    std::fill(vpos.begin(), vpos.end(), 0);
    std::fill(tpos.begin(), tpos.end(), 0);

    const std::size_t n = str.size();
    std::size_t cur = 0;
    while (cur < n) {
        std::size_t m = str.find('M', cur);
        if (m == std::string::npos) break;

        // parse motor id number
        std::size_t p = m + 1;
        unsigned int num = 0;
        bool has_digit = false;
        while (p < n && std::isdigit(static_cast<unsigned char>(str[p]))) {
            has_digit = true;
            num = num * 10u + static_cast<unsigned int>(str[p] - '0');
            ++p;
        }

        if (has_digit && num != 0 && num <= last_action_len) {
            // find keys after this 'M#'
            std::size_t pk = str.find("p:", m);
            if (pk != std::string::npos) ppos[num] = pk + 2;
            std::size_t vk = str.find("v:", m);
            if (vk != std::string::npos) vpos[num] = vk + 2;
            std::size_t tk = str.find("t:", m);
            if (tk != std::string::npos) tpos[num] = tk + 2;
        }
        cur = p;
    }
}

void Robot::cache_imu_values(const std::string& str,
                             std::size_t& gx, std::size_t& gy, std::size_t& gz,
                             std::size_t& pgx, std::size_t& pgy)
{
    std::size_t imu_base = str.rfind("IMU");
    if (imu_base == std::string::npos) {
        gx = gy = gz = pgx = pgy = pgz_in_mcu_str = 0;
        return;
    }

    std::size_t pos;

    pos = str.find("gx:", imu_base);  gx  = (pos == std::string::npos)  ? 0 : pos + 3;
    pos = str.find("gy:", imu_base);  gy  = (pos == std::string::npos)  ? 0 : pos + 3;
    pos = str.find("gz:", imu_base);  gz  = (pos == std::string::npos)  ? 0 : pos + 3;
    pos = str.find("pgx:", imu_base); pgx = (pos == std::string::npos)  ? 0 : pos + 4;
    pos = str.find("pgy:", imu_base); pgy = (pos == std::string::npos)  ? 0 : pos + 4;
    pos = str.find("pgz:", imu_base); pgz_in_mcu_str = (pos == std::string::npos) ? 0 : pos + 4;
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

void Robot::cache_status_values(const std::string& status_str,
                                std::vector<std::size_t>& patt_vec,
                                std::size_t& emergency_pos)
{
    for (auto id : motor_ids) {
        const std::string mid = "M" + std::to_string(id) + ":";
        std::size_t base = status_str.find(mid);
        std::size_t kpos = status_str.find("pattern:", base);
        patt_vec[id] = kpos + 8;  
    }
    std::size_t epos = status_str.find("EMERGENCY:");
    emergency_pos = epos + 10;   
}

// ------- Observation (returns copy; internal buffers reused) -------
std::unordered_map<std::string, std::vector<float>> Robot::get_obs() {
    std::string mcu_str = cli.req(motor_ids);
    auto& parsed = parse_obs(mcu_str);
    return parsed;
}

// ------- Action -------
void Robot::do_action(const std::vector<float>& action, bool torque_ctrl) {
    if (!gains_set)
        throw RobotSetGainsError("Robot's kp and kd must be provided before do_action.");
    if (action.size() != last_action_len)
        estop("action length mismatch.");

    std::vector<float> pos(last_action_len, 0.0f);
    std::vector<float> vel(last_action_len, 0.0f);
    std::vector<float> tau(last_action_len, 0.0f);
    std::vector<float> kp_(last_action_len, 0.0f);
    std::vector<float> kd_(last_action_len, 0.0f);

    if (torque_ctrl) {
        tau = action;
    } else {
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
    
    check_safety();
}
// ------- E-Stop API -------
[[noreturn]] void Robot::estop(const std::string& msg) {
    const auto retry = std::chrono::milliseconds(10);
    for (;;) {
        bool ok = cli.motor_estop(motor_ids); 
        if (ok) break;
        std::this_thread::sleep_for(retry);
    }
    throw RobotEStopError(msg.empty() ? "E-stop triggered" : msg);
}
// ------- Safety check -------
void Robot::check_safety() { 
    std::string status_str = cli.status();
    auto [dis, emg] = check_status(status_str, motor_ids); 

    bool disconn_flag = dis;
    bool emergency_flag = emg;

    if (!disconn_flag) cli_disconn_duration_ms = 0;
    else cli_disconn_duration_ms += 20;

    if (emergency_flag || std::max(cli_disconn_duration_ms, cli_missed_req * 20) >= cli_disconn_timeout_ms)
        throw RobotEStopError("E-stop: connection timeout or emergency flag reported");

    check_obs(obs);
}

std::pair<bool,bool> Robot::check_status(const std::string& status_str, const std::vector<uint8_t>& mids) {
    bool disconn_flag = false;
    bool emergency_flag = false;

    if (status_str.find("OK <STATUS>") == std::string::npos) {
        disconn_flag = true;
    } else {
        for (auto id : mids) {
            std::size_t idx = patt_in_status_str[id];
            char c0 = status_str[idx];        
            if (!std::isdigit(static_cast<unsigned char>(c0))) { disconn_flag = true; break; }
            int pattern = std::stoi(status_str.substr(idx));
            if (pattern != 2) { disconn_flag = true; break; }
        }
        emergency_flag = (status_str.substr(emergency_in_status_str, 2) == "ON");
    }
    return {disconn_flag, emergency_flag};
}

bool Robot::check_mcu(const std::string& mcu_str, const std::vector<uint8_t>& mids) {
    if (mcu_str.find("OK <REQ>") == std::string::npos) {
        cli_missed_req += 1;
        return false;
    }

    for (std::size_t id : mids) {
        if (id == 0 || id > last_action_len) { cli_missed_req += 1; return false; }
        std::size_t ps = p_in_mcu_str[id];
        std::size_t vs = v_in_mcu_str[id];
        std::size_t ts = t_in_mcu_str[id];
        if (ps < mcu_str.size() && mcu_str[ps] == 'N') { cli_missed_req += 1; return false; }
        if (vs < mcu_str.size() && mcu_str[vs] == 'N') { cli_missed_req += 1; return false; }
        if (ts < mcu_str.size() && mcu_str[ts] == 'N') { cli_missed_req += 1; return false; }
    }
    return true;
}

// Parse obs 
std::unordered_map<std::string, std::vector<float>>& Robot::parse_obs(const std::string& mcu_str) {
    if (!check_mcu(mcu_str, motor_ids)) {
        return obs;
    }
    auto& dof_pos   = obs["dof_pos"]; // 6
    auto& dof_vel   = obs["dof_vel"]; // 8
    auto& ang_vel   = obs["ang_vel"];
    auto& proj_grav = obs["proj_grav"];
    
    // ---- 위치: M1..M6 -> dof_pos[0..5] ----
    for (int i = 0; i < 6; ++i) {
        int mid = i + 1; // 1..6
        std::size_t pstart = p_in_mcu_str[mid];
        float val = std::stof(mcu_str.substr(pstart));
        dof_pos[i] = val + pos_offset[joint_names[i]];
    }

    // ---- 속도: M1..M8 -> dof_vel[0..7] ----
    for (int i = 0; i < 8; ++i) {
        int mid = i + 1; // 1..8
        std::size_t vstart = v_in_mcu_str[mid];
        float val = std::stof(mcu_str.substr(vstart));
        dof_vel[i] = val;
    }

    // ---- IMU ----
    ang_vel[0]   = std::stof(mcu_str.substr(gx_in_mcu_str));
    ang_vel[1]   = std::stof(mcu_str.substr(gy_in_mcu_str));
    ang_vel[2]   = std::stof(mcu_str.substr(gz_in_mcu_str));
    proj_grav[0] = std::stof(mcu_str.substr(pgx_in_mcu_str));
    proj_grav[1] = std::stof(mcu_str.substr(pgy_in_mcu_str));
    proj_grav[2] = std::stof(mcu_str.substr(pgz_in_mcu_str));
    
    return obs;
}
// ------- Check obs -------
void Robot::check_obs(const std::unordered_map<std::string, std::vector<float>>& obs) {  
    const auto& q_obs = obs.at("dof_pos"); // 6
    const auto& q_vel = obs.at("dof_vel"); // 8

    const float pos_margin = 0.1745f; // 10 deg
    const float vel_margin = 0.3491f; // 20 deg
    const float vel_th = 8.7275f; // rad/s

    for (size_t i=0;i<joint_names.size();++i) {
        const std::string& name = joint_names[i];

        float pos = q_obs.at(i);
        float vel = q_vel.at(i);

        float lo_pos = rel_min_pos.at(name) + pos_margin;
        float hi_pos = rel_max_pos.at(name) - pos_margin;

        if (pos < lo_pos || pos > hi_pos) {
            std::ostringstream oss; oss<<std::fixed<<std::setprecision(3);
            oss << "E-stop: position limit exceeded on " << name
                << " (pos=" << pos << " rad, allowed [" << lo_pos << ", " << hi_pos << "])";
            estop(oss.str()); 
        }
        if (pos < lo_pos + vel_margin && vel < -vel_th) {
            std::ostringstream oss; oss<<std::fixed<<std::setprecision(3);
            oss << "E-stop: excessive negative velocity near lower limit on " << name
                << " (pos=" << pos << " rad, vel=" << vel << " rad/s)";
            estop(oss.str());  
        }
        if (pos >= hi_pos - vel_margin && vel > vel_th) {
            std::ostringstream oss; oss<<std::fixed<<std::setprecision(3);
            oss << "E-stop: excessive positive velocity near upper limit on " << name
                << " (pos=" << pos << " rad, vel=" << vel << " rad/s)";
            estop(oss.str()); 
        }
    }
}

} // namespace robot
