#include <array>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <sstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <cstring>

#include "robot.hpp"
#include "safety.hpp"

namespace robot {

Robot::Robot()
    :
    last_action(NUM_MOTORS, 0.0f),
    last_torque_ctrl(false),

    kp(NUM_MOTORS, 0.0f),  // P gains for PD control
    kd(NUM_MOTORS, 0.0f),  // D gains for PD control
    gains_set(false),

    motor_pattern(NUM_MOTORS, "N"),  // N: init failed, 0: stop/idle, 1: calibration, 2: normal operation
    motor_err(NUM_MOTORS, "None"),   // Motor error code/string (None: normal operation)

    battery_voltage(""),
    battery_soc(""),
    disconn_count(0),

    mcu(MCU_IP, MCU_PORT, MCU_MOTOR_IDS, MCU_HAS_IMU, MCU_HAS_BAT, MCU_HAS_ESTOP)
{
    // Initialize observation containers for RL state
    obs["dof_pos"] = std::vector<float>(NUM_LIMB_MOTORS, 0.0f);      // 12 leg joint positions (no wheels)
    obs["dof_vel"] = std::vector<float>(NUM_MOTORS, 0.0f);            // 16 motor velocities (all motors)
    obs["ang_vel"] = std::vector<float>(3, 0.0f);                     // IMU angular velocity
    obs["proj_grav"] = std::vector<float>(3, 0.0f);                   // IMU (normalized) projected gravity
    obs["lin_vel"] = std::vector<float>(3, 0.0f);                     // Linear velocity (base) 
    obs["height_map"] = std::vector<float>(HEIGHT_MAP_SIZE, 0.6128f); // Terrain heightmap

    // Wait for MCUs to be ready
    wait_mcu(mcu);
}

// ------- Wait All Nodes -------
void Robot::wait_mcu(McuClient& mcu, std::int32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto retry_sleep = std::chrono::milliseconds(100);

    while (std::chrono::steady_clock::now() < deadline) {
         // Start motors
        bool started = mcu.motor_start();
        if (!(started)) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        // Check motor status for errors
        FxCliMap status = mcu.status();

        auto status_result = safety::check_status_data(status, mcu.motor_ids, mcu.has_bat, mcu.has_estop);
        if (status_result.disconnected) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        if (status_result.emergency && mcu.has_estop) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        if (status_result.battery_low && mcu.has_bat) {
            warn("\033[31m[Warning]\033[0m Battery low (SOC: " + status_result.battery_soc + "%)");
        }

        // Get initial observation and set last_action to current position
        FxCliMap mcu_data = mcu.req();
        update_mcu_obs(mcu_data, mcu);

        if(mcu.req_miss_count > 0){
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }

        // Initialize last_action with current joint positions
        auto& dof_pos = obs.at("dof_pos");

        for (auto mid : mcu.motor_ids) {
            if (!is_limb_mid(mid)) {
                continue;
            }

            const auto idx = MID_TO_IDX[static_cast<std::size_t>(mid)];
            // idx is 0–11 for limb joints, which matches dof_pos idx
            last_action[idx] = dof_pos[idx];
        }

        last_torque_ctrl = false;
        return;
    }
    throw RobotInitError("Motor start timeout");
}

// Set PD control gains (required before position control)
void Robot::set_gains(const std::vector<float>& kp_, const std::vector<float>& kd_) {
    auto throw_error = [](const std::string& msg) {
        throw RobotSetGainsError("\033[31m" + msg + "\033[0m");
    };

    // Validate sizes
    if (kp_.size() != NUM_MOTORS) {
        throw_error(
            "set_gains: kp length mismatch for the robot. Expected " + std::to_string(NUM_MOTORS) + ", but got " + std::to_string(kp_.size()));
    }
    if (kd_.size() != NUM_MOTORS) {
        throw_error("set_gains: kd length mismatch for the robot. Expected " + std::to_string(NUM_MOTORS) + ", but got " + std::to_string(kd_.size()));
    }

    // Wheels use velocity control, so kp must be zero for wheel indices
    if (kp_[LEFT_WHEEL_IDX] != 0.0f || kp_[RIGHT_WHEEL_IDX] != 0.0f) {
        throw_error(
            "set_gains: wheel motor kp must be zero for indices " +
            std::to_string(LEFT_WHEEL_IDX) + ", " + std::to_string(RIGHT_WHEEL_IDX) +
            ". But got kp[" + std::to_string(LEFT_WHEEL_IDX)  + "] = " + std::to_string(kp_[LEFT_WHEEL_IDX])  +
            ", kp["         + std::to_string(RIGHT_WHEEL_IDX) + "] = " + std::to_string(kp_[RIGHT_WHEEL_IDX])
        );
    }

    // All gains must be non-negative
    for (std::size_t i = 0; i < kp_.size(); ++i) {
        if (kp_[i] < 0.0f) {
            throw_error("set_gains: kp must be non-negative. But got kp[" + std::to_string(i) + "] = " + std::to_string(kp_[i]));
        }
    }
    for (std::size_t i = 0; i < kd_.size(); ++i) {
        if (kd_[i] < 0.0f) {
            throw_error("set_gains: kd must be non-negative. But got kd[" + std::to_string(i) + "] = " + std::to_string(kd_[i]));
        }
    }

    kp = kp_;
    kd = kd_;
    gains_set = true;
}

// Parse motor and IMU data into RL observation format
std::unordered_map<std::string, std::vector<float>>& Robot::update_mcu_obs(const FxCliMap& mcu_data, McuClient& mcu){
    // Check if request was acknowledged
    if (auto it_ack = mcu_data.find("ACK"); it_ack != mcu_data.end()) {
        const auto& ack = it_ack->second;
        auto it = ack.find("REQ");
        if (!(it != ack.end() && it->second == "true")) {
            ++mcu.req_miss_count;
            return obs;
        }
    }
    else{
        ++mcu.req_miss_count;
        return obs;
    }

    // Get observation vector references
    auto& dof_pos   = obs.at("dof_pos");   // size: NUM_LIMB_MOTORS
    auto& dof_vel   = obs.at("dof_vel");   // size: NUM_MOTORS
    auto& ang_vel   = obs.at("ang_vel");   // size: 3
    auto& proj_grav = obs.at("proj_grav"); // size: 3

    // Helper to get motor data map
    auto get_motor = [&](size_t index)
        -> const std::unordered_map<std::string, std::string>*
    {
        std::string key = "M" + std::to_string(index); // "M1", "M2", ...
        auto it = mcu_data.find(key);
        if (it == mcu_data.end()) {
            return nullptr;
        }
        return &it->second;
    };

    // (1) Parse Motor data
    for (auto mid : mcu.motor_ids) {
        const auto* m = get_motor(static_cast<std::size_t>(mid));
        if (!m) {
            ++mcu.req_miss_count;
            return obs;
        }
        auto it_p = m->find("p");
        auto it_v = m->find("v");
        if (it_p == m->end() || it_v == m->end() || it_p->second == "N" || it_v->second == "N") {
            ++mcu.req_miss_count;
            return obs;
        }

        std::size_t idx = MID_TO_IDX[mid];  
        if (is_limb_mid(mid)) {
            float p = std::stof(it_p->second);
            float v = std::stof(it_v->second);

            std::string jname{ MID_TO_JOINT_NAMES[mid] };
            float off = POS_OFFSET[jname];

            dof_pos[idx] = p + off;
            dof_vel[idx] = v;
        } else {
            float v = std::stof(it_v->second);
            dof_vel[idx] = v;
        }
    }

    // (2) Parse IMU data
    if(mcu.has_imu){
        auto it_imu = mcu_data.find("IMU");
        if (it_imu == mcu_data.end()) {
            ++mcu.req_miss_count;
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

        // Extract angular velocity and projected gravity
        float gx, gy, gz, pgx, pgy, pgz;
        if (!get_imu("gx", gx)  ||
            !get_imu("gy", gy)  ||
            !get_imu("gz", gz)  ||
            !get_imu("pgx", pgx)||
            !get_imu("pgy", pgy)||
            !get_imu("pgz", pgz))
        {
            ++mcu.req_miss_count;
            return obs;
        }

        ang_vel[0]   = gx;
        ang_vel[1]   = gy;
        ang_vel[2]   = gz;
        proj_grav[0] = pgx;
        proj_grav[1] = pgy;
        proj_grav[2] = pgz;
    }

    mcu.req_miss_count = 0;
    return obs;
}

// Get current observation 
std::unordered_map<std::string, std::vector<float>> Robot::get_obs() {
    // Send action to get latest observation (MIT protocol motors require command to return state)
    do_action(last_action, last_torque_ctrl, false);
    FxCliMap mcu_data = mcu.req();

    auto& mcu_obs = update_mcu_obs(mcu_data, mcu);  ///< update mcu obs
    return mcu_obs;
}

// Do action (RL action → motor commands)
void Robot::do_action(const std::vector<float>& action, bool torque_ctrl, bool safe) {
    if (action.size() != NUM_MOTORS) {
        std::ostringstream oss;
        oss << "\033[31mdo_action: action length mismatch. "
            << "Expected " << NUM_MOTORS
            << ", but Got " << action.size()
            << "\033[0m";
        estop(oss.str());
    }

    std::vector<float> pos(mcu.num_motors, 0.0f);
    std::vector<float> vel(mcu.num_motors, 0.0f);
    std::vector<float> tau(mcu.num_motors, 0.0f);
    std::vector<float> kp_(mcu.num_motors,  0.0f);
    std::vector<float> kd_(mcu.num_motors,  0.0f);


    if (torque_ctrl) {
        // Direct torque control
        for (std::size_t i = 0; i < mcu.num_motors; ++i) {
            auto mid = mcu.motor_ids[i];
            auto idx = MID_TO_IDX[mid];  
            tau[i] = action[idx];
        }
    }
    else {
        // PD position/velocity control
        if (!gains_set)
        throw RobotSetGainsError("\033[31mRobot's kp and kd must be provided before do_action.\033[0m");

        for (std::size_t i = 0; i < mcu.num_motors; ++i) {
            auto mid  = mcu.motor_ids[i];
            auto idx  = MID_TO_IDX[mid];

            if (is_limb_mid(mid)) {
                // Limbs -> position control (need offset)
                std::string jname{ MID_TO_JOINT_NAMES[mid] };
                float off  = POS_OFFSET[jname];
                pos[i]     = action[idx] - off;
                vel[i]     = 0.0f;
            } else {
                // wheels -> velocity control
                pos[i] = 0.0f;
                vel[i] = action[idx];
            }
            kp_[i] = kp[idx];
            kd_[i] = kd[idx];
        }
    }

    // Send commands to motors
    mcu.operation_control(
        pos, vel,
        kp_, kd_,
        tau
    );
        
    last_action = action;
    last_torque_ctrl = torque_ctrl;

    // Safety checks
    if(safe){
        check_safety();
    }
    
}

// Perform comprehensive safety checks
void Robot::check_safety() {
    FxCliMap status = mcu.status();

    auto status_result = safety::check_status_data(status, mcu.motor_ids, mcu.has_bat, mcu.has_estop);

    // Update internal state
    battery_voltage = status_result.battery_voltage;
    battery_soc     = status_result.battery_soc;

    for (auto mid : mcu.motor_ids) {
        std::size_t idx = static_cast<std::size_t>(mid - 1); 
        motor_pattern[idx] = status_result.motor_pattern[idx];
        motor_err[idx]     = status_result.motor_err[idx];
    }

    // Track disconnection count
    bool disconn_flag = (status_result.disconnected);
    if (!disconn_flag) {
        disconn_count = 0;
    } else {
        ++disconn_count;
    }

    // Run all safety checks (limits, errors, battery, etc.)
    int req_miss_count = mcu.req_miss_count;
    auto safety_result = safety::check_all_safety(
        status_result,
        disconn_count, MAX_DISCONN_COUNT,
        req_miss_count, MAX_REQ_MISS_COUNT,
        obs
    );

    // Handle safety violations
    if (safety_result.level == safety::SafetyCheckResult::Level::EMERGENCY) {
        estop(safety_result.message, safety_result.is_physical_estop);
    } else if (safety_result.level == safety::SafetyCheckResult::Level::WARNING) {
        warn(safety_result.message);
    }
}

// Emergency stop: halt all motors immediately
[[noreturn]] void Robot::estop(const std::string& msg, bool is_physical_estop) {
    // Print motor status for debugging
    auto print_motor_status = [&]() -> std::string {
        std::ostringstream oss;
        oss << "\n\n====== Last observed robot status ======\n";
        for (auto id : MOTOR_IDS) {
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
    std::vector<float> estop_action(NUM_MOTORS, 0.0f);

    if(is_physical_estop){
        // Physical emergency: stop immediately without retry
        do_action(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
        mcu.motor_estop();
    }
    else{
        // Software emergency: retry until confirmed
        do_action(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
        bool estop_finished = false;
        for (;;) {
            if(!estop_finished){
                bool front_ok = mcu.motor_estop();
                if(front_ok) estop_finished = true;
            }
            if (estop_finished) break;
            else{
                do_action(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
                std::this_thread::sleep_for(retry_ms);
            }
        }
    }
    std::string estop_err_msg = msg.empty() ? "\033[31m[E-stop]\033[0m" : msg;
    estop_err_msg += print_motor_status();
    std::cout << estop_err_msg;
    throw RobotEStopError(estop_err_msg);
}

// Warning message with throttled printing
void Robot::warn(const std::string& msg) {
    static std::int32_t warning_count = 0;
    static const std::int32_t warning_print_interval = WARNING_PRINT_INTERVAL;

    ++warning_count;

    // Print warning every WARNING_PRINT_INTERVAL
    bool should_print = (warning_count == 1) || (warning_count % warning_print_interval == 0);
    if (!should_print) {
        return;
    }

    auto print_motor_status = [&]() -> std::string {
        std::ostringstream oss;
        oss << "\n\n====== Last observed robot status ======\n";
        for (auto id : MOTOR_IDS) {
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
    std::string estop_err_msg = msg.empty() ? "\033[31m[WARNING]\033[0m" : msg;
    estop_err_msg += print_motor_status();
    std::cout << estop_err_msg;
}

} // namespace robot