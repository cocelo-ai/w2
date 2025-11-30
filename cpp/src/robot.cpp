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

namespace robot {

Robot::Robot()
    :
    last_action_(NUM_MOTORS, 0.0f),
    last_torque_ctrl_(false),

    kp_(NUM_MOTORS, 0.0f),  ///< P gains for PD control
    kd_(NUM_MOTORS, 0.0f),  ///< D gains for PD control
    gains_set_(false),

    motor_pattern_(NUM_MOTORS, "N"),  ///< N: init failed, 0: stop/idle, 1: calibration, 2: normal operation
    motor_err_(NUM_MOTORS, "None"),   ///< Motor error code/string (None: normal operation)

    battery_voltage_(""),
    battery_soc_(""),

    mcu_(MCU_IP, MCU_PORT, MCU_MOTOR_IDS, MCU_HAS_IMU, MCU_HAS_BAT, MCU_HAS_ESTOP)
{
    // Initialize observation containers for RL state
    obs_["dof_pos"]    = std::vector<float>(NUM_LIMB_MOTORS, 0.0f);    ///< 6 leg joint positions (no wheels)
    obs_["dof_vel"]    = std::vector<float>(NUM_MOTORS, 0.0f);         ///< 8 motor velocities (all motors)
    obs_["ang_vel"]    = std::vector<float>(3, 0.0f);                  ///< IMU angular velocity
    obs_["proj_grav"]  = std::vector<float>(3, 0.0f);                  ///< IMU (normalized) projected gravity
    obs_["lin_vel"]    = std::vector<float>(3, 0.0f);                  ///< Linear velocity (base) 
    obs_["height_map"] = std::vector<float>(HEIGHT_MAP_SIZE, 0.6128f); ///< Terrain heightmap

    // Wait for MCUs to be ready
    waitMcu(mcu_);
}

// ------- Wait All Nodes -------
void Robot::waitMcu(mcu::McuClient& m, std::int32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto retry_sleep = std::chrono::milliseconds(100);

    while (std::chrono::steady_clock::now() < deadline) {
         // Start motors
        bool started = m.motorStart();
        if (!(started)) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        // Check motor condition
        updateMcuObs(m);
        mcu::McuStatus status = m.fetchStatus();
        if (status.disconnected || (status.emergency && m.hasEstop()) || m.getReqMissCount() != 0) {
            std::this_thread::sleep_for(retry_sleep);
            continue;
        }
        if (status.battery_low && m.hasBat()) warn("\033[31m[Warning]\033[0m Battery low (SOC: " + status.battery_soc + "%)");

        // Initialize last_action with current joint positions
        auto& dof_pos = obs_.at("dof_pos");
        for (auto mid : m.motorIds()) {
            if (!isLimbMid(mid)) continue;

            const auto obs_idx = MID_TO_OBS_IDX[static_cast<std::size_t>(mid)];
            last_action_[obs_idx] = dof_pos[obs_idx];
        }
        last_torque_ctrl_ = false;
        return;
    }
    throw RobotInitError("Motor start timeout");
}

// Set PD control gains (required before position control)
void Robot::setGains(const std::vector<float>& kp, const std::vector<float>& kd) {
    auto throw_error = [](const std::string& msg) {
        throw RobotSetGainsError("\033[31m" + msg + "\033[0m");
    };

    // Validate sizes
    if (kp.size() != NUM_MOTORS) {
        throw_error(
            "set_gains: kp length mismatch for the robot. Expected " + std::to_string(NUM_MOTORS) + ", but got " + std::to_string(kp.size()));
    }
    if (kd.size() != NUM_MOTORS) {
        throw_error("set_gains: kd length mismatch for the robot. Expected " + std::to_string(NUM_MOTORS) + ", but got " + std::to_string(kd.size()));
    }

    // Wheels use velocity control, so kp must be zero for wheel indices
    if (kp[LEFT_WHEEL_IDX] != 0.0f || kp[RIGHT_WHEEL_IDX] != 0.0f) {
        throw_error(
            "set_gains: wheel motor kp must be zero for indices " +
            std::to_string(LEFT_WHEEL_IDX) + ", " + std::to_string(RIGHT_WHEEL_IDX) +
            ". But got kp[" + std::to_string(LEFT_WHEEL_IDX)  + "] = " + std::to_string(kp[LEFT_WHEEL_IDX])  +
            ", kp["         + std::to_string(RIGHT_WHEEL_IDX) + "] = " + std::to_string(kp[RIGHT_WHEEL_IDX])
        );
    }

    // All gains must be non-negative
    for (std::size_t i = 0; i < kp.size(); ++i) {
        if (kp[i] < 0.0f) {
            throw_error("set_gains: kp must be non-negative. But got kp[" + std::to_string(i) + "] = " + std::to_string(kp[i]));
        }
    }
    for (std::size_t i = 0; i < kd.size(); ++i) {
        if (kd[i] < 0.0f) {
            throw_error("set_gains: kd must be non-negative. But got kd[" + std::to_string(i) + "] = " + std::to_string(kd[i]));
        }
    }

    kp_ = kp;
    kd_ = kd;
    gains_set_ = true;
}

// Update lateste observation
void Robot::updateMcuObs(mcu::McuClient& m){
    auto& dof_pos   = obs_.at("dof_pos");   ///< size: NUM_LIMB_MOTORS
    auto& dof_vel   = obs_.at("dof_vel");   ///< size: NUM_MOTORS
    auto& ang_vel   = obs_.at("ang_vel");   ///< size: 3
    auto& proj_grav = obs_.at("proj_grav"); ///< size: 3

    mcu::McuObs mcu_obs = m.fetchObs();

    for (std::size_t i = 0; i < m.numMotors(); ++i) {
        auto mid = m.motorIds()[i];
        std::size_t obs_idx = MID_TO_OBS_IDX[mid];  

        if (isLimbMid(mid)) {
            std::string joint_name{ MID_TO_JOINT_NAMES[mid] };

            dof_pos[obs_idx] = mcu_obs.p[i] + POS_OFFSET[joint_name];
            dof_vel[obs_idx] = mcu_obs.v[i];
        } else {
            dof_vel[obs_idx] = mcu_obs.v[i];
        }   
    }

    if(m.hasImu()){
        ang_vel[0]   = mcu_obs.g[0];
        ang_vel[1]   = mcu_obs.g[1];
        ang_vel[2]   = mcu_obs.g[2];
        proj_grav[0] = mcu_obs.pg[0];
        proj_grav[1] = mcu_obs.pg[1];
        proj_grav[2] = mcu_obs.pg[2];    
    }   
}

// Get current observation 
std::unordered_map<std::string, std::vector<float>> Robot::getObs() {
    // Send action to get latest observation (MIT protocol motors require command to return state)
    doAction(last_action_, last_torque_ctrl_, false);
    updateMcuObs(mcu_); 
    return obs_;
}

// Do action (RL action → motor commands)
void Robot::doAction(const std::vector<float>& action, bool torque_ctrl, bool safe) {
    if (action.size() != NUM_MOTORS) {
        std::ostringstream oss;
        oss << "\033[31mdo_action: action length mismatch. " << "Expected " << NUM_MOTORS
            << ", but Got " << action.size() << "\033[0m";
        estop(oss.str());
    }

    std::vector<float> pos(mcu_.numMotors(), 0.0f);
    std::vector<float> vel(mcu_.numMotors(), 0.0f);
    std::vector<float> tau(mcu_.numMotors(), 0.0f);
    std::vector<float> kp(mcu_.numMotors(),  0.0f);
    std::vector<float> kd(mcu_.numMotors(),  0.0f);

    if (torque_ctrl) {
        // Direct torque control
        for (std::size_t i = 0; i < mcu_.numMotors(); ++i) {
            auto mid = mcu_.motorIds()[i];
            auto idx = MID_TO_OBS_IDX[mid];  
            tau[i] = action[idx];
        }
    }
    else {
        // PD position/velocity control
        if (!gains_set_)
        throw RobotSetGainsError("\033[31mRobot's kp and kd must be provided before do_action.\033[0m");

        for (std::size_t i = 0; i < mcu_.numMotors(); ++i) {
            auto mid  = mcu_.motorIds()[i];
            auto idx  = MID_TO_OBS_IDX[mid];

            if (isLimbMid(mid)) {
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
            kp[i] = kp_[idx];
            kd[i] = kd_[idx];
        }
    }

    // Send commands to motors
    mcu_.operationControl(pos, vel, kp, kd, tau);
        
    last_action_ = action;
    last_torque_ctrl_ = torque_ctrl;

    // Safety checks
    if(safe) checkSafety();
}

// Perform comprehensive safety checks
void Robot::checkSafety() {
    mcu::McuStatus status = mcu_.fetchStatus();

    // Update internal state
    battery_voltage_ = status.battery_voltage;
    battery_soc_     = status.battery_soc;

    for (std::size_t i = 0; i < mcu_.numMotors(); ++i) {
        auto mid  = mcu_.motorIds()[i];
        auto idx  = MID_TO_OBS_IDX[mid];
        motor_pattern_[idx] = status.motor_pattern[i]; 
        motor_err_[idx]     = status.motor_err[i];  
    }
    
    auto safety_result = safety::checkAllSafety(status, obs_);

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
    const auto retry_ms = std::chrono::milliseconds(1);
    std::vector<float> estop_action(NUM_MOTORS, 0.0f);

    if(is_physical_estop){
        // Physical emergency: stop immediately without retry
        doAction(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
        mcu_.motorEstop();
    }
    else{
        // Deliberately retry ESTOP in an unbounded loop until MCU succeed.
        // "fail-stop" behaviour is preferred
        doAction(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
        bool estop_finished = false;
        for (;;) {
            if(!estop_finished){
                bool front_ok = mcu_.motorEstop();
                if(front_ok) estop_finished = true;
            }
            if (estop_finished) break;
            else{
                doAction(estop_action, /*torque_ctrl=*/true, /*safe=*/false);
                std::this_thread::sleep_for(retry_ms);
            }
        }
    }
    std::string estop_err_msg = msg.empty() ? "\033[31m[E-stop]\033[0m" : msg;
    estop_err_msg += getMotorStatusString();
    std::cout << estop_err_msg;
    throw RobotEStopError(estop_err_msg);
}

// Warning message with throttled printing
void Robot::warn(const std::string& msg) {
    static const std::int32_t warning_print_interval = WARNING_PRINT_INTERVAL;
    static std::int32_t warning_count = 0;
    ++warning_count;

    // Print warning every WARNING_PRINT_INTERVAL
    bool should_print = (warning_count == 1) || (warning_count % warning_print_interval == 0);
    if (!should_print) {
        return;
    }
    std::string estop_err_msg = msg.empty() ? "\033[31m[WARNING]\033[0m" : msg;
    estop_err_msg += getMotorStatusString();
    std::cout << estop_err_msg;
}

// Motor status printing helper
std::string Robot::getMotorStatusString() const {
    std::ostringstream oss;
    oss << "\n\n====== Last Observed Status ======\n";
    for (auto id : MOTOR_IDS) {
        std::size_t idx = static_cast<std::size_t>(id - 1);
        if (idx >= motor_pattern_.size() || idx >= motor_err_.size())
            continue;
        oss << "  M" << static_cast<int>(id) << " | pattern: " << motor_pattern_[idx]
            << ", err: " << motor_err_[idx] << "\n";
    }
    oss << "----------------------------------------\n";
    oss << "  Battery Voltage: " << battery_voltage_ << " V\n";
    oss << "  Battery SOC    : " << battery_soc_    << " %\n";
    oss << "========================================\n";
    return oss.str();
}

} // namespace robot
