#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "robot.hpp"
#include "safety.hpp"

namespace safety {

// -------- Status Data Parsing --------
StatusCheckResult check_status_data(
    const FxCliMap& status,
    const std::vector<uint8_t>& motor_ids,
    bool has_battery,
    bool has_estop)
{
    StatusCheckResult result;
    result.disconnected = false;
    result.emergency = false;
    result.battery_low = false;
    result.motor_pattern.resize(robot::NUM_MOTORS, "N");
    result.motor_err.resize(robot::NUM_MOTORS, "None");
    result.battery_voltage = "";
    result.battery_soc = "";

    // 1) EMERGENCY flag check
    if(has_estop){
        auto it_emg = status.find("EMERGENCY");
        if (it_emg != status.end()) {
            const auto& emg = it_emg->second;
            auto it_val = emg.find("value");
            if (it_val != emg.end() && it_val->second == "ON") {
                result.emergency = true;
                return result;
            }
        }
    }

    // 2) ACK / STATUS check
    auto it_ack = status.find("ACK");
    if (it_ack == status.end()) {
        result.disconnected = true;
        return result;
    }
    const auto& ack = it_ack->second;
    auto it_status = ack.find("STATUS");
    if (it_status == ack.end() || it_status->second != "true") {
        result.disconnected = true;
    }

    // 3) Motor pattern & err check
    for (auto id : motor_ids) {
        std::string key = "M" + std::to_string(id);
        auto it_m = status.find(key);

        const std::size_t idx = static_cast<std::size_t>(id - 1);

        if (it_m == status.end()) {
            if (idx < result.motor_pattern.size()) result.motor_pattern[idx] = "<missing>";
            if (idx < result.motor_err.size())     result.motor_err[idx]     = "<missing>";
            result.disconnected = true;
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
            if (idx < result.motor_pattern.size()) result.motor_pattern[idx] = pattern_str;
            result.disconnected = true;
        }
        if (idx < result.motor_pattern.size())
            result.motor_pattern[idx] = pattern_str;

        if (pattern_str != "2") result.disconnected = true;

        // err
        std::string err_str;
        auto it_err = m.find("err");
        if (it_err != m.end()) {
            err_str = it_err->second;
        } else {
            err_str = "<missing>";
            if (idx < result.motor_err.size()) result.motor_err[idx] = err_str;
            result.disconnected = true;
        }
        if (idx < result.motor_err.size())
            result.motor_err[idx] = err_str;

        if (err_str != "None") result.disconnected = true;
    }

    // 4) Battery info parsing
    if(has_battery){
        auto it_batt = status.find("BATT");
        if (it_batt != status.end()) {
            const auto& batt = it_batt->second;
            auto it_v   = batt.find("V");
            auto it_soc = batt.find("SOC");

            if (it_v != batt.end()) result.battery_voltage = it_v->second;
            else result.disconnected = true;

            if (it_soc != batt.end()) result.battery_soc = it_soc->second;
            else result.disconnected = true;

            try {
                float soc_val = std::stof(result.battery_soc);
                if (soc_val < 2.0f) {
                    result.battery_low = true;
                }
            }
            catch (const std::exception&) {
                result.disconnected = true;
            }
        }
        else {
            result.disconnected = true;
        }
    }

    return result;
}

// -------- Hip Joint Check --------
SafetyCheckResult check_hip_joint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;

    const float lower_safe_pos = rel_min + HIP_POS_SAFETY_MARGIN_RAD;
    const float upper_safe_pos = rel_max - HIP_POS_SAFETY_MARGIN_RAD;

    // Position hard limit
    if (joint_pos < lower_safe_pos || joint_pos > upper_safe_pos) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m position limit exceeded on " << joint_name
            << " (pos=" << joint_pos << " rad, allowed ["
            << lower_safe_pos << ", " << upper_safe_pos << "])";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Global velocity hard limit
    if (std::fabs(joint_vel) > HIP_VEL_HARD_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m velocity limit exceeded on " << joint_name
            << " (vel=" << joint_vel << " rad/s, limit="
            << HIP_VEL_HARD_LIMIT_RAD_S << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward lower limit
    if (joint_pos <= lower_safe_pos + HIP_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel < -HIP_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive negative velocity near lower limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward upper limit
    if (joint_pos >= upper_safe_pos - HIP_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel > HIP_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive positive velocity near upper limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    return result;
}

// -------- Shoulder Joint Check --------
SafetyCheckResult check_shoulder_joint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;

    const float lower_safe_pos = rel_min + SHOULDER_POS_SAFETY_MARGIN_RAD;
    const float upper_safe_pos = rel_max - SHOULDER_POS_SAFETY_MARGIN_RAD;

    // Position hard limit
    if (joint_pos < lower_safe_pos || joint_pos > upper_safe_pos) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m position limit exceeded on " << joint_name
            << " (pos=" << joint_pos << " rad, allowed ["
            << lower_safe_pos << ", " << upper_safe_pos << "])";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Global velocity hard limit
    if (std::fabs(joint_vel) > SHOULDER_VEL_HARD_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m velocity limit exceeded on " << joint_name
            << " (vel=" << joint_vel << " rad/s, limit="
            << SHOULDER_VEL_HARD_LIMIT_RAD_S << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward lower limit
    if (joint_pos <= lower_safe_pos + SHOULDER_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel < -SHOULDER_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive negative velocity near lower limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward upper limit
    if (joint_pos >= upper_safe_pos - SHOULDER_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel > SHOULDER_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive positive velocity near upper limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    return result;
}

// -------- Leg Joint Check --------
SafetyCheckResult check_leg_joint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;

    const float lower_safe_pos = rel_min + LEG_POS_SAFETY_MARGIN_RAD;
    const float upper_safe_pos = rel_max - LEG_POS_SAFETY_MARGIN_RAD;

    // Position hard limit
    if (joint_pos < lower_safe_pos || joint_pos > upper_safe_pos) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m position limit exceeded on " << joint_name
            << " (pos=" << joint_pos << " rad, allowed ["
            << lower_safe_pos << ", " << upper_safe_pos << "])";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Global velocity hard limit
    if (std::fabs(joint_vel) > LEG_VEL_HARD_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m velocity limit exceeded on " << joint_name
            << " (vel=" << joint_vel << " rad/s, limit="
            << LEG_VEL_HARD_LIMIT_RAD_S << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward lower limit
    if (joint_pos <= lower_safe_pos + LEG_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel < -LEG_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive negative velocity near lower limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    // Too fast toward upper limit
    if (joint_pos >= upper_safe_pos - LEG_POS_MARGIN_NEAR_LIMIT_RAD &&
        joint_vel > LEG_VEL_THRESHOLD_NEAR_LIMIT_RAD_S) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "\033[31m[E-stop]\033[0m excessive positive velocity near upper limit on "
            << joint_name
            << " (pos=" << joint_pos << " rad, vel=" << joint_vel << " rad/s)";
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = oss.str();
        return result;
    }

    return result;
}

// -------- Fall Detection --------
SafetyCheckResult check_fall_detection(float proj_grav_z)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;

    if (proj_grav_z > MIN_UPRIGHT_FOR_FALL_DETECTION) {
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = "\033[31m[E-stop]\033[0m robot posture is unstable";
    }

    return result;
}

// -------- Main Safety Check --------
SafetyCheckResult check_all_safety(
    const StatusCheckResult& status_result,
    int disconn_count,
    int max_disconn_count,
    int req_miss_count,
    int max_req_miss_count,
    const std::unordered_map<std::string, std::vector<float>>& obs)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;

    // ======== EMERGENCY CHECKS (Priority 1) ========
    
    // 1. Emergency (physical) button check
    if (status_result.emergency) {
        result.level = SafetyCheckResult::Level::EMERGENCY;
        result.message = "\033[31m[E-stop]\033[0m Emergency stop button was pressed";
        result.is_physical_estop = true;
        return result;
    }

    // Get observation vectors for emergency checks
    const auto& joint_pos_vec = obs.at("dof_pos");
    const auto& joint_vel_vec = obs.at("dof_vel");
    const auto& proj_grav_vec = obs.at("proj_grav");

    // 2. Joint safety checks - iterate through all motors
    for (std::size_t mid = 1; mid <= robot::NUM_MOTORS; ++mid) {
        std::size_t idx = robot::MID_TO_IDX[mid];

        const char* joint_name_c = robot::MID_TO_JOINT_NAMES[mid];
        std::string joint_name{ joint_name_c };

        // Skip wheels (to be implemented)
        if (joint_name.find("wheel") != std::string::npos) {
            continue;
        }

        const float joint_pos = joint_pos_vec.at(idx);
        const float joint_vel = joint_vel_vec.at(idx);

        auto it_max = REL_MAX_POS.find(joint_name);
        auto it_min = REL_MIN_POS.find(joint_name);
        if (it_max == REL_MAX_POS.end() || it_min == REL_MIN_POS.end()) {
            continue;
        }
        float rel_min = it_min->second;
        float rel_max = it_max->second;

        SafetyCheckResult joint_result;
        if (joint_name.find("hip") != std::string::npos) {
            joint_result = check_hip_joint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }
        else if (joint_name.find("shoulder") != std::string::npos) {
            joint_result = check_shoulder_joint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }
        else if (joint_name.find("leg") != std::string::npos) {
            joint_result = check_leg_joint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }

        if (joint_result.level == SafetyCheckResult::Level::EMERGENCY) {
            return joint_result;
        }
    }

    // 3. Fall detection check
    const float pgz = proj_grav_vec[2];
    SafetyCheckResult fall_result = check_fall_detection(pgz);
    if (fall_result.level == SafetyCheckResult::Level::EMERGENCY) {
        return fall_result;
    }

    // ======== WARNING CHECKS (Priority 2) ========
    
    // 4. Disconnection check
    if (disconn_count >= max_disconn_count) {
        result.level = SafetyCheckResult::Level::WARNING;
        result.message = "\033[31m[WARNING]\033[0m FxClient connection timeout";
        return result;
    }

    // 5. Request miss check
    if (req_miss_count >= max_req_miss_count) {
        result.level = SafetyCheckResult::Level::WARNING;
        result.message = "\033[31m[WARNING]\033[0m FxClient request timeout";
        return result;
    }

    // 6. Battery check
    if (status_result.battery_low) {
        result.level = SafetyCheckResult::Level::WARNING;
        result.message = "\033[31m[WARNING]\033[0m Battery low";
        return result;
    }

    return result;
}

} // namespace safety