#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "safety.hpp"

namespace safety {


// -------- Hip Joint Check --------
SafetyCheckResult checkHipJoint(
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
SafetyCheckResult checkShoulderJoint(
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
SafetyCheckResult checkLegJoint(
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
SafetyCheckResult checkFallDetection(float proj_grav_z)
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
SafetyCheckResult checkAllSafety(
    const mcu::McuStatus& status,
    const std::unordered_map<std::string, std::vector<float>>& obs)
{
    SafetyCheckResult result;
    result.level = SafetyCheckResult::Level::SAFE;
    result.is_physical_estop = false;
    
    // ======== EMERGENCY CHECKS (Priority 1) ========
    
    // 1. Emergency (physical) button check
    if (status.emergency) {
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
        std::size_t idx = robot::MID_TO_OBS_IDX[mid];

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
            joint_result = checkHipJoint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }
        else if (joint_name.find("shoulder") != std::string::npos) {
            joint_result = checkShoulderJoint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }
        else if (joint_name.find("leg") != std::string::npos) {
            joint_result = checkLegJoint(joint_name, joint_pos, joint_vel, rel_min, rel_max);
        }

        if (joint_result.level == SafetyCheckResult::Level::EMERGENCY) {
            return joint_result;
        }
    }

    // 3. Fall detection check
    const float pgz = proj_grav_vec[2];
    SafetyCheckResult fall_result = checkFallDetection(pgz);
    if (fall_result.level == SafetyCheckResult::Level::EMERGENCY) {
        return fall_result;
    }

    // ======== WARNING CHECKS (Priority 2) ========
    
    // 4. Disconnection check
    if (status.disconnected) {
        result.level = SafetyCheckResult::Level::WARNING;
        result.message = "\033[31m[WARNING]\033[0m FxClient connection timeout";
        return result;
    }

    // 6. Battery check
    if (status.battery_low) {
        result.level = SafetyCheckResult::Level::WARNING;
        result.message = "\033[31m[WARNING]\033[0m Battery low";
        return result;
    }

    return result;
}

} // namespace safety