/**                                                                                       
                                                                     ▒██▓▒▓████▓░    
                                                                     ▓███████████░   
                                                                     ░▓██████████    
                                                                          ░██░ ███░  
                                                                          ░██   ▒█▒  
  ██████▒░█▓       ▓██░   ███   ███ ▓█░ ███  ░█▒ ▒██████▒  ███████       ░█░   ▒█▓   
  ██     ░█▓      ▒█░▒█░  ████ ▓███ ▓█░ █▓██ ░█▒░█▓       ▒█░    ██  ░████    ▒█     
  ██████░░█▓      ██▓▓██  ██░█ █▒██ ▓█░ █▓ ██▒█▒░█▓  ▒▓██ ▓█░    ██  ▓████  ████░    
  ██     ░██▒▒▒▒ ██    █▓ ██ ▒█▓ ██ ▓█░ █▓  ▓██▒ ▓██▒░▓██  ██▓░▒██░   ▒█▒   ████▒    
                                                                            ░██░     
███████████████████████████████████████████████████████████████████████████████████      
*/

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>

#include <fx_cli/fx_client.h> 
#include "robot.hpp"
#include "mcu.hpp"

namespace safety {

// ════════════════════════════════════════════════════════════════════════════
// Joint Position Limits
// ════════════════════════════════════════════════════════════════════════════

/// @brief Maximum joint positions (relative to calibration offset)
inline const std::unordered_map<std::string, float> REL_MAX_POS = {
    {"left_hip",       0.65f},
    {"right_hip",      0.65f},
    {"left_shoulder",  1.5f},
    {"right_shoulder", 1.5f},
    {"left_leg",       INFINITY},
    {"right_leg",      INFINITY}
};

/// @brief Minimum joint positions (relative to calibration offset)
inline const std::unordered_map<std::string, float> REL_MIN_POS = {
    {"left_hip",       -0.67f},
    {"right_hip",      -0.67f},
    {"left_shoulder",  -1.5f},
    {"right_shoulder", -1.5f},
    {"left_leg",       -0.9f},
    {"right_leg",      -0.9f}
};

// ════════════════════════════════════════════════════════════════════════════
// Joint Safety Parameters
// ════════════════════════════════════════════════════════════════════════════

constexpr float HIP_POS_SAFETY_MARGIN_RAD = 0.175f;
constexpr float HIP_POS_MARGIN_NEAR_LIMIT_RAD = 0.261f;
constexpr float HIP_VEL_THRESHOLD_NEAR_LIMIT_RAD_S = 6.28f;
constexpr float HIP_VEL_HARD_LIMIT_RAD_S = 8.16f;

constexpr float SHOULDER_POS_SAFETY_MARGIN_RAD = 0.175f;
constexpr float SHOULDER_POS_MARGIN_NEAR_LIMIT_RAD = 0.261f;
constexpr float SHOULDER_VEL_THRESHOLD_NEAR_LIMIT_RAD_S = 6.28f;
constexpr float SHOULDER_VEL_HARD_LIMIT_RAD_S = 8.16f;

constexpr float LEG_POS_SAFETY_MARGIN_RAD = 0.175f;
constexpr float LEG_POS_MARGIN_NEAR_LIMIT_RAD = 0.261f;
constexpr float LEG_VEL_THRESHOLD_NEAR_LIMIT_RAD_S = 6.28f;
constexpr float LEG_VEL_HARD_LIMIT_RAD_S = 8.16f;

// ════════════════════════════════════════════════════════════════════════════
// Fall Detection
// ════════════════════════════════════════════════════════════════════════════

constexpr float MIN_UPRIGHT_FOR_FALL_DETECTION = -0.766f;  // ≈ 50 Deg.

// ════════════════════════════════════════════════════════════════════════════
// Data Structures
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Hardware status check result
 * @details Contains connection state, battery info, and motor diagnostics
 */
struct StatusCheckResult {
    bool disconnected;
    bool emergency;
    bool battery_low;
    std::vector<std::string> motor_pattern;
    std::vector<std::string> motor_err;
    std::string battery_voltage;
    std::string battery_soc;
};

/**
 * @brief Safety violation severity and details
 */
struct SafetyCheckResult {
    enum class Level {
        SAFE,       ///< No violations detected
        WARNING,    ///< Non-critical safety violation
        EMERGENCY   ///< Immediate E-stop required
    };
    Level level;              ///< Violation severity
    std::string message;      ///< Descriptive error/warning message
    bool is_physical_estop;   ///< Triggered by physical E-stop button
};

// ════════════════════════════════════════════════════════════════════════════
// Safety Check Functions
// ════════════════════════════════════════════════════════════════════════════


SafetyCheckResult checkAllSafety(
    const mcu::McuStatus& status,
    const std::unordered_map<std::string, std::vector<float>>& obs
);

// ───────────────────────────────────────────────────────────────────────────
// Joint-Specific Checks
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Validates joint position and velocity
 * @param joint_name Joint identifier (e.g., "left_hip")
 * @param joint_pos Current position (rad)
 * @param joint_vel Current velocity (rad/s)
 * @param rel_min Minimum position limit
 * @param rel_max Maximum position limit
 * @return Safety check result
 */
SafetyCheckResult checkHipJoint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max
);

SafetyCheckResult checkShoulderJoint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max
);

SafetyCheckResult checkLegJoint(
    const std::string& joint_name,
    float joint_pos,
    float joint_vel,
    float rel_min,
    float rel_max
);

// ───────────────────────────────────────────────────────────────────────────
// Fall Detection
// ───────────────────────────────────────────────────────────────────────────

/**
 * @brief Detects robot fall via IMU gravity projection
 * @param proj_grav_z Projected gravity on Z-axis
 * @return EMERGENCY if fallen, SAFE otherwise
 */
SafetyCheckResult checkFallDetection(float proj_grav_z);

} // namespace safety