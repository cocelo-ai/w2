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

#pragma once  // robot.hpp

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <fx_cli/fx_client.h> 
#include "mcu.hpp"   
#include "safety.hpp"

namespace robot {

// ════════════════════════════════════════════════════════════════════════════
// Motor Configuration 
// ════════════════════════════════════════════════════════════════════════════

inline const std::vector<uint8_t>       MOTOR_IDS           = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};  ///< All registerd motor IDs 
constexpr std::size_t                   NUM_MOTORS          = 8;                                 ///< Total number of motors
constexpr std::size_t                   NUM_LIMB_MOTORS     = 6;                                 ///< Number of limb  motors (excluding wheels)
inline const std::array<std::size_t, 2> WHEEL_MIDS          = {7, 8};                            ///< Motor IDs for wheel actuators

// ════════════════════════════════════════════════════════════════════════════
// MCU Configuration
// ════════════════════════════════════════════════════════════════════════════

inline const std::vector<uint8_t> MCU_MOTOR_IDS  = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};  
inline const std::string          MCU_IP         = "192.168.10.10";
constexpr int                     MCU_PORT       = 5101;
constexpr bool                    MCU_HAS_IMU    = true; 
constexpr bool                    MCU_HAS_BAT    = true; 
constexpr bool                    MCU_HAS_ESTOP  = true; 

// ──────────────────────────────────────────────────────────────────────────
// Joint Index Mapping for Observation and Action Vectors
// ──────────────────────────────────────────────────────────────────────────

constexpr std::size_t LEFT_HIP_IDX       = 0;   ///< Left hip joint index
constexpr std::size_t RIGHT_HIP_IDX      = 1;   ///< Right hip joint index
constexpr std::size_t LEFT_SHOULDER_IDX  = 2;   ///< Left shoulder joint index
constexpr std::size_t RIGHT_SHOULDER_IDX = 3;   ///< Right shoulder joint index
constexpr std::size_t LEFT_LEG_IDX       = 4;   ///< Left leg joint index
constexpr std::size_t RIGHT_LEG_IDX      = 5;   ///< Right leg joint index
constexpr std::size_t LEFT_WHEEL_IDX     = 6;   ///< Left wheel motor index
constexpr std::size_t RIGHT_WHEEL_IDX    = 7;   ///< Right wheel motor index

// ──────────────────────────────────────────────────────────────────────────
// Motor ID Mapping Tables (MID → Joint Index)
// ──────────────────────────────────────────────────────────────────────────

inline constexpr std::array<std::size_t, NUM_MOTORS + 1> MID_TO_OBS_IDX = {
    0,                   // [0]  unused                        OBS IDX
    LEFT_HIP_IDX,        // [1]  Motor ID 1  → left hip        (= 0 )
    RIGHT_HIP_IDX,       // [2]  Motor ID 2  → right hip       (= 1 )
    LEFT_SHOULDER_IDX,   // [3]  Motor ID 3  → left shoulder   (= 2 )
    RIGHT_SHOULDER_IDX,  // [4]  Motor ID 4  → right shoulder  (= 3 )
    LEFT_LEG_IDX,        // [5]  Motor ID 5  → left leg        (= 4 )
    RIGHT_LEG_IDX,       // [6]  Motor ID 6  → right leg       (= 5 )
    LEFT_WHEEL_IDX,      // [7]  Motor ID 7  → left wheel      (= 6 )
    RIGHT_WHEEL_IDX,     // [8]  Motor ID 8  → right wheel     (= 7 )
};

/**
 * @brief Maps motor IDs to human-readable joint names
 * @note Index 0 is unused as motor IDs start from 1
 */
inline constexpr std::array<const char*, NUM_MOTORS + 1> MID_TO_JOINT_NAMES = {
    "unused",         // [0]  unused
    "left_hip",       // [1]  Motor ID 1  → "left_hip"
    "right_hip",      // [2]  Motor ID 2  → "right_hip"
    "left_shoulder",  // [3]  Motor ID 3  → "left_shoulder"
    "right_shoulder", // [4]  Motor ID 4  → "right_shoulder"
    "left_leg",       // [5]  Motor ID 5  → "left_leg"
    "right_leg",      // [6]  Motor ID 6  → "right_leg"
    "left_wheel",     // [7]  Motor ID 7  → "left_wheel"
    "right_wheel",    // [8]  Motor ID 8  → "right_wheel"
};

// ──────────────────────────────────────────────────────────────────────────
// Joint Calibration
// ──────────────────────────────────────────────────────────────────────────

/**
 * @brief Position offset calibration values for each joint
 * @note Wheels are not included as they don't require position calibration
 */
inline std::unordered_map<std::string, float> POS_OFFSET = {
    {"left_hip",       0.0f}, 
    {"right_hip",      0.0f},
    {"left_shoulder",  0.0f}, 
    {"right_shoulder", 0.0f},
    {"left_leg",       0.0f}, 
    {"right_leg",      0.0f},
};

// ════════════════════════════════════════════════════════════════════════════
// Height-map Configuration Constants
// ════════════════════════════════════════════════════════════════════════════

constexpr std::size_t HEIGHT_MAP_SIZE = 144;  

// ──────────────────────────────────────────────────────────────────────────
// Safety Thresholds
// ──────────────────────────────────────────────────────────────────────────

constexpr int MAX_DISCONN_COUNT  = 5;  ///< Max consecutive disconnections before E-stop
constexpr int MAX_REQ_MISS_COUNT = 5;  ///< Max missed requests before E-stop

/// @brief Interval for printing repeated warnings (in control cycles)
constexpr std::int32_t WARNING_PRINT_INTERVAL = 200;

// ════════════════════════════════════════════════════════════════════════════
// Exception Types
// ════════════════════════════════════════════════════════════════════════════

/// @brief Exception thrown when emergency stop is triggered
struct RobotEStopError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

/// @brief Exception thrown during robot initialization failures
struct RobotInitError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

/// @brief Exception thrown when gain configuration fails
struct RobotSetGainsError : public std::runtime_error { 
    using std::runtime_error::runtime_error; 
};

// ════════════════════════════════════════════════════════════════════════════
// Robot Control Interface
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class Robot
 * @brief High-level interface for robot motor control and state observation
 * 
 * @details This class provides a safe, high-level API for controlling a robot
 *          with 16 motors (12 limb joints + 4 wheels). It handles:
 *          - Motor gain configuration (PD control)
 *          - State observation (positions, velocities, etc.)
 *          - Safety monitoring and emergency stop
 *          - Communication with motor controllers via FxCli
 */
class Robot {
public:
    /**
     * @brief Constructs and initializes the robot interface
     * @throws RobotInitError if initialization fails
     */
    Robot();

    // ───────────────────────────────────────────────────────────────────────
    // Control Configuration
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Sets PD controller gains for all motors
     * @param kp Proportional gains (size must equal NUM_MOTORS)
     * @param kd Derivative gains (size must equal NUM_MOTORS)
     * @throws RobotSetGainsError if gain configuration fails
     * @note Must be called before do_action() can be used
     */
    void setGains(const std::vector<float>& kp, const std::vector<float>& kd);

    // ───────────────────────────────────────────────────────────────────────
    // State Observation
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Retrieves current robot state observation
     * @return Map containing observation vectors (positions, velocities, etc.)
     * @note Returns a copy; internal buffers are reused for efficiency
     * @throws RobotEStopError if safety checks fail
     */
    std::unordered_map<std::string, std::vector<float>> getObs();

    // ───────────────────────────────────────────────────────────────────────
    // Safety Monitoring
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Performs safety checks on robot state
     * @throws RobotEStopError if any safety condition is violated
     * @details Checks for:
     *          - Communication timeouts
     *          - Disconnection counts
     *          - Hardware errors
     */
    void checkSafety();

    // ───────────────────────────────────────────────────────────────────────
    // Motor Control
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Executes a motor control action
     * @param action Motor commands (size must equal NUM_MOTORS)
     * @param torque_ctrl true for torque control, false for position control
     * @param safe If true, performs safety checks after executing
     * @throws RobotEStopError if safety checks fail (when safe=true)
     * @note Requires set_gains() to be called first
     */
    void doAction(const std::vector<float>& action, bool torque_ctrl, bool safe);

    // ───────────────────────────────────────────────────────────────────────
    // Emergency Control & Diagnostics
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Triggers emergency stop and terminates program
     * @param msg Error message describing the reason for E-stop
     */
    [[noreturn]] void estop(const std::string& msg, bool is_physical_estop=false);

    /**
     * @brief Logs a warning message with throttling
     * @param msg Warning message to log
     */
    void warn(const std::string& msg);

    // ───────────────────────────────────────────────────────────────────────
    // Python Interface Accessors
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Gets current PD controller gains
     * @return Pair of vectors: {kp, kd}
     */
    inline std::pair<std::vector<float>, std::vector<float>> getGains() const {
        return {kp_, kd_};
    }

    /**
     * @brief Checks if gains have been configured
     * @return true if set_gains() has been successfully called
     */
    inline bool gainsReady() const {
        return gains_set_;
    }

private:
    // ───────────────────────────────────────────────────────────────────────
    // Internal Utilities
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Initializes and waits for MCU to be ready for operation
     * @param mcu MCU client to initialize (contains motor IDs and capability flags)
     * @param timeout_ms Maximum wait time in milliseconds (default: 10000ms)
     * @throws RobotInitError if timeout is exceeded or initialization fails
     * @details Performs the following steps:
     *          1. Starts all motors on the MCU
     *          2. Checks motor status for errors
     *          3. Validates emergency stop button (if MCU has_estop)
     *          4. Checks battery level (if MCU has_battery)
     *          5. Gets initial observation data
     *          6. Initializes last_action with current joint positions
     */
    void waitMcu(mcu::McuClient& mcu, std::int32_t timeout_ms=10000);

    /**
     * @brief Parses and updates robot observation state from MCU data
     * @param mcu_data Raw observation data from motor controller (FxCli response)
     * @param mcu MCU client being updated (motor IDs, IMU/battery flags, req_miss_count)
     * @return Reference to internal observation map (obs)
     * @details Extracts the following data:
     *          - Motor positions and velocities for limb joints (→ dof_pos, dof_vel)
     *          - Motor velocities for wheel motors (→ dof_vel)
     *          - IMU data: angular velocity and projected gravity (if mcu.has_imu)
     *          Updates mcu.req_miss_count on failure, resets to 0 on success
     */
    std::unordered_map<std::string, std::vector<float>>& updateMcuObs(mcu::McuClient& m);

    /**
     * @brief Checks if a motor ID corresponds to a limb joint
     * @param mid Motor ID to check
     * @return true if motor is a limb joint (not a wheel)
     */
    bool isLimbMid(std::size_t mid) { return std::find(WHEEL_MIDS.begin(), WHEEL_MIDS.end(), mid) == WHEEL_MIDS.end();}


    std::string getMotorStatusString() const;
    // ───────────────────────────────────────────────────────────────────────
    // State Variables
    // ───────────────────────────────────────────────────────────────────────

    std::vector<float> last_action_;     ///< Last commanded action (for monitoring)
    bool last_torque_ctrl_;              ///< Last control mode used

    // Control gains
    std::vector<float> kp_;              ///< Proportional gains
    std::vector<float> kd_;              ///< Derivative gains
    bool gains_set_;                     ///< Flag indicating if gains are configured

    // Motor configuration
    std::vector<std::string> motor_pattern_;  ///< Motor connection patterns
    std::vector<std::string> motor_err_;      ///< Motor error states

    // Battery monitoring
    std::string battery_voltage_;        ///< Current battery voltage
    std::string battery_soc_;            ///< Battery state of charge (%)

    // Observation data
    std::unordered_map<std::string, std::vector<float>> obs_;  ///< Current observation

    // Hardware interface
    mcu::McuClient mcu_;
};

} // namespace robot
