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

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include <fx_cli/fx_client.h> 

namespace mcu {

/// @brief Observation data read from a single MCU (MCU-specific, lightweight structure)
struct McuObs {
    // Vectors sized by number of motors
    std::vector<float> p;   ///< Motor positions
    std::vector<float> v;   ///< Motor velocities
    std::vector<float> t;   ///< Motor torques (optional if not needed)

    // Fixed-size arrays
    std::array<float, 3> g;   ///< IMU angular velocity or gyro (gx, gy, gz)
    std::array<float, 3> pg;  ///< Projected gravity (pgx, pgy, pgz)

    McuObs() = default;

    explicit McuObs(std::size_t num_motors)
        : p(num_motors, 0.0f)
        , v(num_motors, 0.0f)
        , t(num_motors, 0.0f)
        , g{0.0f, 0.0f, 0.0f}
        , pg{0.0f, 0.0f, 0.0f}
    {}
};


/// @brief Hardware/communication status information for a single MCU
/// @details Lightweight structure adapted from safety::StatusCheckResult to MCU level
struct McuStatus {
    bool disconnected;                      ///< Whether FxCli is disconnected
    bool emergency;                         ///< Whether emergency stop button is ON
    bool battery_low;                       ///< Whether battery SOC is below threshold

    std::vector<std::string> motor_pattern; ///< Motor pattern status (size = num_motors)
    std::vector<std::string> motor_err;     ///< Motor error codes (size = num_motors)
    std::string battery_voltage;            ///< Battery voltage (as string)
    std::string battery_soc;                ///< Battery SOC

    McuStatus() = default;

    explicit McuStatus(std::size_t num_motors)
        : disconnected(false)
        , emergency(false)
        , battery_low(false)
        , motor_pattern(num_motors, "N")
        , motor_err(num_motors, "None")
        , battery_voltage("")
        , battery_soc("")
    {}
};


/**
 * @brief MCU client wrapper with capability flags
 * @details Encapsulated client wrapping FxCli and managing motor/IMU/battery state per MCU
 */
class McuClient {
public:
    McuClient(const std::string& ip,
              int port,
              std::vector<uint8_t> ids,
              bool has_imu,
              bool has_bat,
              bool has_estop);

    // ─────────────────────────────────────────────────────────────
    // Read-only accessors
    // ─────────────────────────────────────────────────────────────

    /// @brief List of motor IDs controlled by this MCU (read-only)
    const std::vector<uint8_t>& motorIds()            const noexcept { return motor_ids_; }
    std::size_t                 numMotors()           const noexcept { return num_motors_; }
    bool                        hasImu()              const noexcept { return has_imu_; }
    bool                        hasBat()              const noexcept { return has_bat_; }
    bool                        hasEstop()            const noexcept { return has_estop_; }
    int                         getReqMissCount()     const noexcept { return req_miss_count_; }

    // ─────────────────────────────────────────────────────────────
    // FxCli wrapper API (exposing only necessary external controls)
    // ─────────────────────────────────────────────────────────────

    /// @brief Start all motors belonging to this MCU
    bool motorStart() {
        return cli_.motor_start(motor_ids_);
    }

    /// @brief Emergency-stop all motors belonging to this MCU
    bool motorEstop() {
        return cli_.motor_estop(motor_ids_);
    }

    /// @brief Send operation control command for this MCU's motors
    void operationControl(
        const std::vector<float>& pos,
        const std::vector<float>& vel,
        const std::vector<float>& kp,
        const std::vector<float>& kd,
        const std::vector<float>& tau)
    {
        cli_.operation_control(motor_ids_, pos, vel, kp, kd, tau);
    }

    /// @brief Read status from MCU, update mcu_obs, and return the result
    /// @return const Obs& : const reference to internal buffer (mcu_obs)
    const McuObs& fetchObs();
    const McuStatus& fetchStatus();

private:
    // ─────────────────────────────────────────────────────────────
    // Internal state (fully encapsulated)
    // ─────────────────────────────────────────────────────────────

    FxCli                cli_;         ///< Native FxCli instance
    std::vector<uint8_t> motor_ids_;   ///< Motor IDs controlled by this MCU
    std::size_t          num_motors_;  ///< Number of motors controlled by this MCU
    bool has_imu_;                     ///< True if this MCU has IMU sensor
    bool has_bat_;                     ///< True if this MCU has battery monitoring
    bool has_estop_;                   ///< True if this MCU has emergency stop button
    int  req_miss_count_;              ///< Request miss counter for this MCU
    int  max_req_miss_count_;
    int  status_disconn_count_;
    int  max_status_disconn_count_;
    
    McuObs mcu_obs_;                   ///< Latest observation result
    McuStatus mcu_status_;             ///< Latest status result
};

} // namespace mcu