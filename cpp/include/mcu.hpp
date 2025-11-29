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

#include <fx_cli/fx_client.h> 

namespace robot {

/**
 * @brief MCU client wrapper with capability flags
 * @details Encapsulates FxCli with MCU-specific hardware capabilities
 */
struct McuClient {
    FxCli cli;                              ///< Native FxCli instance
    std::vector<uint8_t> motor_ids;         ///< Motor IDs controlled by this MCU
    std::size_t num_motors;                 ///< Number of motors controlled by this MCU
    bool has_imu;                           ///< True if this MCU has IMU sensor
    bool has_bat;                           ///< True if this MCU has battery monitoring
    bool has_estop;                         ///< True if this MCU has emergency stop button
    int  req_miss_count;                    ///< Request miss counter for this MCU
    
    McuClient(const std::string& ip,
              int port,
              std::vector<uint8_t> ids,
              bool has_imu,
              bool has_bat,
              bool has_estop);

    // ─────────────────────────────────────────────────────────────
    // Convenience wrapper methods (FxCli → McuClient API)
    // ─────────────────────────────────────────────────────────────

    /// @brief Start all motors belonging to this MCU
    inline bool motor_start() {
        return cli.motor_start(motor_ids);
    }

    /// @brief Emergency-stop all motors belonging to this MCU
    inline bool motor_estop() {
        return cli.motor_estop(motor_ids);
    }

    /// @brief Get status map from this MCU
    inline FxCliMap status() {
        return cli.status();
    }

    /// @brief Request state for this MCU's motors
    inline FxCliMap req() {
        return cli.req(motor_ids);
    }

    /// @brief Send operation control command for this MCU's motors
    inline void operation_control(
        const std::vector<float>& pos,
        const std::vector<float>& vel,
        const std::vector<float>& kp,
        const std::vector<float>& kd,
        const std::vector<float>& tau)
    {
        cli.operation_control(
            motor_ids,
            pos, vel,
            kp, kd,
            tau
        );
    }
};

} // namespace robot
