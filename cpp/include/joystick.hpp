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

#pragma once  // joystick.hpp

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace joystick {

// ════════════════════════════════════════════════════════════════════════════
// Exceptions
// ════════════════════════════════════════════════════════════════════════════

/// @brief Thrown when E-stop is triggered via joystick
class JoystickEstopError : public std::runtime_error {
public:
    explicit JoystickEstopError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/// @brief Thrown when sleep mode is triggered
class JoystickSleepError : public std::runtime_error {
public:
    explicit JoystickSleepError(const std::string& msg)
        : std::runtime_error(msg) {}
};

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

static constexpr double DEADZONE = 0.03;
static constexpr double HOLD_SEC = 2.0;
static constexpr double JOYSTICK_MAX_VAL = 32767.0;

// ════════════════════════════════════════════════════════════════════════════
// Data Structures
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Joystick command output bundle
 * @details Contains (scaled) command vector, mode changes, and control flags
 */
struct JoystickOutput {
    std::vector<double> cmd_vector;  ///< Scaled command values (6-DOF)
    std::optional<int>  mode_id;     ///< Current mode ID (if changed)
    bool estop;                      ///< Emergency stop triggered
    bool wake;                       ///< Wake from sleep requested
    bool sleep;                      ///< Sleep mode requested

    JoystickOutput()
        : cmd_vector(6, 0.0),
          mode_id(std::nullopt),
          estop(false),
          wake(false),
          sleep(false) {}
};

// ════════════════════════════════════════════════════════════════════════════
// Joystick Interface
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class Joystick
 * @brief Asynchronous joystick input handler with smoothing and mode switching
 * @note Thread-safe; runs background reader thread for input events
 */
class Joystick {
public:
    /**
     * @brief Constructs joystick handler with optional configuration
     * @param max_cmd Maximum command values per axis (size 6)
     * @param smoothness Exponential smoothing factor
     * @param mapping Custom button/axis mapping
     */
    Joystick(const std::vector<double>& max_cmd = {},
             double smoothness = 50.0,
             const std::map<std::string,int>& mapping = {});
    ~Joystick();

    /**
     * @brief Retrieves current joystick command
     * @return Command output with smoothed values and control flags
     * @throws JoystickEstopError if E-stop is triggered
     * @throws JoystickSleepError if sleep is requested
     */
    JoystickOutput getCmd();

    /**
     * @brief Checks device connection status
     * @return true if joystick is connected
     */
    bool isConnected() const { return !disconnected_.load(); }

private:
    // ───────────────────────────────────────────────────────────────────────
    // Constants
    // ───────────────────────────────────────────────────────────────────────

    static constexpr int TOTAL_IDX_NUM = 6;
    static const std::set<std::string> REQUIRED_KEYS;

    // ───────────────────────────────────────────────────────────────────────
    // Device & Threading
    // ───────────────────────────────────────────────────────────────────────
    int  openDevice(int timeout_ms);
    void flushEventBuffer(int fd);
    void readerThreadFunc();

    // ───────────────────────────────────────────────────────────────────────
    // Initialization
    // ───────────────────────────────────────────────────────────────────────
    
    void validateMaxCmd(const std::vector<double>& max_cmd);
    void initializeMapping(const std::map<std::string,int>& mapping);
    void setupScales();

    // ───────────────────────────────────────────────────────────────────────
    // Event Processing
    // ───────────────────────────────────────────────────────────────────────
    
    void processEventBatch(const std::vector<std::pair<std::string,int>>& batch);

    void updateMode();
    void updateEstopFlag();
    void updateSleepFlag();
    void updateWakeFlag();

    // ───────────────────────────────────────────────────────────────────────
    // Configuration
    // ───────────────────────────────────────────────────────────────────────
    
    std::map<std::string,int> mapping_;          ///< Input code to name mapping
    std::map<std::string,int> key_to_cmd_idx_;   ///< Axis name to command index
    std::map<std::string,std::string> btn_alias_;///< Button aliases

    // ───────────────────────────────────────────────────────────────────────
    // Command State
    // ───────────────────────────────────────────────────────────────────────
    
    std::vector<double> joystick_input_;    ///< Raw joystick values
    std::vector<double> robot_cmd_;         ///< Smoothed output commands
    std::vector<double> robot_prev_cmd_;    ///< Previous frame commands
    std::vector<double> max_cmd_;           ///< Maximum command limits
    std::vector<double> dz_th_;             ///< Deadzone thresholds
    std::map<int,double> scales_;           ///< Axis scaling factors

    // ───────────────────────────────────────────────────────────────────────
    // Button & Mode State
    // ───────────────────────────────────────────────────────────────────────
    
    std::map<std::string,int> extra_btn_input_;  ///< Non-axis button states
    std::optional<int> mode_id_;                 ///< Current active mode
    std::optional<int> last_new_mode_;           ///< Last requested mode change

    // ───────────────────────────────────────────────────────────────────────
    // Control Flags
    // ───────────────────────────────────────────────────────────────────────
    std::atomic<bool> estop_flag_;       ///< Emergency stop activated
    std::atomic<bool> sleep_flag_;       ///< Sleep mode requested
    std::atomic<bool> wake_flag_;        ///< Wake from sleep requested
    std::atomic<bool> disconnected_;     ///< Device disconnected
    std::atomic<bool> stop_thread_;      ///< Stop reader thread

    // ───────────────────────────────────────────────────────────────────────
    // Timing (Hold Detection)
    // ───────────────────────────────────────────────────────────────────────
    
    std::optional<std::chrono::steady_clock::time_point> abs_z_pressed_since_;   ///< Z trigger hold start
    std::optional<std::chrono::steady_clock::time_point> abs_rz_pressed_since_;  ///< RZ trigger hold start

    // ───────────────────────────────────────────────────────────────────────
    // Event Queue (Thread-safe)
    // ───────────────────────────────────────────────────────────────────────
    
    std::deque<std::vector<std::pair<std::string,int>>> event_queue_;  ///< Pending input events
    std::mutex queue_mutex_;                                           ///< Queue protection

    // ───────────────────────────────────────────────────────────────────────
    // Hardware Interface
    // ───────────────────────────────────────────────────────────────────────
    
    std::thread reader_thread_;  ///< Background input reader
    int device_fd_;              ///< Joystick device file descriptor
};

} // namespace joystick
