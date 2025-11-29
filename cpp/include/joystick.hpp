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
    JoystickOutput get_cmd();

    /**
     * @brief Checks device connection status
     * @return true if joystick is connected
     */
    bool is_connected() const { return !disconnected.load(); }

private:
    // ───────────────────────────────────────────────────────────────────────
    // Constants
    // ───────────────────────────────────────────────────────────────────────

    static constexpr int TOTAL_IDX_NUM = 6;
    static const std::set<std::string> REQUIRED_KEYS;

    // ───────────────────────────────────────────────────────────────────────
    // Device & Threading
    // ───────────────────────────────────────────────────────────────────────
    int  open_device(int timeout_ms);
    void reader_thread_func();

    // ───────────────────────────────────────────────────────────────────────
    // Initialization
    // ───────────────────────────────────────────────────────────────────────
    
    void validate_max_cmd(const std::vector<double>& max_cmd);
    void initialize_mapping(const std::map<std::string,int>& mapping);
    void setup_scales();

    // ───────────────────────────────────────────────────────────────────────
    // Event Processing
    // ───────────────────────────────────────────────────────────────────────
    
    void process_event_batch(const std::vector<std::pair<std::string,int>>& batch);

    void update_mode();
    void update_estop_flag();
    void update_sleep_flag();
    void update_wake_flag();

    // ───────────────────────────────────────────────────────────────────────
    // Configuration
    // ───────────────────────────────────────────────────────────────────────
    
    std::map<std::string,int> mapping;          ///< Input code to name mapping
    std::map<std::string,int> key_to_cmd_idx;   ///< Axis name to command index
    std::map<std::string,std::string> btn_alias;///< Button aliases

    // ───────────────────────────────────────────────────────────────────────
    // Command State
    // ───────────────────────────────────────────────────────────────────────
    
    std::vector<double> joystick_input;    ///< Raw joystick values
    std::vector<double> robot_cmd;         ///< Smoothed output commands
    std::vector<double> robot_prev_cmd;    ///< Previous frame commands
    std::vector<double> max_cmd;           ///< Maximum command limits
    std::vector<double> dz_th;             ///< Deadzone thresholds
    std::map<int,double> scales;           ///< Axis scaling factors

    // ───────────────────────────────────────────────────────────────────────
    // Button & Mode State
    // ───────────────────────────────────────────────────────────────────────
    
    std::map<std::string,int> extra_btn_input;  ///< Non-axis button states
    std::optional<int> mode_id;                 ///< Current active mode
    std::optional<int> last_new_mode;           ///< Last requested mode change

    // ───────────────────────────────────────────────────────────────────────
    // Control Flags
    // ───────────────────────────────────────────────────────────────────────
    std::atomic<bool> estop_flag;       ///< Emergency stop activated
    std::atomic<bool> sleep_flag;       ///< Sleep mode requested
    std::atomic<bool> wake_flag;        ///< Wake from sleep requested
    std::atomic<bool> disconnected;     ///< Device disconnected
    std::atomic<bool> stop_thread;      ///< Stop reader thread

    // ───────────────────────────────────────────────────────────────────────
    // Timing (Hold Detection)
    // ───────────────────────────────────────────────────────────────────────
    
    std::optional<std::chrono::steady_clock::time_point> abs_z_pressed_since;   ///< Z trigger hold start
    std::optional<std::chrono::steady_clock::time_point> abs_rz_pressed_since;  ///< RZ trigger hold start

    // ───────────────────────────────────────────────────────────────────────
    // Event Queue (Thread-safe)
    // ───────────────────────────────────────────────────────────────────────
    
    std::deque<std::vector<std::pair<std::string,int>>> event_queue;  ///< Pending input events
    std::mutex queue_mutex;                                           ///< Queue protection

    // ───────────────────────────────────────────────────────────────────────
    // Hardware Interface
    // ───────────────────────────────────────────────────────────────────────
    
    std::thread reader_thread;  ///< Background input reader
    int device_fd;              ///< Joystick device file descriptor
};

} // namespace joystick