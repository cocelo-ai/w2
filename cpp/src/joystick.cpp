#include <fcntl.h>
#include <unistd.h>
#include <glob.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <linux/input.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <cmath>

#include "joystick.hpp"

namespace joystick {

// Required mapping keys
const std::set<std::string> Joystick::REQUIRED_KEYS = {
    "LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y", "LEFT_BTN", "RIGHT_BTN"
};

// LPF coefficients
static double LPF_ALPHA = 0.0;
static double LPF_ALPHA_STOP = 0.0;

// Constructor: initialize state, validate mapping, start thread
Joystick::Joystick(const std::vector<double>& max_cmd_input,
                   double smoothness,
                   const std::map<std::string,int>& mapping_input)
    : joystick_input_(TOTAL_IDX_NUM, 0.0),
      robot_cmd_(TOTAL_IDX_NUM, 0.0),
      robot_prev_cmd_(TOTAL_IDX_NUM, 0.0),
      max_cmd_(TOTAL_IDX_NUM, 1.0),
      dz_th_(TOTAL_IDX_NUM, 0.0),
      mode_id_(std::nullopt),
      last_new_mode_(std::nullopt),
      estop_flag_(false),
      sleep_flag_(false),
      wake_flag_(false),
      disconnected_(true),
      stop_thread_(false),
      abs_z_pressed_since_(std::nullopt),
      abs_rz_pressed_since_(std::nullopt),
      device_fd_(-1)
{
    // Face button aliases
    btn_alias_["BTN_X"] = "X";
    btn_alias_["BTN_NORTH"] = "X";
    btn_alias_["BTN_B"] = "B";
    btn_alias_["BTN_EAST"] = "B";
    btn_alias_["BTN_A"] = "A";
    btn_alias_["BTN_GAMEPAD"] = "A";
    btn_alias_["BTN_SOUTH"] = "A";
    btn_alias_["BTN_Y"] = "Y";
    btn_alias_["BTN_WEST"] = "Y";

    // Clip smoothness to [0, 100]
    if (smoothness < 0.0)  smoothness = 0.0;
    if (smoothness > 100.0) smoothness = 100.0;
    
    // Compute LPF coefficients based on smoothness
    LPF_ALPHA = 0.9757 * (1.0 - std::exp(-0.0732 * smoothness));
    LPF_ALPHA_STOP = 0.9281 * (1.0 - std::exp(-0.0574 * smoothness));

    LPF_ALPHA      = std::clamp(LPF_ALPHA, 0.0, 0.99);
    LPF_ALPHA_STOP = std::clamp(LPF_ALPHA_STOP, 0.0, 0.99);

    validateMaxCmd(max_cmd_input);
    initializeMapping(mapping_input);
    setupScales();

    // Initialize extra button inputs to neutral state
    extra_btn_input_["hatX"] = 0;
    extra_btn_input_["hatY"] = 0;
    extra_btn_input_["X"] = 0;
    extra_btn_input_["B"] = 0;
    extra_btn_input_["A"] = 0;
    extra_btn_input_["Y"] = 0;
    extra_btn_input_["ABS_Z"] = 0;
    extra_btn_input_["ABS_RZ"] = 0;

    // Start background reader thread
    reader_thread_ = std::thread(&Joystick::readerThreadFunc, this);
}

// Destructor: stop thread, close device
Joystick::~Joystick() {
    stop_thread_ = true;
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
    if (device_fd_ >= 0) {
        ::close(device_fd_);
        device_fd_ = -1;
    }
}

// Validate max_cmd: non-negative, at most 6, default 1.0
void Joystick::validateMaxCmd(const std::vector<double>& max_cmd_input) {
    max_cmd_.assign(TOTAL_IDX_NUM, 1.0);
    if (!max_cmd_input.empty()) {
        if (max_cmd_input.size() > static_cast<size_t>(TOTAL_IDX_NUM)) {
            throw std::invalid_argument(
                "max_cmd can have at most " + std::to_string(TOTAL_IDX_NUM) +
                " elements (got " + std::to_string(max_cmd_input.size()) + ").");
        }
        for (size_t i = 0; i < max_cmd_input.size(); ++i) {
            double v = max_cmd_input[i];
            if (v < 0.0) {
                throw std::invalid_argument(
                    "max_cmd[" + std::to_string(i) + "] must not be negative, got " +
                    std::to_string(v) + ".");
            }
            max_cmd_[i] = v;
        }
    }
    dz_th_.resize(TOTAL_IDX_NUM);
    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        dz_th_[i] = max_cmd_[i] * DEADZONE;
    }
}

// Validate and set up logical mapping + evdev code mapping
void Joystick::initializeMapping(const std::map<std::string,int>& mapping_input) {
    if (mapping_input.empty()) {
        // Default mapping for F710 gamepad
        mapping_["LEFT_X"]  = 2;
        mapping_["LEFT_Y"]  = 0;
        mapping_["RIGHT_X"] = 1;
        mapping_["RIGHT_Y"] = 3;
        mapping_["LEFT_BTN"]  = 4;
        mapping_["RIGHT_BTN"] = 5;
    } else {
        std::set<std::string> provided_keys;
        for (const auto& kv : mapping_input) {
            provided_keys.insert(kv.first);
        }
        std::set<std::string> missing;
        std::set<std::string> extra;

        for (const auto& k : REQUIRED_KEYS) {
            if (!provided_keys.count(k)) missing.insert(k);
        }
        for (const auto& k : provided_keys) {
            if (!REQUIRED_KEYS.count(k)) extra.insert(k);
        }
        if (!missing.empty() || !extra.empty()) {
            std::string msg = "Invalid mapping keys.";
            if (!missing.empty()) {
                msg += " Missing keys: {";
                for (const auto& k : missing) msg += k + ", ";
                if (!missing.empty()) { msg.pop_back(); msg.pop_back(); }
                msg += "}.";
            }
            if (!extra.empty()) {
                msg += " Unknown keys: {";
                for (const auto& k : extra) msg += k + ", ";
                if (!extra.empty()) { msg.pop_back(); msg.pop_back(); }
                msg += "}.";
            }
            throw std::invalid_argument(msg);
        }

        std::set<int> used;
        for (const auto& [key, idx] : mapping_input) {
            if (idx < 0 || idx >= TOTAL_IDX_NUM) {
                throw std::invalid_argument(
                    "Index " + std::to_string(idx) + " is not between 0 and " +
                    std::to_string(TOTAL_IDX_NUM - 1) + " (inclusive)."
                );
            }
            if (used.count(idx)) {
                throw std::invalid_argument("Duplicate indices found in mapping.");
            }
            used.insert(idx);
        }
        mapping_ = mapping_input;
    }

    // Map evdev axis/button codes to command indices
    key_to_cmd_idx_.clear();
    key_to_cmd_idx_["ABS_X"]  = mapping_["LEFT_X"];
    key_to_cmd_idx_["ABS_Y"]  = mapping_["LEFT_Y"];
    key_to_cmd_idx_["ABS_RX"] = mapping_["RIGHT_X"];
    key_to_cmd_idx_["ABS_RY"] = mapping_["RIGHT_Y"];
    key_to_cmd_idx_["BTN_TL"] = mapping_["LEFT_BTN"];
    key_to_cmd_idx_["BTN_TR"] = mapping_["RIGHT_BTN"];
}

// Set up scaling factors for axes and shoulder buttons
void Joystick::setupScales() {
    scales_.clear();
    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        scales_[i] = 0.0;
    }
    int idx;
    idx = mapping_["LEFT_X"];
    scales_[idx] = -max_cmd_[idx] / JOYSTICK_MAX_VAL;
    idx = mapping_["LEFT_Y"];
    scales_[idx] = -max_cmd_[idx] / JOYSTICK_MAX_VAL;
    idx = mapping_["RIGHT_X"];
    scales_[idx] = -max_cmd_[idx] / JOYSTICK_MAX_VAL;
    idx = mapping_["RIGHT_Y"];
    scales_[idx] = -max_cmd_[idx] / JOYSTICK_MAX_VAL;

    idx = mapping_["LEFT_BTN"];
    scales_[idx] = 1.0;
    idx = mapping_["RIGHT_BTN"];
    scales_[idx] = 1.0;
}

// Open F710 joystick device with timeout
int Joystick::openDevice(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    while (true) {
        glob_t glob_result;
        memset(&glob_result, 0, sizeof(glob_result));
        int ret = glob("/dev/input/by-id/*F710*-event-joystick",
                       GLOB_NOSORT, nullptr, &glob_result);
        if (ret == 0 && glob_result.gl_pathc > 0) {
            for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
                int fd = ::open(glob_result.gl_pathv[i], O_RDONLY | O_NONBLOCK);
                if (fd >= 0) {
                    globfree(&glob_result);
                    
                    // FIX: Flush stale events from kernel evdev buffer
                    // This prevents reading old button presses from previous process instances
                    flushEventBuffer(fd);
                    
                    return fd;
                }
            }
        }
        globfree(&glob_result);
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// FIX: Flush stale events remaining in kernel buffer after process restart
void Joystick::flushEventBuffer(int fd) {
    struct input_event ev;
    int flush_count = 0;
    
    // Read and discard all pending events until buffer is empty
    while (true) {
        ssize_t bytes = ::read(fd, &ev, sizeof(ev));
        if (bytes == static_cast<ssize_t>(sizeof(ev))) {
            flush_count++;
            continue;  // Keep reading and discarding
        } else if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // No more data available (normal condition)
            break;
        } else {
            // Read error occurred
            break;
        }
    }
    
    if (flush_count > 0) {
        std::cout << "[Joystick] Flushed " << flush_count 
                  << " stale events from kernel buffer\n";
    }
}

// Background reader thread: read evdev events into queue
void Joystick::readerThreadFunc() {
    while (!stop_thread_) {
        if (disconnected_ || device_fd_ < 0) {
            if (device_fd_ >= 0) {
                ::close(device_fd_);
                device_fd_ = -1;
            }

            device_fd_ = openDevice(1000);
            disconnected_ = (device_fd_ < 0);

            if (disconnected_) {
                joystick_input_.assign(TOTAL_IDX_NUM, 0.0);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(device_fd_, &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 0.1 second timeout
        int sel = ::select(device_fd_ + 1, &rfds, nullptr, nullptr, &tv);
        if (sel < 0) {
            disconnected_ = true;
            continue;
        }
        if (sel == 0) {
            continue;
        }

        std::vector<std::pair<std::string,int>> batch;
        struct input_event ev;
        while (true) {
            ssize_t bytes = ::read(device_fd_, &ev, sizeof(ev));
            if (bytes == static_cast<ssize_t>(sizeof(ev))) {
                std::string code;
                int value = 0;

                if (ev.type == EV_ABS) {
                    // Absolute axis events
                    switch (ev.code) {
                    case 0:  code = "ABS_X";      break;
                    case 1:  code = "ABS_Y";      break;
                    case 2:  code = "ABS_Z";      break;
                    case 3:  code = "ABS_RX";     break;
                    case 4:  code = "ABS_RY";     break;
                    case 5:  code = "ABS_RZ";     break;
                    case 16: code = "ABS_HAT0X";  break;
                    case 17: code = "ABS_HAT0Y";  break;
                    default: code = "ABS_" + std::to_string(ev.code); break;
                    }
                    value = ev.value;
                } else if (ev.type == EV_KEY) {
                    // Button press/release events
                    switch (ev.code) {
                    case 304: code = "BTN_A";         break;
                    case 305: code = "BTN_B";         break;
                    case 306: code = "BTN_C";         break;
                    case 307: code = "BTN_X";         break;
                    case 308: code = "BTN_Y";         break;
                    case 309: code = "BTN_Z";         break;
                    case 310: code = "BTN_TL";        break;
                    case 311: code = "BTN_TR";        break;
                    case 312: code = "BTN_TL2";       break;
                    case 313: code = "BTN_TR2";       break;
                    case 314: code = "BTN_SELECT";    break;
                    case 315: code = "BTN_START";     break;
                    case 316: code = "BTN_MODE";      break;
                    case 317: code = "BTN_THUMBL";    break;
                    case 318: code = "BTN_THUMBR";    break;
                    case 544: code = "BTN_DPAD_UP";   break;
                    case 545: code = "BTN_DPAD_DOWN"; break;
                    case 546: code = "BTN_DPAD_LEFT"; break;
                    case 547: code = "BTN_DPAD_RIGHT";break;
                    default:  code = "KEY_" + std::to_string(ev.code); break;
                    }
                    value = (ev.value > 0) ? 1 : 0;
                } else {
                    continue;
                }

                batch.emplace_back(code, value);
                disconnected_ = false;
            } else {
                if (bytes < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        break;
                    } else {
                        disconnected_ = true;
                        break;
                    }
                }
                if (bytes == 0) {
                    disconnected_ = true;
                }
                break;
            }
        }

        if (!batch.empty()) {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            event_queue_.push_back(std::move(batch));
            if (event_queue_.size() > 64) {
                event_queue_.pop_front();
            }
        }
    }

    if (device_fd_ >= 0) {
        ::close(device_fd_);
        device_fd_ = -1;
    }
}

// Decode one batch of events into internal state
void Joystick::processEventBatch(const std::vector<std::pair<std::string,int>>& batch) {
    for (const auto& item : batch) {
        const std::string& code = item.first;
        int state = item.second;

        // D-pad horizontal axis
        if (code == "ABS_HAT0X") {
            if (state < 0)      extra_btn_input_["hatX"] = -1;
            else if (state > 0) extra_btn_input_["hatX"] = 1;
            else                extra_btn_input_["hatX"] = 0;
            continue;
        }
        // D-pad vertical axis
        if (code == "ABS_HAT0Y") {
            if (state < 0)      extra_btn_input_["hatY"] = -1;
            else if (state > 0) extra_btn_input_["hatY"] = 1;
            else                extra_btn_input_["hatY"] = 0;
            continue;
        }

        // Left trigger (analog)
        if (code == "ABS_Z") {
            extra_btn_input_["ABS_Z"] = state;
            continue;
        }
        // Right trigger (analog)
        if (code == "ABS_RZ") {
            extra_btn_input_["ABS_RZ"] = state;
            continue;
        }

        // Face buttons (with aliasing for different driver versions)
        auto alias_it = btn_alias_.find(code);
        if (alias_it != btn_alias_.end()) {
            extra_btn_input_[alias_it->second] = state;
            continue;
        }

        // Joystick axes and shoulder buttons
        auto cmd_it = key_to_cmd_idx_.find(code);
        if (cmd_it != key_to_cmd_idx_.end()) {
            int idx = cmd_it->second;
            if (code == "BTN_TL" || code == "BTN_TR") {
                joystick_input_[idx] = state ? 1.0 : 0.0;
            } else {
                joystick_input_[idx] = static_cast<double>(state);
            }
            continue;
        }
    }
}

// Update mode selection based on D-pad + face button combinations
void Joystick::updateMode() {
    std::optional<int> new_mode;

    int hatX = extra_btn_input_["hatX"];
    int hatY = extra_btn_input_["hatY"];
    int X = extra_btn_input_["X"];
    int B = extra_btn_input_["B"];
    int A = extra_btn_input_["A"];
    int Y = extra_btn_input_["Y"];

    // D-pad up combinations
    if (hatY == -1) {
        if (Y)      new_mode = 1;
        else if (B) new_mode = 2;
        else if (A) new_mode = 3;
        else if (X) new_mode = 4;
    }
    // D-pad right combinations
    else if (hatX == 1) {
        if (Y)      new_mode = 5;
        else if (B) new_mode = 6;
        else if (A) new_mode = 7;
        else if (X) new_mode = 8;
    }
    // D-pad down combinations
    else if (hatY == 1) {
        if (Y)      new_mode = 9;
        else if (B) new_mode = 10;
        else if (A) new_mode = 11;
        else if (X) new_mode = 12;
    }
    // D-pad left combinations
    else if (hatX == -1) {
        if (Y)      new_mode = 13;
        else if (B) new_mode = 14;
        else if (A) new_mode = 15;
        else if (X) new_mode = 16;
    }

    // Only trigger mode change on button press (not hold)
    if (new_mode.has_value() && !last_new_mode_.has_value()) {
        mode_id_ = new_mode;
    } else {
        mode_id_.reset();
    }
    last_new_mode_ = new_mode;
}

// Check if both triggers pressed simultaneously -> emergency stop
void Joystick::updateEstopFlag() {
    bool new_estop =
        (extra_btn_input_["ABS_Z"] > 0 && extra_btn_input_["ABS_RZ"] > 0);
    estop_flag_.store(new_estop);
}

// Right trigger held for HOLD_SEC seconds -> sleep mode
void Joystick::updateSleepFlag() {
    auto now = std::chrono::steady_clock::now();
    bool rz_active = extra_btn_input_["ABS_RZ"] > 0;
    if (rz_active) {
        if (!abs_rz_pressed_since_.has_value()) {
            abs_rz_pressed_since_ = now;
        }
        double elapsed =
            std::chrono::duration<double>(now - abs_rz_pressed_since_.value()).count();
        bool held = elapsed >= HOLD_SEC;
        sleep_flag_.store(held);
    } else {
        abs_rz_pressed_since_.reset();
        sleep_flag_.store(false);
    }
}

// Left trigger held for HOLD_SEC seconds -> wake from sleep
void Joystick::updateWakeFlag() {
    auto now = std::chrono::steady_clock::now();
    bool z_active = extra_btn_input_["ABS_Z"] > 0;
    if (z_active) {
        if (!abs_z_pressed_since_.has_value()) {
            abs_z_pressed_since_ = now;
        }
        double elapsed =
            std::chrono::duration<double>(now - abs_z_pressed_since_.value()).count();
        bool held = elapsed >= HOLD_SEC;
        wake_flag_.store(held);
    } else {
        abs_z_pressed_since_.reset();
        wake_flag_.store(false);
    }
}

// Drain event queue, apply smoothing filter, return command output
JoystickOutput Joystick::getCmd() {
    // Process all pending events from the queue
    while (true) {
        std::vector<std::pair<std::string,int>> batch;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (event_queue_.empty()) {
                break;
            }
            batch = std::move(event_queue_.front());
            event_queue_.pop_front();
        }
        processEventBatch(batch);
    }

    // Apply deadzone and low-pass filter to each axis
    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        double raw_cmd = joystick_input_[i] * scales_[i];
        
        // Apply deadzone threshold
        if (std::fabs(raw_cmd) < dz_th_[i]) {
            raw_cmd = 0.0;
        }

        // Apply low-pass filter (skip for digital buttons)
        if (i != mapping_["LEFT_BTN"] && i != mapping_["RIGHT_BTN"]) {
            double filtered;
            if (raw_cmd == 0.0) {
                // Use slower filter when stopping
                filtered =
                    LPF_ALPHA_STOP * robot_prev_cmd_[i] +
                    (1.0 - LPF_ALPHA_STOP) * raw_cmd;
            } else {
                // Use faster filter when moving
                filtered =
                    LPF_ALPHA * robot_prev_cmd_[i] +
                    (1.0 - LPF_ALPHA) * raw_cmd;
            }

            robot_prev_cmd_[i] = filtered;
            robot_cmd_[i]      = filtered;
        } else {
            robot_cmd_[i] = raw_cmd;
        }

        // Clip to max_cmd limits
        double maxval = max_cmd_[i];
        if (robot_cmd_[i] > maxval - 1e-3) {
            robot_cmd_[i] = maxval;
        } else if (robot_cmd_[i] < -maxval + 1e-3) {
            robot_cmd_[i] = -maxval;
        }

        robot_prev_cmd_[i] = robot_cmd_[i];
    }

    // Update control flags
    updateMode();
    updateEstopFlag();
    updateSleepFlag();
    updateWakeFlag();

    // Throw exceptions for critical events
    if (estop_flag_.load()) {
        throw JoystickEstopError("E-stop triggered by joystick input.");
    }
    if (sleep_flag_.load()) {
        throw JoystickSleepError("Sleep triggered by joystick input.");
    }

    // Return command output
    JoystickOutput output;
    output.cmd_vector = robot_cmd_;
    output.mode_id = mode_id_;
    output.estop = estop_flag_.load();
    output.wake = wake_flag_.load();
    output.sleep = sleep_flag_.load();
    return output;
}

} // namespace joystick
