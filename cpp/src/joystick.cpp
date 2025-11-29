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

// Constructor: init state, validate mapping, start thread
Joystick::Joystick(const std::vector<double>& max_cmd_input,
                   double smoothness,
                   const std::map<std::string,int>& mapping_input)
    : joystick_input(TOTAL_IDX_NUM, 0.0),
      robot_cmd(TOTAL_IDX_NUM, 0.0),
      robot_prev_cmd(TOTAL_IDX_NUM, 0.0),
      max_cmd(TOTAL_IDX_NUM, 1.0),
      dz_th(TOTAL_IDX_NUM, 0.0),
      mode_id(std::nullopt),
      last_new_mode(std::nullopt),
      estop_flag(false),
      sleep_flag(false),
      wake_flag(false),
      disconnected(true),
      stop_thread(false),
      abs_z_pressed_since(std::nullopt),
      abs_rz_pressed_since(std::nullopt),
      device_fd(-1)
{
    // Face button aliases
    btn_alias["BTN_X"] = "X";
    btn_alias["BTN_NORTH"] = "X";
    btn_alias["BTN_B"] = "B";
    btn_alias["BTN_EAST"] = "B";
    btn_alias["BTN_A"] = "A";
    btn_alias["BTN_GAMEPAD"] = "A";
    btn_alias["BTN_SOUTH"] = "A";
    btn_alias["BTN_Y"] = "Y";
    btn_alias["BTN_WEST"] = "Y";

    // Clip smoothness to [0, 100]
    if (smoothness < 0.0)  smoothness = 0.0;
    if (smoothness > 100.0) smoothness = 100.0;
    
    // Compute LPF coefficients based on smoothness
    LPF_ALPHA = 0.9757 * (1.0 - std::exp(-0.0732 * smoothness));
    LPF_ALPHA_STOP = 0.9281 * (1.0 - std::exp(-0.0574 * smoothness));

    LPF_ALPHA      = std::clamp(LPF_ALPHA, 0.0, 0.99);
    LPF_ALPHA_STOP = std::clamp(LPF_ALPHA_STOP, 0.0, 0.99);

    validate_max_cmd(max_cmd_input);
    initialize_mapping(mapping_input);
    setup_scales();

    // Neutral extra inputs
    extra_btn_input["hatX"] = 0;
    extra_btn_input["hatY"] = 0;
    extra_btn_input["X"] = 0;
    extra_btn_input["B"] = 0;
    extra_btn_input["A"] = 0;
    extra_btn_input["Y"] = 0;
    extra_btn_input["ABS_Z"] = 0;
    extra_btn_input["ABS_RZ"] = 0;

    // Start reader thread
    reader_thread = std::thread(&Joystick::reader_thread_func, this);
}

// Destructor: stop thread, close device
Joystick::~Joystick() {
    stop_thread = true;
    if (reader_thread.joinable()) {
        reader_thread.join();
    }
    if (device_fd >= 0) {
        ::close(device_fd);
        device_fd = -1;
    }
}

// max_cmd: non-negative, at most 6, default 1.0
void Joystick::validate_max_cmd(const std::vector<double>& max_cmd_input) {
    max_cmd.assign(TOTAL_IDX_NUM, 1.0);
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
            max_cmd[i] = v;
        }
    }
    dz_th.resize(TOTAL_IDX_NUM);
    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        dz_th[i] = max_cmd[i] * DEADZONE;
    }
}

// Logical mapping + evdev code mapping
void Joystick::initialize_mapping(const std::map<std::string,int>& mapping_input) {
    if (mapping_input.empty()) {
        // Default mapping 
        mapping["LEFT_X"]  = 2;
        mapping["LEFT_Y"]  = 0;
        mapping["RIGHT_X"] = 1;
        mapping["RIGHT_Y"] = 3;
        mapping["LEFT_BTN"]  = 4;
        mapping["RIGHT_BTN"] = 5;
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
        mapping = mapping_input;
    }

    // Evdev code -> command index
    key_to_cmd_idx.clear();
    key_to_cmd_idx["ABS_X"]  = mapping["LEFT_X"];
    key_to_cmd_idx["ABS_Y"]  = mapping["LEFT_Y"];
    key_to_cmd_idx["ABS_RX"] = mapping["RIGHT_X"];
    key_to_cmd_idx["ABS_RY"] = mapping["RIGHT_Y"];
    key_to_cmd_idx["BTN_TL"] = mapping["LEFT_BTN"];
    key_to_cmd_idx["BTN_TR"] = mapping["RIGHT_BTN"];
}

// Scales for axes and shoulder buttons
void Joystick::setup_scales() {
    scales.clear();
    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        scales[i] = 0.0;
    }
    int idx;
    idx = mapping["LEFT_X"];
    scales[idx] = -max_cmd[idx] / JOYSTICK_MAX_VAL;
    idx = mapping["LEFT_Y"];
    scales[idx] = -max_cmd[idx] / JOYSTICK_MAX_VAL;
    idx = mapping["RIGHT_X"];
    scales[idx] = -max_cmd[idx] / JOYSTICK_MAX_VAL;
    idx = mapping["RIGHT_Y"];
    scales[idx] = -max_cmd[idx] / JOYSTICK_MAX_VAL;

    idx = mapping["LEFT_BTN"];
    scales[idx] = 1.0;
    idx = mapping["RIGHT_BTN"];
    scales[idx] = 1.0;
}

// Open F710 joystick device with timeout
int Joystick::open_device(int timeout_ms) {
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

// Background reader thread: read evdev events into queue
void Joystick::reader_thread_func() {
    while (!stop_thread) {
        if (disconnected || device_fd < 0) {
            if (device_fd >= 0) {
                ::close(device_fd);
                device_fd = -1;
            }

            device_fd = open_device(1000);
            disconnected = (device_fd < 0);

            if (disconnected) {
                joystick_input.assign(TOTAL_IDX_NUM, 0.0);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(device_fd, &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 0.1 s
        int sel = ::select(device_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel < 0) {
            disconnected = true;
            continue;
        }
        if (sel == 0) {
            continue;
        }

        std::vector<std::pair<std::string,int>> batch;
        struct input_event ev;
        while (true) {
            ssize_t bytes = ::read(device_fd, &ev, sizeof(ev));
            if (bytes == static_cast<ssize_t>(sizeof(ev))) {
                std::string code;
                int value = 0;

                if (ev.type == EV_ABS) {
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
                disconnected = false;
            } else {
                if (bytes < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        break;
                    } else {
                        disconnected = true;
                        break;
                    }
                }
                if (bytes == 0) {
                    disconnected = true;
                }
                break;
            }
        }

        if (!batch.empty()) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            event_queue.push_back(std::move(batch));
            if (event_queue.size() > 64) {
                event_queue.pop_front();
            }
        }
    }

    if (device_fd >= 0) {
        ::close(device_fd);
        device_fd = -1;
    }
}

// Decode one batch of events into state
void Joystick::process_event_batch(const std::vector<std::pair<std::string,int>>& batch) {
    for (const auto& item : batch) {
        const std::string& code = item.first;
        int state = item.second;

        if (code == "ABS_HAT0X") {
            if (state < 0)      extra_btn_input["hatX"] = -1;
            else if (state > 0) extra_btn_input["hatX"] = 1;
            else                extra_btn_input["hatX"] = 0;
            continue;
        }
        if (code == "ABS_HAT0Y") {
            if (state < 0)      extra_btn_input["hatY"] = -1;
            else if (state > 0) extra_btn_input["hatY"] = 1;
            else                extra_btn_input["hatY"] = 0;
            continue;
        }

        if (code == "ABS_Z") {
            extra_btn_input["ABS_Z"] = state;
            continue;
        }
        if (code == "ABS_RZ") {
            extra_btn_input["ABS_RZ"] = state;
            continue;
        }

        auto alias_it = btn_alias.find(code);
        if (alias_it != btn_alias.end()) {
            extra_btn_input[alias_it->second] = state;
            continue;
        }

        auto cmd_it = key_to_cmd_idx.find(code);
        if (cmd_it != key_to_cmd_idx.end()) {
            int idx = cmd_it->second;
            if (code == "BTN_TL" || code == "BTN_TR") {
                joystick_input[idx] = state ? 1.0 : 0.0;
            } else {
                joystick_input[idx] = static_cast<double>(state);
            }
            continue;
        }
    }
}

// D-pad + face buttons -> mode 1..16
void Joystick::update_mode() {
    std::optional<int> new_mode;

    int hatX = extra_btn_input["hatX"];
    int hatY = extra_btn_input["hatY"];
    int X = extra_btn_input["X"];
    int B = extra_btn_input["B"];
    int A = extra_btn_input["A"];
    int Y = extra_btn_input["Y"];

    if (hatY == -1) {
        if (Y)      new_mode = 1;
        else if (B) new_mode = 2;
        else if (A) new_mode = 3;
        else if (X) new_mode = 4;
    } else if (hatX == 1) {
        if (Y)      new_mode = 5;
        else if (B) new_mode = 6;
        else if (A) new_mode = 7;
        else if (X) new_mode = 8;
    } else if (hatY == 1) {
        if (Y)      new_mode = 9;
        else if (B) new_mode = 10;
        else if (A) new_mode = 11;
        else if (X) new_mode = 12;
    } else if (hatX == -1) {
        if (Y)      new_mode = 13;
        else if (B) new_mode = 14;
        else if (A) new_mode = 15;
        else if (X) new_mode = 16;
    }

    if (new_mode.has_value() && !last_new_mode.has_value()) {
        mode_id = new_mode;
    } else {
        mode_id.reset();
    }
    last_new_mode = new_mode;
}

// Both triggers > 0 -> estop
void Joystick::update_estop_flag() {
    bool new_estop =
        (extra_btn_input["ABS_Z"] > 0 && extra_btn_input["ABS_RZ"] > 0);
    estop_flag.store(new_estop);
}

// Right trigger held HOLD_SEC -> sleep
void Joystick::update_sleep_flag() {
    auto now = std::chrono::steady_clock::now();
    bool rz_active = extra_btn_input["ABS_RZ"] > 0;
    if (rz_active) {
        if (!abs_rz_pressed_since.has_value()) {
            abs_rz_pressed_since = now;
        }
        double elapsed =
            std::chrono::duration<double>(now - abs_rz_pressed_since.value()).count();
        bool held = elapsed >= HOLD_SEC;
        sleep_flag.store(held);
    } else {
        abs_rz_pressed_since.reset();
        sleep_flag.store(false);
    }
}

// Left trigger held HOLD_SEC -> wake
void Joystick::update_wake_flag() {
    auto now = std::chrono::steady_clock::now();
    bool z_active = extra_btn_input["ABS_Z"] > 0;
    if (z_active) {
        if (!abs_z_pressed_since.has_value()) {
            abs_z_pressed_since = now;
        }
        double elapsed =
            std::chrono::duration<double>(now - abs_z_pressed_since.value()).count();
        bool held = elapsed >= HOLD_SEC;
        wake_flag.store(held);
    } else {
        abs_z_pressed_since.reset();
        wake_flag.store(false);
    }
}

// Drain events, update state, filter, return output (or throw)
JoystickOutput Joystick::get_cmd() {
    while (true) {
        std::vector<std::pair<std::string,int>> batch;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (event_queue.empty()) {
                break;
            }
            batch = std::move(event_queue.front());
            event_queue.pop_front();
        }
        process_event_batch(batch);
    }

    for (int i = 0; i < TOTAL_IDX_NUM; ++i) {
        double raw_cmd = joystick_input[i] * scales[i];
        if (std::fabs(raw_cmd) < dz_th[i]) {
            raw_cmd = 0.0;
        }

        // 2. 그 다음 LPF
        if (i != mapping["LEFT_BTN"] && i != mapping["RIGHT_BTN"]) {
            double filtered;
            if (raw_cmd == 0.0) {
                filtered =
                    LPF_ALPHA_STOP * robot_prev_cmd[i] +
                    (1.0 - LPF_ALPHA_STOP) * raw_cmd;
            } else {
                filtered =
                    LPF_ALPHA * robot_prev_cmd[i] +
                    (1.0 - LPF_ALPHA) * raw_cmd;
            }

            robot_prev_cmd[i] = filtered;
            robot_cmd[i]      = filtered;
        } else {
            robot_cmd[i] = raw_cmd;
        }

        double maxval = max_cmd[i];
        if (robot_cmd[i] > maxval - 1e-3) {
            robot_cmd[i] = maxval;
        } else if (robot_cmd[i] < -maxval + 1e-3) {
            robot_cmd[i] = -maxval;
        }

        robot_prev_cmd[i] = robot_cmd[i];
    }

    update_mode();
    update_estop_flag();
    update_sleep_flag();
    update_wake_flag();

    if (estop_flag.load()) {
        throw JoystickEstopError("E-stop triggered by joystick input.");
    }
    if (sleep_flag.load()) {
        throw JoystickSleepError("Sleep triggered by joystick input.");
    }

    JoystickOutput output;
    output.cmd_vector = robot_cmd;
    output.mode_id = mode_id;
    output.estop = estop_flag.load();
    output.wake = wake_flag.load();
    output.sleep = sleep_flag.load();
    return output;
}

} // namespace joystick