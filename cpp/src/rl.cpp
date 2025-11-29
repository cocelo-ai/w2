#include <algorithm>

#include "rl.hpp"
#include "robot.hpp"

namespace rl {

RL::RL() {
    modes.reserve(128);
    // Map observation keys to their expected lengths
    obs_to_length = {
        {"dof_pos",     robot::NUM_LIMB_MOTORS},
        {"dof_vel",     robot::NUM_MOTORS},
        {"lin_vel",     3},
        {"ang_vel",     3},
        {"proj_grav",   3},
        {"last_action", robot::NUM_MOTORS},
        {"height_map",  robot::HEIGHT_MAP_SIZE}
    };
    last_action.assign(robot::NUM_MOTORS, 0.0f);
    last_obs.clear();
    last_cmd.clear();
    last_last_action_opt = nullptr;
    scaled_action.assign(robot::NUM_MOTORS, 0.0f);
    single_frame_len = 0;
}

// Add or update a mode configuration
void RL::add_mode(const ModeDesc& mode) {
    if(mode.id < 1 || mode.id > MAX_MODE_ID){
        return;
    }
    for (auto& m : modes) {
        if (m.id == mode.id) {
            m = mode;
            if (cur_mode && cur_mode->id == mode.id) cur_mode = &m;
            return;
        }
    }
    modes.push_back(mode);
}

// Activate a mode and configure state/action buffers
void RL::set_mode(int mode_id) {
    if (mode_id < 1 || mode_id > MAX_MODE_ID) return;
    for (auto& m : modes) {
        if (m.id == mode_id) {
            obs_to_length["command"] = static_cast<std::size_t>(m.cmd_vector_length);

            // Set pointers to mode-specific configurations
            cur_mode              = &m;
            action_scale          = &m.action_scale;
            stacked_obs_order     = &m.stacked_obs_order;
            non_stacked_obs_order = &m.non_stacked_obs_order;
            obs_scale             = &m.obs_scale;
            stack_size            = m.stack_size;
            inference             = m.inference;

            // Calculate state length 
            std::size_t single_len = 0;
            for (const auto& k : m.stacked_obs_order) {
                single_len += get_obs_len(k);
            }
            single_frame_len = single_len;
            single_frame.assign(single_frame_len, 0.0f);

            std::size_t state_len = single_len * static_cast<std::size_t>(m.stack_size);
            for (const auto& k : m.non_stacked_obs_order) {
                state_len += get_obs_len(k);
            }
            state.assign(state_len, 0.0f);

            // Check the valid action_scale length
            if (action_scale->size() < robot::NUM_MOTORS) {
                throw std::runtime_error("action_scale shorter than last_action length");
            }

            // Initialize state history 
            for(int i=0; i<stack_size; ++i){
                build_state(last_obs, last_cmd, last_last_action_opt, false);
            }

            return;
        }
    }
}

// Build state vector from observations, commands, and last action
std::vector<float> RL::build_state(
    const std::unordered_map<std::string, std::vector<float>>& obs,
    const std::unordered_map<std::string, std::vector<float>>& cmd,
    const std::vector<float>* last_action_opt,
    bool check_mode
) {
    if(check_mode) ensure_mode();

    // Initialize state history on first call
    static std::int32_t build_count = 0;
    if(build_count == 0){
        build_count++;
        for(int i =0; i<stack_size; ++i){
            build_state(last_obs, last_cmd, nullptr, false);
        }
    }

    // Update last action if provided
    if (last_action_opt) {
        const auto& v = *last_action_opt;
        if (v.size() != robot::NUM_MOTORS) throw std::runtime_error("scaled_last_action length mismatch");
        last_action = v;
    }

    // Build single frame from stacked observations
    std::size_t i = 0;
    for (const auto& obs_key : *stacked_obs_order) {
        std::size_t obs_len = get_obs_len(obs_key);
        const std::vector<float>& scale = get_obs_scale(obs_key, obs_len);
        const std::vector<float>* src = nullptr;

        // Route observation source based on key type
        if (obs_key == "command") {
            // Command vectors are stored under "cmd_vector" in cmd map
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (obs_key == "last_action") {
            src = &last_action;
        } else {
            auto ito = obs.find(obs_key);
            if (ito != obs.end()) src = &ito->second;
        }

        // Fill frame with scaled values or carry forward previous values
        if (!src) {
            for (std::size_t k = 0; k < obs_len; ++k) {
                single_frame[i] = state[i];
                ++i;
            }
        } else {
            for (std::size_t j = 0; j < obs_len; ++j){
                single_frame[i] = (*src)[j] * scale[j];
                ++i;
            }
        }
    }

    // Shift frame history (temporal stacking)
    const std::size_t L = single_frame_len;
    const int S = stack_size;
    if (S > 1) {
        for (int k = S - 1; k > 0; --k) {
            std::copy(state.begin() + (k - 1) * L,
                      state.begin() + k * L,
                      state.begin() + k * L);
        }
    }
    std::copy(single_frame.begin(), single_frame.end(), state.begin());

    // Append non-stacked observations to state
    std::size_t base = L * static_cast<std::size_t>(S);
    for (const auto& obs_key : *non_stacked_obs_order) {
        std::size_t obs_len = get_obs_len(obs_key);
        const std::vector<float>& scale = get_obs_scale(obs_key, obs_len);
        const std::vector<float>* src = nullptr;

        if (obs_key == "command") {
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (obs_key == "last_action") {
            src = &last_action;
        } else {
            auto ito = obs.find(obs_key);
            if (ito != obs.end()) src = &ito->second;
        }

        if (!src) {
            base += obs_len;
        } else {
            for (std::size_t j = 0; j < obs_len; ++j){
            state[base + j] = (*src)[j] * scale[j];
            }
            base += obs_len;
        }
    }

    last_obs = obs;
    last_cmd = cmd;
    last_last_action_opt = last_action_opt;

    return state;
}

// Run inference and scale action output
std::vector<float> RL::select_action(const std::vector<float>& state) {
    ensure_mode();
    last_action  = inference(state);

    // Scale actions by mode-specific factors
    for (std::size_t i = 0; i < robot::NUM_MOTORS; ++i) {
        scaled_action[i] = last_action[i] * (*action_scale)[i];
    }
    return scaled_action;
}

// Verify mode is set before operations
void RL::ensure_mode() const {
    if (!cur_mode) throw std::runtime_error("Mode is not set. Call set_mode() first.");
}

// Lookup observation length by key
std::size_t RL::get_obs_len(const std::string& key) const {
    auto it = obs_to_length.find(key);
    if (it == obs_to_length.end()) throw std::runtime_error("Unknown observation key: " + key);
    return it->second;
}

// Get scaling factors for observation, pad with 1.0 if needed
const std::vector<float>& RL::get_obs_scale(const std::string& key, std::size_t len) const {
    auto it = obs_scale->find(key);
    if (it != obs_scale->end() && it->second.size() >= len) return it->second;

    // Create padded scale vector if missing or too short
    padding_buffer.assign(len, 1.0f);
    if (it != obs_scale->end()) {
        const auto& v = it->second;
        for (std::size_t i = 0; i < v.size() && i < len; ++i) padding_buffer[i] = v[i];
    }
    return padding_buffer;
}

} // namespace rl