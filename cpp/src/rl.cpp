#include <algorithm>

#include "rl.hpp"

namespace rl {

RL::RL() {
    modes_.reserve(128);
    // Map observation keys to their expected lengths
    obs_to_length_ = {
        {"dof_pos",     robot::NUM_LIMB_MOTORS},
        {"dof_vel",     robot::NUM_MOTORS},
        {"lin_vel",     3},
        {"ang_vel",     3},
        {"proj_grav",   3},
        {"last_action", robot::NUM_MOTORS},
        {"height_map",  robot::HEIGHT_MAP_SIZE}
    };
    last_action_.assign(robot::NUM_MOTORS, 0.0f);
    last_obs_.clear();
    last_cmd_.clear();
    last_last_action_opt_ = nullptr;
    scaled_action_.assign(robot::NUM_MOTORS, 0.0f);
    single_frame_len_ = 0;
}

// Add or update a mode configuration
void RL::addMode(const ModeDesc& mode) {
    if(mode.id < 1 || mode.id > MAX_MODE_ID){
        return;
    }
    for (auto& m : modes_) {
        if (m.id == mode.id) {
            m = mode;
            if (cur_mode_ && cur_mode_->id == mode.id) cur_mode_ = &m;
            return;
        }
    }
    modes_.push_back(mode);
}

// Activate a mode and configure state/action buffers
void RL::setMode(int mode_id) {
    if (mode_id < 1 || mode_id > MAX_MODE_ID) return;
    for (auto& m : modes_) {
        if (m.id == mode_id) {
            obs_to_length_["command"] = static_cast<std::size_t>(m.cmd_vector_length);

            // Set pointers to mode-specific configurations
            cur_mode_              = &m;
            action_scale_          = &m.action_scale;
            stacked_obs_order_     = &m.stacked_obs_order;
            non_stacked_obs_order_ = &m.non_stacked_obs_order;
            obs_scale_             = &m.obs_scale;
            stack_size_            = m.stack_size;
            inference_             = m.inference;

            // Calculate state length 
            std::size_t single_len = 0;
            for (const auto& k : m.stacked_obs_order) {
                single_len += getObsLen(k);
            }
            single_frame_len_ = single_len;
            single_frame_.assign(single_frame_len_, 0.0f);

            std::size_t state_len = single_len * static_cast<std::size_t>(m.stack_size);
            for (const auto& k : m.non_stacked_obs_order) {
                state_len += getObsLen(k);
            }
            state_.assign(state_len, 0.0f);

            // Check the valid action_scale length
            if (action_scale_->size() < robot::NUM_MOTORS) {
                throw std::runtime_error("action_scale shorter than last_action length");
            }

            // Initialize state history 
            for(int i=0; i<stack_size_; ++i){
                buildState(last_obs_, last_cmd_, last_last_action_opt_, false);
            }

            return;
        }
    }
}

// Build state vector from observations, commands, and last action
std::vector<float> RL::buildState(
    const std::unordered_map<std::string, std::vector<float>>& obs,
    const std::unordered_map<std::string, std::vector<float>>& cmd,
    const std::vector<float>* last_action_opt,
    bool check_mode
) {
    if(check_mode) ensureMode();

    // Initialize state history on first call
    if(!has_history_){
        has_history_ = true;
        for(int i =0; i<stack_size_; ++i){
            buildState(last_obs_, last_cmd_, nullptr, false);
        }
    }

    // Update last action if provided
    if (last_action_opt) {
        const auto& v = *last_action_opt;
        if (v.size() != robot::NUM_MOTORS) throw std::runtime_error("scaled_last_action length mismatch");
        last_action_ = v;
    }

    // Build single frame from stacked observations
    std::size_t i = 0;
    for (const auto& obs_key : *stacked_obs_order_) {
        std::size_t obs_len = getObsLen(obs_key);
        const std::vector<float>& scale = getObsScale(obs_key, obs_len);
        const std::vector<float>* src = nullptr;

        // Route observation source based on key type
        if (obs_key == "command") {
            // Command vectors are stored under "cmd_vector" in cmd map
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (obs_key == "last_action") {
            src = &last_action_;
        } else {
            auto ito = obs.find(obs_key);
            if (ito != obs.end()) src = &ito->second;
        }

        // Fill frame with scaled values or carry forward previous values
        if (!src) {
            for (std::size_t k = 0; k < obs_len; ++k) {
                single_frame_[i] = state_[i];
                ++i;
            }
        } else {
            for (std::size_t j = 0; j < obs_len; ++j){
                single_frame_[i] = (*src)[j] * scale[j];
                ++i;
            }
        }
    }

    // Shift frame history (temporal stacking)
    const std::size_t L = single_frame_len_;
    const int S = stack_size_;
    if (S > 1) {
        for (int k = S - 1; k > 0; --k) {
            std::copy(state_.begin() + (k - 1) * L,
                      state_.begin() + k * L,
                      state_.begin() + k * L);
        }
    }
    std::copy(single_frame_.begin(), single_frame_.end(), state_.begin());

    // Append non-stacked observations to state
    std::size_t base = L * static_cast<std::size_t>(S);
    for (const auto& obs_key : *non_stacked_obs_order_) {
        std::size_t obs_len = getObsLen(obs_key);
        const std::vector<float>& scale = getObsScale(obs_key, obs_len);
        const std::vector<float>* src = nullptr;

        if (obs_key == "command") {
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (obs_key == "last_action") {
            src = &last_action_;
        } else {
            auto ito = obs.find(obs_key);
            if (ito != obs.end()) src = &ito->second;
        }

        if (!src) {
            base += obs_len;
        } else {
            for (std::size_t j = 0; j < obs_len; ++j){
            state_[base + j] = (*src)[j] * scale[j];
            }
            base += obs_len;
        }
    }

    last_obs_ = obs;
    last_cmd_ = cmd;
    last_last_action_opt_ = last_action_opt;

    return state_;
}

// Run inference and scale action output
std::vector<float> RL::selectAction(const std::vector<float>& state) {
    ensureMode();
    last_action_  = inference_(state);

    // Scale actions by mode-specific factors
    for (std::size_t i = 0; i < robot::NUM_MOTORS; ++i) {
        scaled_action_[i] = last_action_[i] * (*action_scale_)[i];
    }
    return scaled_action_;
}

// Verify mode is set before operations
void RL::ensureMode() const {
    if (!cur_mode_) throw std::runtime_error("Mode is not set. Call set_mode() first.");
}

// Lookup observation length by key
std::size_t RL::getObsLen(const std::string& key) const {
    auto it = obs_to_length_.find(key);
    if (it == obs_to_length_.end()) throw std::runtime_error("Unknown observation key: " + key);
    return it->second;
}

// Get scaling factors for observation, pad with 1.0 if needed
const std::vector<float>& RL::getObsScale(const std::string& key, std::size_t len) const {
    auto it = obs_scale_->find(key);
    if (it != obs_scale_->end() && it->second.size() >= len) return it->second;

    // Create padded scale vector if missing or too short
    padding_buffer_.assign(len, 1.0f);
    if (it != obs_scale_->end()) {
        const auto& v = it->second;
        for (std::size_t i = 0; i < v.size() && i < len; ++i) padding_buffer_[i] = v[i];
    }
    return padding_buffer_;
}

} // namespace rl