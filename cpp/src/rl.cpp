#include "rl.hpp"
#include <algorithm>

namespace rl {

RL::RL() {
    obs_to_length_ = {
        {"dof_pos", 6}, {"dof_vel", 8}, {"lin_vel", 3},
        {"ang_vel", 3}, {"proj_grav", 3}, {"last_action", 8},
        {"height_map", 144}
    };
    last_action_len_ = obs_to_length_.at("last_action");
    last_action_.assign(last_action_len_, 0.0f);
    scaled_action_.assign(last_action_len_, 0.0f);
    single_frame_len_ = 0;
}

void RL::add_mode(const ModeDesc& mode) {
    for (auto& m : modes_) {
        if (m.id == mode.id) {
            m = mode;
            if (mode_ && mode_->id == mode.id) mode_ = &m;
            return;
        }
    }
    modes_.push_back(mode);
}

void RL::set_mode(int mode_id) {
    if (mode_id <= 0) return;
    for (auto& m : modes_) {
        if (m.id == mode_id) {
            obs_to_length_["command"] = static_cast<std::size_t>(m.cmd_vector_length);

            std::size_t single_len = 0;
            for (const auto& k : m.stacked_obs_order) single_len += get_obs_len_(k);
            single_frame_len_ = single_len;
            single_frame_.assign(single_frame_len_, 0.0f);

            std::size_t state_len = single_len * static_cast<std::size_t>(m.stack_size);
            for (const auto& k : m.non_stacked_obs_order) state_len += get_obs_len_(k);

            state_.assign(state_len, 0.0f);
            last_action_.assign(last_action_len_, 0.0f);
            scaled_action_.assign(last_action_len_, 0.0f);

            mode_ = &m;

            cached_action_scale_      = &m.action_scale;
            // command scale is stored inside obs_scale map; no separate cmd_scale
            cached_stacked_order_     = &m.stacked_obs_order;
            cached_non_stacked_order_ = &m.non_stacked_obs_order;
            cached_stack_size_        = m.stack_size;
            cached_obs_scale_map_     = &m.obs_scale;
            infer_                    = m.inference;

            if (cached_action_scale_->size() < last_action_len_) {
                throw std::runtime_error("action_scale shorter than last_action length");
            }
            return;
        }
    }
}

std::vector<float> RL::build_state(
    const std::unordered_map<std::string, std::vector<float>>& obs,
    const std::unordered_map<std::string, std::vector<float>>& cmd,
    const std::vector<float>* last_action_opt
) {
    ensure_mode_();

    if (last_action_opt) {
        const auto& v = *last_action_opt;
        if (v.size() != last_action_len_) throw std::runtime_error("scaled_last_action length mismatch");
        last_action_ = v;
    }

    std::size_t i = 0;
    for (const auto& key : *cached_stacked_order_) {
        std::size_t obs_len = get_obs_len_(key);
        // Retrieve per-element scale; command is now handled via obs_scale map
        const std::vector<float>& scale = get_obs_scale_(key, obs_len);

        const std::vector<float>* src = nullptr;
        if (key == "command") {
            // Command vectors are stored under "cmd_vector" in cmd map
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (key == "last_action") {
            src = &last_action_;
        } else {
            auto ito = obs.find(key);
            if (ito != obs.end()) src = &ito->second;
        }

        if (!src && key != "last_action") {
            for (std::size_t k = 0; k < obs_len; ++k) {
                single_frame_[i] = state_[i];
                ++i;
            }
        } else {
            const std::vector<float>& v = src ? *src : last_action_;
            for (std::size_t j = 0; j < obs_len; ++j) single_frame_[i++] = v[j] * scale[j];
        }
    }

    const std::size_t L = single_frame_len_;
    const int S = cached_stack_size_;
    if (S > 1) {
        for (int k = S - 1; k > 0; --k) {
            std::copy(state_.begin() + (k - 1) * L,
                      state_.begin() + k * L,
                      state_.begin() + k * L);
        }
    }
    std::copy(single_frame_.begin(), single_frame_.end(), state_.begin());

    std::size_t base = L * static_cast<std::size_t>(S);
    for (const auto& key : *cached_non_stacked_order_) {
        std::size_t obs_len = get_obs_len_(key);
        // Retrieve per-element scale; command is now handled via obs_scale map
        const std::vector<float>& scale = get_obs_scale_(key, obs_len);

        const std::vector<float>* src = nullptr;
        if (key == "command") {
            auto itc = cmd.find("cmd_vector");
            if (itc != cmd.end()) src = &itc->second;
        } else if (key == "last_action") {
            src = &last_action_;
        } else {
            auto ito = obs.find(key);
            if (ito != obs.end()) src = &ito->second;
        }

        if (!src && key != "last_action") {
            base += obs_len;
        } else {
            const std::vector<float>& v = src ? *src : last_action_;
            for (std::size_t j = 0; j < obs_len; ++j) state_[base + j] = v[j] * scale[j];
            base += obs_len;
        }
    }

    return state_;
}

std::vector<float> RL::select_action(const std::vector<float>& state) {
    ensure_mode_();
    if (!infer_) throw std::runtime_error("inference callback not set");
    std::vector<float> action = infer_(state);

    const std::size_t n = last_action_len_;
    for (std::size_t i = 0; i < n; ++i) {
        scaled_action_[i] = action[i] * (*cached_action_scale_)[i];
    }
    last_action_ = std::move(action);
    return scaled_action_;
}

void RL::ensure_mode_() const {
    if (!mode_) throw std::runtime_error("Mode is not set. Call set_mode() first.");
}

std::size_t RL::get_obs_len_(const std::string& key) const {
    auto it = obs_to_length_.find(key);
    if (it == obs_to_length_.end()) throw std::runtime_error("Unknown observation key: " + key);
    return it->second;
}

const std::vector<float>& RL::get_obs_scale_(const std::string& key, std::size_t len) const {
    auto it = cached_obs_scale_map_->find(key);
    if (it != cached_obs_scale_map_->end() && it->second.size() >= len) return it->second;

    padding_buffer_.assign(len, 1.0f);
    if (it != cached_obs_scale_map_->end()) {
        const auto& v = it->second;
        for (std::size_t i = 0; i < v.size() && i < len; ++i) padding_buffer_[i] = v[i];
    }
    return padding_buffer_;
}

} // namespace rl
