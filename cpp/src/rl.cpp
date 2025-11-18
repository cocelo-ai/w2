#include "rl.hpp"
#include <algorithm>

namespace rl {

RL::RL() {
    modes.reserve(128);
    obs_to_length = {
        {"dof_pos", 6}, {"dof_vel", 8}, {"lin_vel", 3},
        {"ang_vel", 3}, {"proj_grav", 3}, {"last_action", 8},
        {"height_map", 144}
    };
    last_action_len = obs_to_length.at("last_action");
    last_action.assign(last_action_len, 0.0f);
    scaled_action.assign(last_action_len, 0.0f);
    single_frame_len = 0;
}

void RL::add_mode(const ModeDesc& mode) {
    if(mode.id < 1 || mode.id > 16){
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

void RL::set_mode(int mode_id) {
    if (mode_id < 1 || mode_id > 16) return;
    for (auto& m : modes) {
        if (m.id == mode_id) {
            // Assign command's length
            obs_to_length["command"] = static_cast<std::size_t>(m.cmd_vector_length);

            // Assign state's length
            std::size_t single_len = 0;
            for (const auto& k : m.stacked_obs_order){
                single_len += get_obs_len(k);
            }
            single_frame_len = single_len;
            single_frame.assign(single_frame_len, 0.0f);

            std::size_t state_len = single_len * static_cast<std::size_t>(m.stack_size);
            for (const auto& k : m.non_stacked_obs_order){
                state_len += get_obs_len(k);
            }
            state.assign(state_len, 0.0f);

            // Assign action's length
            last_action.assign(last_action_len, 0.0f);
            scaled_action.assign(last_action_len, 0.0f);

            // Assign the mode property
            cur_mode              = &m;
            action_scale          = &m.action_scale;
            stacked_obs_order     = &m.stacked_obs_order;
            non_stacked_obs_order = &m.non_stacked_obs_order;
            obs_scale             = &m.obs_scale;
            stack_size            = m.stack_size;
            inference             = m.inference;

            if (action_scale->size() < last_action_len) {
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
        if (v.size() != last_action_len) throw std::runtime_error("scaled_last_action length mismatch");
        last_action = v;
    }

    std::size_t i = 0;
    for (const auto& obs_key : *stacked_obs_order) {
        // Get Observation and it's length & scale.
        std::size_t obs_len = get_obs_len(obs_key);
        const std::vector<float>& scale = get_obs_scale(obs_key, obs_len);
        const std::vector<float>* src = nullptr;
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

    std::size_t base = L * static_cast<std::size_t>(S);
    for (const auto& obs_key : *non_stacked_obs_order) {
        // Get Observation and it's length & scale.
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
    return state;
}

std::vector<float> RL::select_action(const std::vector<float>& state) {
    ensure_mode_();
    last_action  = inference(state);

    const std::size_t n = last_action_len;
    for (std::size_t i = 0; i < n; ++i) {
        scaled_action[i] = last_action[i] * (*action_scale)[i];
    }
    return scaled_action;
}

void RL::ensure_mode_() const {
    if (!cur_mode) throw std::runtime_error("Mode is not set. Call set_mode() first.");
}

std::size_t RL::get_obs_len(const std::string& key) const {
    auto it = obs_to_length.find(key);
    if (it == obs_to_length.end()) throw std::runtime_error("Unknown observation key: " + key);
    return it->second;
}

const std::vector<float>& RL::get_obs_scale(const std::string& key, std::size_t len) const {
    auto it = obs_scale->find(key);
    if (it != obs_scale->end() && it->second.size() >= len) return it->second;

    padding_buffer.assign(len, 1.0f);
    if (it != obs_scale->end()) {
        const auto& v = it->second;
        for (std::size_t i = 0; i < v.size() && i < len; ++i) padding_buffer[i] = v[i];
    }
    return padding_buffer;
}

} // namespace rl