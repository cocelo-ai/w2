#pragma once
// Pure C++ header: no pybind / Python / onnxruntime includes.

#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <functional>

namespace rl {

struct ModeDesc {
    int id = 0;
    std::vector<std::string> stacked_obs_order;
    std::vector<std::string> non_stacked_obs_order;
    std::unordered_map<std::string, std::vector<float>> obs_scale;
    std::vector<float> action_scale;
    int stack_size = 1;
    int cmd_vector_length = 0;
    std::function<std::vector<float>(const std::vector<float>&)> inference;
};

class RL {
public:
    RL();
    void add_mode(const ModeDesc& mode);
    void set_mode(int mode_id);
    std::vector<float> build_state(
        const std::unordered_map<std::string, std::vector<float>>& obs,
        const std::unordered_map<std::string, std::vector<float>>& cmd,
        const std::vector<float>* last_action_opt
    );
    std::vector<float> select_action(const std::vector<float>& state);

private:
    std::unordered_map<std::string, std::size_t> obs_to_length;
    const ModeDesc* cur_mode{nullptr};
    std::vector<ModeDesc> modes;

    std::vector<float> single_frame;
    std::size_t single_frame_len{0};
    int stack_size{1};

    std::vector<float> state;
    std::vector<float> last_action;
    std::vector<float> scaled_action;

    std::size_t last_action_len{0};
    const std::vector<float>* action_scale{nullptr};
    const std::vector<std::string>* stacked_obs_order{nullptr};
    const std::vector<std::string>* non_stacked_obs_order{nullptr};
    const std::unordered_map<std::string, std::vector<float>>* obs_scale{nullptr};
    std::function<std::vector<float>(const std::vector<float>&)> inference;

    void ensure_mode_() const;
    std::size_t get_obs_len(const std::string& key) const;
    const std::vector<float>& get_obs_scale(const std::string& key, std::size_t len) const;
    mutable std::vector<float> padding_buffer;
};

} // namespace rl