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

// 파이썬 Mode에서 뽑아온 최소 정보 + 추론 콜백
struct ModeDesc {
    int id = 0;
    std::vector<std::string> stacked_obs_order;
    std::vector<std::string> non_stacked_obs_order;
    std::unordered_map<std::string, std::vector<float>> obs_scale; // per-observation scales (including "command")
    std::vector<float> action_scale;
    int stack_size = 1;
    int cmd_vector_length = 0;
    std::function<std::vector<float>(const std::vector<float>&)> inference;
};

class RL {
public:
    RL();

    void add_mode(const ModeDesc& mode);
    void set_mode(int mode_id);  // mode_id 없으면 호출하지 않음

    std::vector<float> build_state(
        const std::unordered_map<std::string, std::vector<float>>& obs,
        const std::unordered_map<std::string, std::vector<float>>& cmd,
        const std::vector<float>* last_action_opt  // nullptr=미적용
    );

    std::vector<float> select_action(const std::vector<float>& state);

private:
    std::unordered_map<std::string, std::size_t> obs_to_length_;
    const ModeDesc* mode_{nullptr};
    std::vector<ModeDesc> modes_;

    std::vector<float> single_frame_;
    std::size_t single_frame_len_{0};

    std::vector<float> state_;
    std::vector<float> last_action_;
    std::vector<float> scaled_action_;

    // 캐시 (모드 종속)
    const std::vector<float>* cached_action_scale_{nullptr};
    std::size_t last_action_len_{0};
    const std::vector<std::string>* cached_stacked_order_{nullptr};
    const std::vector<std::string>* cached_non_stacked_order_{nullptr};
    int cached_stack_size_{1};
    const std::unordered_map<std::string, std::vector<float>>* cached_obs_scale_map_{nullptr};
    std::function<std::vector<float>(const std::vector<float>&)> infer_;

    void ensure_mode_() const;
    std::size_t get_obs_len_(const std::string& key) const;
    const std::vector<float>& get_obs_scale_(const std::string& key, std::size_t len) const;

    mutable std::vector<float> padding_buffer_;
};

} // namespace rl
