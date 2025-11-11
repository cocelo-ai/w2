#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// ------------------------- Exceptions -------------------------
class ModeConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ------------------------- Helpers -------------------------
std::unordered_map<std::string, std::size_t> get_obs_to_length_map();

// number or per-dimension vector (float ONLY)
using ScaleSpec = std::variant<float, std::vector<float>>;

// 외부 정책 인터페이스 (순수 C++)
class IPolicy {
public:
    virtual ~IPolicy() = default;
    virtual std::vector<float> inference(const std::vector<float>& state) = 0;
};

struct ModeConfig {
    int id = 0;
    std::vector<std::string> stacked_obs_order;
    std::vector<std::string> non_stacked_obs_order;
    std::unordered_map<std::string, ScaleSpec> obs_scale; // scalar or vector<float> per key
    std::optional<ScaleSpec> action_scale;                // optional; defaults to 1.0 for last_action
    int stack_size = 1;
    std::string policy_path;
    std::string policy_type = "MLP"; // "MLP" or "LSTM"
    int cmd_vector_length = 0;
    // command scale is now specified via obs_scale["command"] instead of a separate field
};

// ------------------------- Mode (float-only) -------------------------
class Mode {
public:
    using ObsLengthMap = std::unordered_map<std::string, std::size_t>;

    explicit Mode(const ModeConfig& cfg);

    // 외부에서 정책 주입(선택): 생성자에서 이미 자동 주입됨
    void set_policy(std::shared_ptr<IPolicy> policy);

    // 1D 관측 → 액션 (float ONLY)
    std::vector<float> inference(const std::vector<float>& obs1d);

    // 읽기용 메타
    const ObsLengthMap& obs_lengths() const { return obs_to_length_; }
    int id() const { return id_; }
    int stack_size() const { return stack_size_; }
    int state_length() const { return static_cast<int>(state_len_); }
    int action_length() const { return static_cast<int>(last_action_len_); }
    const std::string& policy_type() const { return policy_type_; }
    const std::string& policy_path() const { return policy_path_; }

    const std::vector<std::string>& stacked_obs_order() const { return stacked_obs_order_; }
    const std::vector<std::string>& non_stacked_obs_order() const { return non_stacked_obs_order_; }

    // scales (읽기 전용)
    const std::vector<float>& action_scale() const { return action_scale_; }
    // command scale is now part of obs_scale map; no separate cmd_scale()
    const std::unordered_map<std::string, std::vector<float>>& obs_scale() const { return obs_scale_norm_; }

    // ★ 새로 추가: cmd_vector_length getter (파이썬에서 바로 읽을 수 있도록)
    int cmd_vector_length() const { return cmd_vector_length_; }

private:
    static std::vector<float> normalize_scale(const ScaleSpec& spec, std::size_t length);

    // 스케일 적용
    void apply_scales_inplace(std::vector<float>& state) const;

    // 레이아웃 유효성 검사 및 길이 계산
    void compute_state_layout_();

    // 소문자화
    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return s;
    }

private:
    // 구성 메타
    ObsLengthMap obs_to_length_;
    int id_{0};
    std::vector<std::string> stacked_obs_order_;
    std::vector<std::string> non_stacked_obs_order_;
    std::unordered_map<std::string, std::vector<float>> obs_scale_norm_; // per key normalized
    std::vector<float> action_scale_;                                    // length = last_action
    int stack_size_{1};
    std::string policy_path_;
    std::string policy_type_;
    int cmd_vector_length_{0};
    // cmd_scale_norm_ removed; command scales are stored in obs_scale_norm_ under key "command"

    // 파생 메타
    std::size_t state_len_{0};
    std::size_t last_action_len_{0};

    // 정책 (주입)
    std::shared_ptr<IPolicy> policy_;
};
