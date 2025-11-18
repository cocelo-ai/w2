#include <sstream>
#include <algorithm>
#include <filesystem>

#include "mode.hpp"
#include "onnxpolicy.hpp"  

using namespace onnxpolicy;

// C++ onnxpolicy를 IPolicy로 어댑팅 (float-only)
template <typename Impl>
class PolicyAdapter final : public IPolicy {
public:
    explicit PolicyAdapter(const std::string& path) : impl_(path) {}
    std::vector<float> inference(const std::vector<float>& state) override {
        return impl_.inference(state);
    }
private:
    Impl impl_;
};

// ------------------------- Helpers -------------------------
std::unordered_map<std::string, std::size_t> get_obs_to_length_map() {
    return {
        {"dof_pos", 6},
        {"dof_vel", 8},
        {"lin_vel", 3},
        {"ang_vel", 3},
        {"proj_grav", 3},
        {"last_action", 8},
        {"height_map", 144},
        {"command", 0}
    };
}

static bool is_regular_file_existing(const std::string& p) {
    namespace fs = std::filesystem;
    fs::path path(p);
    return fs::exists(path) && fs::is_regular_file(path);
}

std::vector<float> Mode::normalize_scale(const ScaleSpec& spec, std::size_t length) {
    std::vector<float> out;
    if (std::holds_alternative<float>(spec)) {
        out.assign(length, std::get<float>(spec));
    } else {
        const auto& v = std::get<std::vector<float>>(spec);
        if (v.size() != length) {
            std::ostringstream oss;
            oss << "scale length mismatch, got: " << v.size() << ", expected: " << length;
            throw ModeConfigError(oss.str());
        }
        out = v;
    }
    return out;
}

// ------------------------- Mode -------------------------
Mode::Mode(const ModeConfig& cfg)
{
    // id
    if (cfg.id < 1 || cfg.id > 16) {
        throw ModeConfigError("'id' must be between >=1 and <=16, but got " + std::to_string(cfg.id));
    }
    id_ = cfg.id;

    // orders
    stacked_obs_order_     = cfg.stacked_obs_order;
    non_stacked_obs_order_ = cfg.non_stacked_obs_order;

    // obs length map
    obs_to_length_ = get_obs_to_length_map();

    // cmd length
    if (cfg.cmd_vector_length < 0) {
        throw ModeConfigError("cmd_vector_length must be >= 0, but got " + std::to_string(cfg.cmd_vector_length));
    }
    cmd_vector_length_ = cfg.cmd_vector_length;
    obs_to_length_["command"] = static_cast<std::size_t>(cmd_vector_length_);

    // stack size
    if (cfg.stack_size < 1) {
        throw ModeConfigError("stack_size must be >= 1, but got " + std::to_string(cfg.stack_size));
    }
    stack_size_ = cfg.stack_size;

    // policy path
    if (cfg.policy_path.empty()) throw ModeConfigError("policy_path is required but missing");
    policy_path_ = cfg.policy_path;
    if (!is_regular_file_existing(policy_path_)) {
        throw std::runtime_error("policy_path not found or not a file: " + policy_path_);
    }
    {
        std::string ext;
        auto dot = policy_path_.find_last_of('.');
        if (dot != std::string::npos) ext = policy_path_.substr(dot);
        std::string lower = ext;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
        if (lower != ".onnx") {
            throw ModeConfigError("policy_path must be a .onnx file, but got '" + ext + "'");
        }
    }

    // policy type (그대로 보존)
    policy_type_ = cfg.policy_type;
    {
        std::string tl = to_lower(policy_type_);
        if (!(tl == "mlp" || tl == "lstm")) {
            throw ModeConfigError("Unsupported policy_type: " + policy_type_);
        }
    }

    // obs_scale normalization (including "command")
    // For each observation key used in stacked and non-stacked orders, populate obs_scale_norm_.
    // If a scale is provided in cfg.obs_scale, normalize it to the correct length; otherwise, default to 1.0.
    auto normalize_for_key = [&](const std::string& k) {
        auto itLen = obs_to_length_.find(k);
        if (itLen == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key: " + k);
        }
        const std::size_t len = itLen->second;
        auto itScale = cfg.obs_scale.find(k);
        if (itScale == cfg.obs_scale.end()) {
            obs_scale_norm_[k] = std::vector<float>(len, 1.0f);
        } else {
            try {
                obs_scale_norm_[k] = normalize_scale(itScale->second, len);
            }
            catch (const ModeConfigError& e) {
                throw ModeConfigError("obs_scale for '" + k + "': " + std::string(e.what()));
            }
        }
    };
    // Normalize scales for stacked observations
    for (const auto& k : stacked_obs_order_) {
        normalize_for_key(k);
    }
    // Normalize scales for non-stacked observations
    for (const auto& k : non_stacked_obs_order_) {
        normalize_for_key(k);
    }

    // action_scale (length of last_action)
    last_action_len_ = obs_to_length_.at("last_action");
    if (cfg.action_scale.has_value()) {
        try {
            action_scale_ = normalize_scale(*cfg.action_scale, last_action_len_);
        } catch (const ModeConfigError& e) {
            throw ModeConfigError(std::string("action_scale: ") + e.what());
        }
    } else {
        action_scale_.assign(last_action_len_, 1.0f);
    }

    // 상태 길이 계산
    compute_state_layout_();

    // ★ 자동 policy 주입 (onnxpolicy 활용; float I/O)
    {
        const std::string tl = to_lower(policy_type_);
        if (tl == "mlp") {
            policy_ = std::make_shared<PolicyAdapter<MLPPolicy>>(policy_path_);
        } else { // "lstm"
            policy_ = std::make_shared<PolicyAdapter<LSTMPolicy>>(policy_path_);
        }
        // 더미 상태로 길이 검증
        std::vector<float> dummy(static_cast<std::size_t>(state_len_), 0.0f);
        auto out = policy_->inference(dummy);
        if (out.size() != last_action_len_) {
            std::ostringstream oss;
            oss << "Policy 'inference' output length mismatch: got " << out.size()
                << ", expected " << last_action_len_ << " ('last_action' length)";
            throw ModeConfigError(oss.str());
        }
    }
}

void Mode::compute_state_layout_() {
    // 총 state 길이 = (stacked 부분 * stack_size) + non_stacked 부분
    std::size_t total = 0;
    for (const auto& k : stacked_obs_order_) {
        auto it = obs_to_length_.find(k);
        if (it == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key in stacked_obs_order: " + k);
        }
        total += it->second;
    }
    total *= static_cast<std::size_t>(stack_size_);

    for (const auto& k : non_stacked_obs_order_) {
        auto it = obs_to_length_.find(k);
        if (it == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key in non_stacked_obs_order: " + k);
        }
        total += it->second;
    }
    state_len_ = total;
}

std::vector<float> Mode::inference(const std::vector<float>& state1d) {
    if (state1d.size() != state_len_) {
        std::ostringstream oss;
        oss << "state length mismatch: got " << state1d.size() << ", expected " << state_len_;
        throw std::runtime_error(oss.str());
    }
    std::vector<float> action = policy_->inference(state1d);

    return action;
}