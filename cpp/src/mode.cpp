#include <sstream>
#include <algorithm>
#include <filesystem>

#include "mode.hpp"

using namespace onnxpolicy;


// Adapts a concrete onnxpolicy implementation to the IPolicy interface
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

// Checks if a regular file exists at the given path
static bool isRegularFileExisting(const std::string& p) {
    namespace fs = std::filesystem;
    fs::path path(p);
    return fs::exists(path) && fs::is_regular_file(path);
}

// Normalizes a ScaleSpec into a vector of given length:
//  - if spec is a scalar, broadcast it
//  - if spec is a vector, validate its length
//  - otherwise,           throw ModeConfigError
std::vector<float> Mode::normalizeScale(const ScaleSpec& spec, std::size_t length) {
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
    // ---- id ----
    if (cfg.id < 1 || cfg.id > rl::MAX_MODE_ID) {
        throw ModeConfigError("'id' must be between 1 and " + std::to_string(rl::MAX_MODE_ID) + ", but got " + std::to_string(cfg.id));
    }
    id_ = cfg.id;

    // Observation ordering (stacked / non-stacked parts)
    stacked_obs_order_     = cfg.stacked_obs_order;
    non_stacked_obs_order_ = cfg.non_stacked_obs_order;

    // Base lengths for each observation key
    obs_to_length_ = getObsToLengthMap();

    // Command vector length (affects "command" observation)
    if (cfg.cmd_vector_length < 0) {
        throw ModeConfigError("cmd_vector_length must be >= 0, but got " + std::to_string(cfg.cmd_vector_length));
    }
    cmd_vector_length_ = cfg.cmd_vector_length;
    obs_to_length_["command"] = static_cast<std::size_t>(cmd_vector_length_);

    // Number of stacked frames (stack size)
    if (cfg.stack_size < 1) {
        throw ModeConfigError("stack_size must be >= 1, but got " + std::to_string(cfg.stack_size));
    }
    stack_size_ = cfg.stack_size;

    // Policy file path (.onnx)
    if (cfg.policy_path.empty()) throw ModeConfigError("policy_path is required but missing");
    policy_path_ = cfg.policy_path;
    if (!isRegularFileExisting(policy_path_)) {
        throw std::runtime_error("policy_path not found or not a file: " + policy_path_);
    }
    {
        // Enforce .onnx extension (case-insensitive)
        std::string ext;
        auto dot = policy_path_.find_last_of('.');
        if (dot != std::string::npos) ext = policy_path_.substr(dot);
        std::string lower = ext;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
        if (lower != ".onnx") {
            throw ModeConfigError("policy_path must be a .onnx file, but got '" + ext + "'");
        }
    }

    // Policy type selection (MLP / LSTM)
    policy_type_ = cfg.policy_type;
    {
        std::string tl = toLower(policy_type_);
        if (!(tl == "mlp" || tl == "lstm")) {
            throw ModeConfigError("Unsupported policy_type: " + policy_type_);
        }
    }

    // ---- Observation scales ----
    //
    // For each observation key used in stacked and non-stacked orders:
    //   - if cfg.obs_scale has a value → normalize it
    //   - otherwise → default scale 1.0
    auto normalizeForKey = [&](const std::string& k) {
        auto itLen = obs_to_length_.find(k);
        if (itLen == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key: " + k);
        }
        const std::size_t len = itLen->second;
        auto itScale = cfg.obs_scale.find(k);
        if (itScale == cfg.obs_scale.end()) {
            // Default: no scaling (scale: 1.0)
            obs_scale_norm_[k] = std::vector<float>(len, 1.0f);
        } else {
            try {
                obs_scale_norm_[k] = normalizeScale(itScale->second, len);
            }
            catch (const ModeConfigError& e) {
                // Add context for this key
                throw ModeConfigError("obs_scale for '" + k + "': " + std::string(e.what()));
            }
        }
    };
    // Normalize scales for stacked observations
    for (const auto& k : stacked_obs_order_) {
        normalizeForKey(k);
    }
    // Normalize scales for non-stacked observations
    for (const auto& k : non_stacked_obs_order_) {
        normalizeForKey(k);
    }

    // ---- Action scale ----
    //
    // Length must match "last_action" (policy output dimension).
    last_action_len_ = obs_to_length_.at("last_action");
    if (cfg.action_scale.has_value()) {
        try {
            action_scale_ = normalizeScale(*cfg.action_scale, last_action_len_);
        } catch (const ModeConfigError& e) {
            throw ModeConfigError(std::string("action_scale: ") + e.what());
        }
    } else {
        action_scale_.assign(last_action_len_, 1.0f);
    }

    // Compute flattened state layout and total length
    computeStateLayout_();

    // ---- Policy construction & output shape check ----
    //
    // Use ONNX policies and verify that the output length
    // matches the length of "last_action".
    {
        const std::string tl = toLower(policy_type_);
        if (tl == "mlp") {
            policy_ = std::make_shared<PolicyAdapter<MLPPolicy>>(policy_path_);
        } else { // "lstm"
            policy_ = std::make_shared<PolicyAdapter<LSTMPolicy>>(policy_path_);
        }

        // Sanity check: inference output size vs last_action length
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

// Compute total state length and validate all observation keys
void Mode::computeStateLayout_() {
    // total = (sum(stacked obs lengths) * stack_size) + sum(non-stacked obs lengths)
    std::size_t total = 0;

    // Stacked part
    for (const auto& k : stacked_obs_order_) {
        auto it = obs_to_length_.find(k);
        if (it == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key in stacked_obs_order: " + k);
        }
        total += it->second;
    }
    total *= static_cast<std::size_t>(stack_size_);

    // Non-stacked part
    for (const auto& k : non_stacked_obs_order_) {
        auto it = obs_to_length_.find(k);
        if (it == obs_to_length_.end()) {
            throw ModeConfigError("unknown observation key in non_stacked_obs_order: " + k);
        }
        total += it->second;
    }
    state_len_ = total;
}

// Run policy inference on a flattened state vector
std::vector<float> Mode::inference(const std::vector<float>& state1d) {
    if (state1d.size() != state_len_) {
        std::ostringstream oss;
        oss << "state length mismatch: got " << state1d.size() << ", expected " << state_len_;
        throw std::runtime_error(oss.str());
    }
    std::vector<float> action = policy_->inference(state1d);

    return action;
}