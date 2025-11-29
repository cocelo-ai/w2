/**                                                                                       
                                                                     ▒██▓▒▓████▓░    
                                                                     ▓███████████░   
                                                                     ░▓██████████    
                                                                          ░██░ ███░  
                                                                          ░██   ▒█▒  
  ██████▒░█▓       ▓██░   ███   ███ ▓█░ ███  ░█▒ ▒██████▒  ███████       ░█░   ▒█▓   
  ██     ░█▓      ▒█░▒█░  ████ ▓███ ▓█░ █▓██ ░█▒░█▓       ▒█░    ██  ░████    ▒█     
  ██████░░█▓      ██▓▓██  ██░█ █▒██ ▓█░ █▓ ██▒█▒░█▓  ▒▓██ ▓█░    ██  ▓████  ████░    
  ██     ░██▒▒▒▒ ██    █▓ ██ ▒█▓ ██ ▓█░ █▓  ▓██▒ ▓██▒░▓██  ██▓░▒██░   ▒█▒   ████▒    
                                                                            ░██░     
███████████████████████████████████████████████████████████████████████████████████      
*/

#pragma once  // mode.hpp

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


// ════════════════════════════════════════════════════════════════════════════
// Exceptions
// ════════════════════════════════════════════════════════════════════════════

/// @brief Thrown when mode configuration is invalid
class ModeConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ════════════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════════════

/// @brief Returns observation name → dimension mapping
std::unordered_map<std::string, std::size_t> get_obs_to_length_map();

/// @brief Scale specification: scalar or per-dimension vector
using ScaleSpec = std::variant<float, std::vector<float>>;

// ════════════════════════════════════════════════════════════════════════════
// Policy Interface
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Abstract policy interface for inference
 * @note Pure virtual; implemented by ML backends (ONNX only)
 */
class IPolicy {
public:
    virtual ~IPolicy() = default;

    /**
     * @brief Runs inference on input state
     * @param state Preprocessed observation vector
     * @return Action vector
     */
    virtual std::vector<float> inference(const std::vector<float>& state) = 0;
};

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Mode configuration specification
 * @details Defines observation layout, scaling, and policy parameters
 */
struct ModeConfig {
    int id = 0;                                             ///< Unique mode identifier
    std::vector<std::string> stacked_obs_order;             ///< Order of stacked observations
    std::vector<std::string> non_stacked_obs_order;         ///< Order of non-stacked observations
    std::unordered_map<std::string, ScaleSpec> obs_scale;   ///< observation scaling map
    std::optional<ScaleSpec> action_scale;                  ///< Action output scaling
    int stack_size = 1;                                     ///< Stack size
    std::string policy_path;                                ///< Model file path
    std::string policy_type = "MLP";                        ///< "MLP" or "LSTM"
    int cmd_vector_length = 0;                              ///< Command input dimension
};

// ════════════════════════════════════════════════════════════════════════════
// Mode Runtime
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class Mode
 * @brief Robot control mode with policy inference and observation processing
 * @details Handles observation stacking, scaling, and policy execution
 */
class Mode {
public:
    using ObsLengthMap = std::unordered_map<std::string, std::size_t>;

    /**
     * @brief Constructs mode from configuration
     * @param cfg Mode configuration
     * @throws ModeConfigError if configuration is invalid
     */
    explicit Mode(const ModeConfig& cfg);

    // ───────────────────────────────────────────────────────────────────────
    // Inference
    // ───────────────────────────────────────────────────────────────────────

    /**
     * @brief Runs policy inference on flattened state
     * @param state1d Preprocessed 1D observation vector
     * @return Scaled action vector
     */
    std::vector<float> inference(const std::vector<float>& state1d);

    // ───────────────────────────────────────────────────────────────────────
    // Metadata Accessors
    // ───────────────────────────────────────────────────────────────────────

    const ObsLengthMap& obs_lengths() const { return obs_to_length_; }
    int id() const { return id_; }
    int stack_size() const { return stack_size_; }
    int state_length() const { return static_cast<int>(state_len_); }
    int action_length() const { return static_cast<int>(last_action_len_); }
    int cmd_vector_length() const { return cmd_vector_length_; }

    const std::string& policy_type() const { return policy_type_; }
    const std::string& policy_path() const { return policy_path_; }

    const std::vector<std::string>& stacked_obs_order() const { return stacked_obs_order_; }
    const std::vector<std::string>& non_stacked_obs_order() const { return non_stacked_obs_order_; }

    // ───────────────────────────────────────────────────────────────────────
    // Scaling Parameters
    // ───────────────────────────────────────────────────────────────────────

    const std::vector<float>& action_scale() const { return action_scale_; }
    const std::unordered_map<std::string, std::vector<float>>& obs_scale() const { return obs_scale_norm_; }

private:
    // ───────────────────────────────────────────────────────────────────────
    // Internal Utilities
    // ───────────────────────────────────────────────────────────────────────

     /**
     * @brief Normalizes scale spec to vector form
     * @param spec Scalar or vector scale specification
     * @param length Target dimension
     * @return Normalized scale vector
     */
    static std::vector<float> normalize_scale(const ScaleSpec& spec, std::size_t length);

    /// @brief Validates observation layout and computes state dimensions
    void compute_state_layout_();

    /// @brief Converts string to lowercase
    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return s;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Configuration State
    // ───────────────────────────────────────────────────────────────────────

    ObsLengthMap obs_to_length_;                                        ///< Observation dimensions
    int id_{0};                                                         ///< Mode ID
    std::vector<std::string> stacked_obs_order_;                        ///< Stacked obs names
    std::vector<std::string> non_stacked_obs_order_;                    ///< Non-stacked obs names
    std::unordered_map<std::string, std::vector<float>> obs_scale_norm_;///< Normalized obs scales
    std::vector<float> action_scale_;                                   ///< Action output scales
    int stack_size_{1};                                                 ///< Stack size
    std::string policy_path_;                                           ///< Model file path
    std::string policy_type_;                                           ///< Policy architecture ("MLP" or "LSTM")
    int cmd_vector_length_{0};                                          ///< Command dimension

    // ───────────────────────────────────────────────────────────────────────
    // Derived Metadata
    // ───────────────────────────────────────────────────────────────────────

    std::size_t state_len_{0};
    std::size_t last_action_len_{0};

    // ───────────────────────────────────────────────────────────────────────
    // Policy Backend
    // ───────────────────────────────────────────────────────────────────────

    std::shared_ptr<IPolicy> policy_;
};
