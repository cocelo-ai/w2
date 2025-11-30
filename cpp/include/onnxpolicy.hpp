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

#pragma once

#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace onnxpolicy {

// ════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Clips value to [-1, 1] range with NaN safety
 * @param x Input value
 * @return Clamped value (non-finite → 0.0)
 */
inline float clipUnit(float x) noexcept {
    x = std::isfinite(x) ? x : 0.0f;
    return std::fmin(std::fmax(x, -1.0f), 1.0f);
}

/**
 * @brief Extracts tensor shape from ONNX session
 * @param session Active ONNX session
 * @param idx Tensor index
 * @param input true for input tensors, false for outputs
 * @return Shape vector (dynamic dimensions may be -1)
 */
std::vector<int64_t> getShape(const Ort::Session& session, size_t idx, bool input=true);

/**
 * @brief Retrieves tensor name from ONNX session
 * @param session Active ONNX session
 * @param idx Tensor index
 * @param input true for input tensors, false for outputs
 * @return Tensor name as string
 */
std::string getName(const Ort::Session& session, size_t idx, bool input=true);

/**
 * @brief Replaces dynamic dimension (-1/0) with fallback value
 * @param x Dimension value
 * @param fallback Replacement for dynamic dimensions
 * @return Concrete dimension value
 */
int64_t valueOr(const int64_t x, int64_t fallback);


// ════════════════════════════════════════════════════════════════════════════
// MLP Policy (Feedforward)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class MLPPolicy
 * @brief Stateless feedforward policy for robot control
 * @details Single-pass inference: state → action (clipped to [-1, 1])
 */
class MLPPolicy {
public:
    /**
     * @brief Loads ONNX MLP model from file
     * @param weightPath Path to .onnx model file
     * @throws std::runtime_error if model loading fails
     */
    explicit MLPPolicy(const std::string& weightPath);

    /**
     * @brief Runs inference on observation state
     * @param state Input observation vector (size: state_dim)
     * @return Clipped action vector [-1, 1]
     */
    std::vector<float> inference(const std::vector<float>& state);

private:
    // ───────────────────────────────────────────────────────────────────────
    // ONNX Runtime Resources
    // ───────────────────────────────────────────────────────────────────────
    
    Ort::Env env_;                              ///< ONNX runtime environment
    Ort::SessionOptions so_;                    ///< Session configuration
    Ort::Session session_;                      ///< Model session

    Ort::MemoryInfo mem_info_{nullptr};         ///< CPU memory allocator
    Ort::RunOptions run_opts_{};                ///< Inference options

    // ───────────────────────────────────────────────────────────────────────
    // Model Metadata
    // ───────────────────────────────────────────────────────────────────────
    
    std::string input_name_;                    ///< Input tensor name
    std::string output_name_;                   ///< Output tensor name
    const char* input_name_c_{nullptr};         ///< C-string cache (API compat)
    const char* output_name_c_{nullptr};        ///< C-string cache (API compat)

    bool batch_required_{false};                ///< Expects batch dimension
    int64_t state_dim_{-1};                     ///< Input state dimension

    std::vector<int64_t> input_dims_template_;  ///< Input shape template
};

// ════════════════════════════════════════════════════════════════════════════
// LSTM Policy (Recurrent)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class LSTMPolicy
 * @brief Stateful recurrent policy with hidden state management
 * 
 * @details Supports flexible input/output name matching with fallbacks:
 * 
 * **State Input** (2D expected):
 *   - Primary: "state", "obs", "observation", "observations"
 *   - Fallback: "input", "input_0", "input0"
 * 
 * **Hidden State Inputs** (3D expected):
 *   - h_in: "h_in", "hidden_in", "h0", "h", "input_1", "input1"
 *   - c_in: "c_in", "cell_in", "c0", "c", "input_2", "input2"
 * 
 * **Hidden State Outputs** (3D expected):
 *   - h_out: "h_out", "hn", "hidden", "h", "output_1", "output1"
 *   - c_out: "c_out", "cn", "cell", "c", "output_2", "output2"
 * 
 * @note Additional inputs with dynamic dimensions are bound to zero tensors
 */
class LSTMPolicy {
public:
    /**
     * @brief Loads ONNX LSTM model and initializes hidden states
     * @param weightPath Path to .onnx model file
     * @throws std::runtime_error if model loading or validation fails
     */
    explicit LSTMPolicy(const std::string& weightPath);

    /**
     * @brief Runs single-timestep inference with state update
     * @param state Input observation vector
     * @return Clipped action vector [-1, 1]
     * @note Automatically updates internal h/c states for next call
     */
    std::vector<float> inference(const std::vector<float>& state);

private:
    /**
     * @brief Updates internal hidden states from model outputs
     * @param outs Output tensors from ONNX inference
     */
    void updateHiddenFromOutputs(const std::vector<Ort::Value>& outs);

    Ort::Env env_;
    Ort::SessionOptions so_;
    Ort::Session session_;

    Ort::MemoryInfo mem_info_{nullptr};
    Ort::RunOptions run_opts_{};

    // ───────────────────────────────────────────────────────────────────────
    // Input/Output Name Management
    // ───────────────────────────────────────────────────────────────────────
    
    std::unordered_map<std::string, size_t> input_index_by_name_;  ///< Name → index mapping
    std::vector<std::string> input_names_;                         ///< Ordered input names
    std::vector<const char*> input_cstrs_;                         ///< C-string cache

    std::vector<std::string> output_names_;                        ///< Ordered output names
    std::vector<const char*> output_cstrs_;                        ///< C-string cache
    
    // ───────────────────────────────────────────────────────────────────────
    // Tensor Index Cache
    // ───────────────────────────────────────────────────────────────────────
    size_t state_idx_{static_cast<size_t>(-1)};  ///< State input index
    size_t h_idx_{static_cast<size_t>(-1)};      ///< Hidden state input index
    size_t c_idx_{static_cast<size_t>(-1)};      ///< Cell state input index

    // ───────────────────────────────────────────────────────────────────────
    // Model Dimensions
    // ───────────────────────────────────────────────────────────────────────
    
    int64_t h_dim_{1};         ///< Hidden state dimension
    int64_t c_dim_{1};         ///< Cell state dimension
    int64_t batch_size_{1};    ///< Batch size (typically 1)
    int64_t seq_len_{1};       ///< Sequence length (typically 1)
    int64_t state_dim_{-1};    ///< Observation dimension

    // ───────────────────────────────────────────────────────────────────────
    // Recurrent State Buffers
    // ───────────────────────────────────────────────────────────────────────
    
    std::string state_name_{"state"};           ///< Resolved state input name
    std::vector<float> policy_h_in_;            ///< Hidden state (persistent)
    std::vector<float> policy_c_in_;            ///< Cell state (persistent)

    std::vector<int64_t> hc_dims_;              ///< Hidden state shape template
    std::vector<int64_t> cc_dims_;              ///< Cell state shape template

    // ───────────────────────────────────────────────────────────────────────
    // Extra Input Handling (Zero Padding)
    // ───────────────────────────────────────────────────────────────────────
    
    std::vector<std::vector<int64_t>> extra_input_dims_;  ///< Materialized dims for extra inputs
    std::vector<std::vector<float>> zero_holders_;        ///< Zero-filled tensors for binding
};

} // namespace onnxpolicy