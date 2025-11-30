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
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "robot.hpp"

namespace rl {

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

constexpr int MAX_MODE_ID = 16;

// ════════════════════════════════════════════════════════════════════════════
// Mode Configuration
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Complete specification for a reinforcement learning control mode
 * 
 * @details A mode defines how to construct the neural network input state
 *          from raw observations, commands, and action history:
 * 
 * **Temporal Stacking**:
 *   - stacked_obs: Concatenates last N frames (e.g., joint positions)
 *   - non_stacked_obs: Uses only current frame (e.g., height map)
 * 
 * **State Construction Order**:
 *   [stacked_obs₀...ₙ | non_stacked_obs | command | last_action]
 *   └─ temporally stacked ─┘ └──── current frame ────┘ └─ history ─┘
 * 
 * **Scaling**:
 *   - obs_scale: Scales observations to the training distribution
 *   - action_scale: Output denormalization to actuator range
 */
struct ModeDesc {
    int id = 0;  ///< Unique mode identifier

    // ───────────────────────────────────────────────────────────────────────
    // Observation Layout
    // ───────────────────────────────────────────────────────────────────────
    
    std::vector<std::string> stacked_obs_order;      ///< Keys with temporal stacking
    std::vector<std::string> non_stacked_obs_order;  ///< Single-frame keys

    // ───────────────────────────────────────────────────────────────────────
    // Scaling Parameters
    // ───────────────────────────────────────────────────────────────────────
    
    std::unordered_map<std::string, std::vector<float>> obs_scale;  ///< Input scaling per obs key
    std::vector<float> action_scale;                                ///< Output denormalization

    // ───────────────────────────────────────────────────────────────────────
    // Temporal Configuration
    // ───────────────────────────────────────────────────────────────────────
    
    int stack_size = 1;          ///< History depth for stacked observations
    int cmd_vector_length = 0;   ///< Command input dimension

    // ───────────────────────────────────────────────────────────────────────
    // Policy Backend
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Neural network inference callback
     * @param state Preprocessed state vector (scaled, stacked, concatenated)
     * @return Raw action output (before action_scale)
     */
    std::function<std::vector<float>(const std::vector<float>&)> inference;
};

// ════════════════════════════════════════════════════════════════════════════
// Multi-Mode RL Controller
// ════════════════════════════════════════════════════════════════════════════

/**
 * @class RL
 * @brief Multi-mode reinforcement learning policy manager with temporal stacking
 * 
 * @details Core responsibilities:
 *   1. **Mode Management**: Register and switch between control modes
 *   2. **State Construction**: Build neural network input from:
 *      - Temporally stacked observations (joint angles, velocities, etc.)
 *      - Single-frame observations (IMU data)
 *      - External commands (joystick, high-level planner)
 *      - Action history (for recurrent dependencies)
 *   3. **Scaling**: Scales observations to the training distribution
 *   4. **Inference**: Execute policy and denormalize actions to actuator range
 */
class RL {
public:
    /**
     * @brief Initializes RL controller with observation dimension registry
     */
    RL();

    // ───────────────────────────────────────────────────────────────────────
    // Mode Registration
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Registers or updates a control mode
     * @param mode Complete mode specification
     * @note If mode.id already exists, overwrites the previous configuration
     */
    void addMode(const ModeDesc& mode);

    // ───────────────────────────────────────────────────────────────────────
    // Mode Activation
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Activates a mode and reallocates internal buffers
     * @param mode_id ID of previously registered mode
     * @throws std::runtime_error if mode_id not found
     * 
     * @details Performs:
     *   - Buffer reallocation for state/frame storage
     *   - Pointer caching for performance
     *   - History reset (fills stack with zeros if no prior obs exists)
     */
    void setMode(int mode_id);

    // ───────────────────────────────────────────────────────────────────────
    // State Construction
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Constructs neural network input state from raw data
     * @param obs Robot observations (joint states, IMU, battery, etc.)
     * @param cmd External command vector (e.g., joystick input)
     * @param last_action_opt Previous action output (nullptr on first call)
     * @param check_mode If true, throws if no mode is active
     * @return Fully constructed and scaled state vector
     */
    std::vector<float> buildState(
        const std::unordered_map<std::string, std::vector<float>>& obs,
        const std::unordered_map<std::string, std::vector<float>>& cmd,
        const std::vector<float>* last_action_opt,
        bool check_mode = true
    );

    // ───────────────────────────────────────────────────────────────────────
    // Policy Execution
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Executes policy inference with action denormalization
     * @param state Preprocessed state from build_state()
     * @return Scaled action ready for actuator commands
     * 
     * @details Flow:
     *   1. Calls mode.inference(state) → raw network output
     *   2. Applies action_scale element-wise: action[i] *= scale[i]
     *   3. Caches both raw and scaled actions for next build_state()
     */
    std::vector<float> selectAction(const std::vector<float>& state);

private:
    // ───────────────────────────────────────────────────────────────────────
    // Observation Schema
    // ───────────────────────────────────────────────────────────────────────
    
    /// @brief Global observation dimension registry (name → length)
    std::unordered_map<std::string, std::size_t> obs_to_length_;

    // ───────────────────────────────────────────────────────────────────────
    // Mode Registry & Active State
    // ───────────────────────────────────────────────────────────────────────
    
    const ModeDesc* cur_mode_{nullptr};  ///< Active mode (nullptr if unset)
    std::vector<ModeDesc> modes_;        ///< Registered modes (indexed by id)

    // ───────────────────────────────────────────────────────────────────────
    // Frame Buffers (Temporal Stacking)
    // ───────────────────────────────────────────────────────────────────────

    std::vector<float> single_frame_;
    std::size_t single_frame_len_{0};  ///< Cached length of single_frame
    int stack_size_{1};                ///< Current mode's stack depth
    bool has_history_{false};          ///< Initialization flag

    // ───────────────────────────────────────────────────────────────────────
    // Full State Buffer
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Complete neural network input vector
     * @details Layout: [stacked_obs × stack_size | non_stacked_obs | cmd]
     */
    std::vector<float> state_;

    // ───────────────────────────────────────────────────────────────────────
    // Action Buffers
    // ───────────────────────────────────────────────────────────────────────
    
    std::vector<float> last_action_;    ///< Raw network output (unscaled)
    std::vector<float> scaled_action_;  ///< Actuator-ready commands (after action_scale)

    // ───────────────────────────────────────────────────────────────────────
    // History Cache (Stack Bootstrapping)
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Cached inputs from last build_state() call
     * @details Used to initialize temporal stack on mode switch or first call
     */
    std::unordered_map<std::string, std::vector<float>> last_obs_;
    std::unordered_map<std::string, std::vector<float>> last_cmd_;
    const std::vector<float>* last_last_action_opt_{nullptr};

    // ───────────────────────────────────────────────────────────────────────
    // Cached Pointers (Performance Optimization)
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Direct pointers into cur_mode to avoid repeated map lookups
     * @note Invalidated on set_mode(); repopulated during mode activation
     */
    const std::vector<float>* action_scale_{nullptr};
    const std::vector<std::string>* stacked_obs_order_{nullptr};
    const std::vector<std::string>* non_stacked_obs_order_{nullptr};
    const std::unordered_map<std::string, std::vector<float>>* obs_scale_{nullptr};
    std::function<std::vector<float>(const std::vector<float>&)> inference_;

    // ───────────────────────────────────────────────────────────────────────
    // Internal Utilities
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Validates that a mode is active
     * @throws std::runtime_error if cur_mode is nullptr
     */
    void ensureMode() const;

    /**
     * @brief Retrieves expected dimension for observation key
     * @param key Observation name (e.g., "dof_pos")
     * @return Dimension from obs_to_length registry
     * @throws std::runtime_error if key not found
     */
    std::size_t getObsLen(const std::string& key) const;

    /**
     * @brief Retrieves or generates scale vector for observation
     * @param key Observation name
     * @param len Expected dimension
     * @return Scale vector (from obs_scale or default [1.0, ...])
     */
    const std::vector<float>& getObsScale(const std::string& key, std::size_t len) const;

    // ───────────────────────────────────────────────────────────────────────
    // Scratch Memory
    // ───────────────────────────────────────────────────────────────────────
    
    /**
     * @brief Temporary storage for padded/broadcasted scale vectors
     * @note Mutable to allow modification in const methods (get_obs_scale)
     */
    mutable std::vector<float> padding_buffer_;
};

} // namespace rl