#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fx_cli/fx_client.h> 

namespace robot {

// -------- Exceptions --------
struct RobotEStopError : public std::runtime_error { using std::runtime_error::runtime_error; };
struct RobotInitError : public std::runtime_error { using std::runtime_error::runtime_error; };
struct RobotSetGainsError : public std::runtime_error { using std::runtime_error::runtime_error; };

class Robot {
public:
    Robot();

    // ------- Set gains -------
    void set_gains(const std::vector<float>& kp, const std::vector<float>& kd);

    // ------- Safety -------
    void check_safety(float POS_SAFETY_MARGIN_RAD,
                      float POS_MARGIN_NEAR_LIMIT_RAD,
                      float VEL_THRESHOLD_NEAR_LIMIT_RAD_S,
                      float VEL_HARD_LIMIT_RAD_S);

    // ------- Observation (returns copy; internal buffers reused) -------
    std::unordered_map<std::string, std::vector<float>> get_obs();

    // ------- Action -------
    void do_action(const std::vector<float>& action, bool torque_ctrl, bool safe);

    // ------- E-Stop API  -------
    [[noreturn]] void estop(const std::string& msg = std::string());

    // ======= Read-only getters for Python =======
    inline std::pair<std::vector<float>, std::vector<float>> get_gains() const {
        return {kp, kd};
    }
    inline bool gains_ready() const {
        return gains_set;
    }
    inline std::size_t action_len() const {
        return last_action_len;
    }


private:
    // HW 준비 대기
    void wait(std::int32_t timeout_ms = 10000);
    std::unordered_map<std::string, std::vector<float>>& parse_obs(const FxCliMap& mcu_obs);
    std::pair<bool,bool> check_status(const FxCliMap& status, const std::vector<uint8_t>& mids);

    const size_t last_action_len;
    std::vector<float> last_action;
    bool last_torque_ctrl;

    std::vector<float> kp;
    std::vector<float> kd;
    bool gains_set;

    // config / ids
    std::vector<uint8_t> motor_ids;

    // conn state
    int cli_disconn_timeout_ms;
    int cli_disconn_duration_ms;
    int cli_missed_req;

    // Obs container & properties
    std::unordered_map<std::string, std::vector<float>> obs;
    std::unordered_map<std::string, float> pos_offset;
    std::unordered_map<std::string, float> rel_max_pos, rel_min_pos;
    std::vector<std::string> joint_names;

    // Native FxCli handle
    FxCli cli;
};

} // namespace robot
