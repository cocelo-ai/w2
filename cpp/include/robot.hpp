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

    // ------- Safety check -------
    void check_safety();
    void check_obs(const std::unordered_map<std::string, std::vector<float>>& obs);

    // ------- Observation (returns copy; internal buffers reused) -------
    std::unordered_map<std::string, std::vector<float>> get_obs();

    // ------- Action -------
    void do_action(const std::vector<float>& action, bool torque_ctrl=false);

    // ------- Cache values -------
    void cache_motor_values(const std::string& str, std::vector<std::size_t>& ppos_in_mcu_str, std::vector<std::size_t>& vpos_in_mcu_str, std::vector<std::size_t>& tpos_in_mcu_str);
    void cache_imu_values(const std::string& str, std::size_t& gx_in_mcu_str, std::size_t& gy_in_mcu_str, std::size_t& gz_in_mcu_str, std::size_t& pgx_in_mcu_str, std::size_t& pgy_in_mcu_str);
    void cache_status_values(const std::string& status_str, std::vector<std::size_t>& patt_in_status_str,std::size_t& emergency_in_status_str);

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
    void wait(std::int32_t timeout_ms = 30000);

    // Status check (native string parsing)
    std::pair<bool,bool> check_status(const std::string& status_str, const std::vector<uint8_t>& mids);

    // MCU data sanity (native string)
    bool check_mcu(const std::string& mcu_str, const std::vector<uint8_t>& mids);

    // Parse obs (in-place into pre-sized vectors, native string parsing)
    std::unordered_map<std::string, std::vector<float>>& parse_obs(const std::string& mcu);

    const size_t last_action_len;

    std::vector<float> kp;
    std::vector<float> kd;
    bool gains_set;

    // config / ids
    std::vector<uint8_t> motor_ids;

    // conn state
    int cli_disconn_timeout_ms;
    int cli_disconn_duration_ms;
    int cli_missed_req;

    // state (pre-sized & reused)
    std::unordered_map<std::string, std::vector<float>> obs;
    std::unordered_map<std::string, float> pos_offset;
    std::unordered_map<std::string, float> rel_max_pos, rel_min_pos;
    std::vector<std::string> joint_names;

    // 효율적 파싱을 위한 위치 캐싱
    std::vector<std::size_t> p_in_mcu_str; // p: value start
    std::vector<std::size_t> v_in_mcu_str; // v: value start
    std::vector<std::size_t> t_in_mcu_str; // t: value start

    std::size_t gx_in_mcu_str; 
    std::size_t gy_in_mcu_str; 
    std::size_t gz_in_mcu_str; 
    std::size_t pgx_in_mcu_str; 
    std::size_t pgy_in_mcu_str; 
    std::size_t pgz_in_mcu_str; 

    std::vector<std::size_t> patt_in_status_str;  
    std::size_t emergency_in_status_str; 

    // Native FxCli handle
    FxCli cli;
};

} // namespace robot
