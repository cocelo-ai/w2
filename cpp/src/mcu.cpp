#include "mcu.hpp"

namespace mcu {

McuClient::McuClient(const std::string& ip,
                     int port,
                     std::vector<uint8_t> ids,
                     bool has_imu,
                     bool has_bat,
                     bool has_estop)
    : cli_(ip, port)
    , motor_ids_(std::move(ids))
    , num_motors_(motor_ids_.size())
    , has_imu_(has_imu)
    , has_bat_(has_bat)
    , has_estop_(has_estop)
    , req_miss_count_(0)
    , max_req_miss_count_(5)
    , status_disconn_count_(0)
    , max_status_disconn_count_(5)
    , mcu_obs_(num_motors_)
    , mcu_status_(num_motors_)      
{}

// Fetch the latest observation from the MCU and update obs_
const McuObs& McuClient::fetchObs() {
    auto& p  = mcu_obs_.p;              ///> Motor positions            (size: num_motors)
    auto& v  = mcu_obs_.v;              ///> Motor velocities           (size: num_motors)
    auto& t  = mcu_obs_.t;              ///> Motor torques              (size: num_motors)
    auto& g  = mcu_obs_.g;              ///> IMU angular velocity       (size: 3)
    auto& pg = mcu_obs_.pg;             ///> Projected gravity from IMU (size: 3)

    FxCliMap mcu_data = cli_.req(motor_ids_);

    // Check for ACK from MCU
    if (auto it_ack = mcu_data.find("ACK"); it_ack != mcu_data.end()) {
        const auto& ack = it_ack->second;
        auto it = ack.find("REQ");
        if (!(it != ack.end() && it->second == "true")) {
            ++req_miss_count_;
            return mcu_obs_;
        }
    } else {
        ++req_miss_count_;
        return mcu_obs_;
    }

    // Helper to parse individual motor data
    auto getMotor = [&](std::size_t index)
        -> const std::unordered_map<std::string, std::string>*
    {
        std::string key = "M" + std::to_string(index); // e.g., "M1", "M2", ...
        auto it = mcu_data.find(key);
        if (it == mcu_data.end()) {
            return nullptr;
        }
        return &it->second;
    };

    // Parse motor data (position, velocity, torque)
    for (std::size_t i = 0; i < motor_ids_.size(); ++i) {
        auto mid = motor_ids_[i];

        const auto* m = getMotor(static_cast<std::size_t>(mid));
        if (!m) {
            ++req_miss_count_;
            return mcu_obs_;
        }

        auto it_p = m->find("p");
        auto it_v = m->find("v");
        auto it_t = m->find("t");  // Torque may not exist depending on protocol

        if (it_p == m->end() || it_v == m->end() ||
            it_p->second == "N" || it_v->second == "N") {
            ++req_miss_count_;
            return mcu_obs_;
        }

        p[i] = std::stof(it_p->second);
        v[i] = std::stof(it_v->second);

        if (it_t != m->end() && it_t->second != "N") {
            t[i] = std::stof(it_t->second);
        } else {
            t[i] = 0.0f;
        }
    }

    //  Parse IMU data if available
    if (has_imu_) {
        auto it_imu = mcu_data.find("IMU");
        if (it_imu == mcu_data.end()) {
            ++req_miss_count_;
            return mcu_obs_;
        }

        const auto& imu = it_imu->second;
        auto getImu = [&](const char* key, float& out) -> bool {
            auto it = imu.find(key);
            if (it == imu.end() || it->second == "N")
                return false;
            out = std::stof(it->second);
            return true;
        };

        float gx, gy, gz, pgx, pgy, pgz;
        if (!getImu("gx", gx)   || !getImu("gy", gy)  || !getImu("gz", gz)   ||
            !getImu("pgx", pgx) || !getImu("pgy", pgy)|| !getImu("pgz", pgz))
        {
            ++req_miss_count_;
            return mcu_obs_;
        }

        g[0]  = gx;
        g[1]  = gy;
        g[2]  = gz;
        pg[0] = pgx;
        pg[1] = pgy;
        pg[2] = pgz;
    }

    req_miss_count_ = 0;
    return mcu_obs_;
}

// Fetch MCU status packet and update status information
const McuStatus& McuClient::fetchStatus() {
    FxCliMap data = cli_.status();   // Request a status packet from the MCU

    // Initialize status with default values
    mcu_status_.disconnected   = false;
    mcu_status_.emergency      = false;
    mcu_status_.battery_low    = false;
    mcu_status_.battery_voltage.clear();
    mcu_status_.battery_soc.clear();
    mcu_status_.motor_pattern.assign(num_motors_, "N");
    mcu_status_.motor_err.assign(num_motors_, "None");

    // Check emergency stop button
    if (has_estop_) {
        auto it_emg = data.find("EMERGENCY");
        if (it_emg != data.end()) {
            const auto &emg = it_emg->second;
            auto it_val = emg.find("value");
            if (it_val != emg.end() && it_val->second == "ON") {
                // Physical E-stop button is pressed
                mcu_status_.emergency = true;
                status_disconn_count_ = 0;
                return mcu_status_;
            }
        }
    }

    // Check ACK / STATUS
    auto it_ack = data.find("ACK");
    if (it_ack == data.end()) {
        // ACK block missing, treat as invalid packet
        ++status_disconn_count_;
        if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
        return mcu_status_;
    }

    const auto &ack = it_ack->second;
    auto it_status = ack.find("STATUS");
    if (it_status == ack.end() || it_status->second != "true") {
        ++status_disconn_count_;
        if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
        return mcu_status_;
    }

    // Parse motor pattern & err
    for (std::size_t i = 0; i < motor_ids_.size(); ++i) {
        const auto id = motor_ids_[i]; 
        std::string key = "M" + std::to_string(id);

        auto it_m = data.find(key);
        if (it_m == data.end()) {
            mcu_status_.motor_pattern[i] = "<missing>";
            mcu_status_.motor_err[i]     = "<missing>";
            ++status_disconn_count_;
            if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
            return mcu_status_;     
        }

        const auto &m = it_m->second;

        auto it_pattern = m.find("pattern");
        if (it_pattern != m.end()) {
            mcu_status_.motor_pattern[i] = it_pattern->second;
        } else {
            mcu_status_.motor_pattern[i] = "<missing>";
            ++status_disconn_count_;
            if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
            return mcu_status_;     
        }

        auto it_err = m.find("err");
        if (it_err != m.end()) {
            mcu_status_.motor_err[i] = it_err->second;
        } else {
            mcu_status_.motor_err[i] = "<missing>";
            ++status_disconn_count_;
            if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
            return mcu_status_;     
        }

        if (mcu_status_.motor_pattern[i] != "2" ||
            mcu_status_.motor_err[i]     != "None") {
            ++status_disconn_count_;
            if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
            return mcu_status_;     

        }
    }

    // Parse battery information
    if (has_bat_) {
        auto it_batt = data.find("BATT");
        if (it_batt == data.end()) {
            ++status_disconn_count_;
            if(status_disconn_count_ > max_status_disconn_count_){
                mcu_status_.disconnected = true;
                return mcu_status_;
            }   
        } else {
            const auto &batt = it_batt->second;

            auto it_v   = batt.find("V");
            auto it_soc = batt.find("SOC");

            if (it_v != batt.end()) {
                mcu_status_.battery_voltage = it_v->second;
            } else {
                ++status_disconn_count_;
                if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
                return mcu_status_;     
            }

            if (it_soc != batt.end()) {
                mcu_status_.battery_soc = it_soc->second;

                try {
                    float soc_val = std::stof(mcu_status_.battery_soc);
    
                    if (soc_val < 2.0f) {
                        mcu_status_.battery_low = true;
                    }
                } catch (const std::exception &) {
                    ++status_disconn_count_;
                    if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
                    return mcu_status_;     
                }
            } else {
                ++status_disconn_count_;
                if(status_disconn_count_ > max_status_disconn_count_) mcu_status_.disconnected = true; 
                return mcu_status_;  
            }
        }
    }

    status_disconn_count_ = 0;
    if(req_miss_count_ > max_req_miss_count_) mcu_status_.disconnected = true;
    return mcu_status_;
}

} // namespace mcu
