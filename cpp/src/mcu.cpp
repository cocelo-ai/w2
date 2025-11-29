#include <utility>  

#include "mcu.hpp"

namespace robot {

McuClient::McuClient(const std::string& ip,
                     int port,
                     std::vector<uint8_t> ids,
                     bool has_imu,
                     bool has_bat,
                     bool has_estop)
    : cli(ip, port)
    , motor_ids(std::move(ids))
    , num_motors(motor_ids.size())
    , has_imu(has_imu)
    , has_bat(has_bat)
    , has_estop(has_estop)
    , req_miss_count(0)
{}

} // namespace robot
