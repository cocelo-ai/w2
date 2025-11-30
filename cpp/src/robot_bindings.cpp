#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robot.hpp"

namespace py = pybind11;
using namespace robot;


PYBIND11_MODULE(robot, m) {
    m.doc() = "C++ port of robot.py";

    // Exceptions
    py::register_exception<RobotEStopError>(m, "RobotEStopError");
    py::register_exception<RobotSetGainsError>(m, "RobotSetGainsError");
    py::register_exception<RobotInitError>(m, "RobotInitError");

    py::class_<Robot>(m, "Robot")
        .def(py::init<>())

        .def("set_gains", &Robot::setGains, py::arg("kp"), py::arg("kd"),
             "Set PD gains")

        // Check 1D action
        .def("do_action",
             [](Robot& self, py::object action, bool torque_ctrl) {
                 if (py::isinstance<py::sequence>(action)) {
                     py::sequence seq = action;
                     if (py::len(seq) > 0 && py::isinstance<py::sequence>(seq[0])) {
                         self.estop("action must be a 1D list");
                     }
                 }
                 std::vector<float> a = action.cast<std::vector<float>>();
                 self.doAction(a, torque_ctrl, true);
             },
             py::arg("action"), py::arg("torque_ctrl") = false)

        .def("get_obs", &Robot::getObs)
        .def("estop",[](Robot& self, const std::string& msg) {
            self.estop(msg, /*is_physical_estop=*/false);
            },
    py::arg("msg") = std::string()
)
        .def("get_gains", &Robot::getGains)
        .def_property_readonly("gains_set", &Robot::gainsReady);
}