// joystick_bindings.cpp
//
// Python bindings for the simplified joystick reader. This file
// exposes the `joystick` module with a `Joystick` class and
// associated exceptions via pybind11. The binding mirrors the
// behaviour of the original Python implementation but delegates
// almost all heavy lifting to the underlying C++ code.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "joystick.hpp"

namespace py = pybind11;
using namespace joystick;

PYBIND11_MODULE(joystick, m) {
    m.doc() = "Simplified C++ implementation of a joystick controller";

    // Expose custom exceptions. They will translate C++ exceptions
    // into Python exceptions with the same name.
    py::register_exception<JoystickEstopError>(m, "JoystickEstopError");
    py::register_exception<JoystickSleepError>(m, "JoystickSleepError");

    // Bind JoystickOutput. While users will generally not need to
    // construct this class directly (it is produced by getCmd()),
    // providing bindings allows for Python introspection of the
    // structure returned from C++.
    py::class_<JoystickOutput>(m, "JoystickOutput")
        .def(py::init<>())
        .def_readwrite("cmd_vector", &JoystickOutput::cmd_vector,
                       "Command vector (6 floats)")
        .def_readwrite("mode_id", &JoystickOutput::mode_id,
                       "Mode ID (optional int)")
        .def_readwrite("estop", &JoystickOutput::estop,
                       "Emergency stop flag")
        .def_readwrite("wake", &JoystickOutput::wake,
                       "Wake flag (hold left trigger 2s)")
        .def_readwrite("sleep", &JoystickOutput::sleep,
                       "Sleep flag (hold right trigger 2s)")
        .def("__repr__", [](const JoystickOutput& out) {
            std::string mode_str = out.mode_id.has_value()
                ? std::to_string(out.mode_id.value())
                : "None";
            std::string cmd_str = "[";
            for (size_t i = 0; i < out.cmd_vector.size(); ++i) {
                cmd_str += std::to_string(out.cmd_vector[i]);
                if (i + 1 < out.cmd_vector.size()) cmd_str += ", ";
            }
            cmd_str += "]";
            return "JoystickOutput(cmd_vector=" + cmd_str +
                   ", mode_id=" + mode_str +
                   ", estop=" + (out.estop ? "True" : "False") +
                   ", wake="  + (out.wake  ? "True" : "False") +
                   ", sleep=" + (out.sleep ? "True" : "False") + ")";
        });

    // Bind Joystick class. Provide constructors, getCmd and
    // isConnected methods. The getCmd method returns a Python
    // dictionary for compatibility with existing Python code.
    py::class_<Joystick>(m, "Joystick")
        .def(py::init<const std::vector<double>&, double, const std::map<std::string,int>&>(),
            py::arg("max_cmd") = std::vector<double>(),
            py::arg("smoothness") = 50.0,
            py::arg("mapping") = std::map<std::string,int>(),
            R"pbdoc(
                Initialize a joystick controller.

                Parameters
                ----------
                max_cmd : list of float, optional
                    Maximum command values for each axis (up to 6 elements).
                    Missing elements default to 1.0.
                smoothness : float, optional
                    A value between 0 and 100. Larger values apply stronger
                    low-pass filtering (smoother but more laggy).
                mapping : dict, optional
                    Custom mapping of human-friendly names ("LEFT_X", etc.)
                    to command indices (0-5). Required keys are LEFT_X,
                    LEFT_Y, RIGHT_X, RIGHT_Y, LEFT_BTN, and RIGHT_BTN.
            )pbdoc")

        .def("get_cmd", [](Joystick& js) {
            // Obtain the current command state from C++. Convert
            // JoystickOutput into a Python dict for compatibility
            // with the original Python interface.
            JoystickOutput out = js.getCmd();
            py::dict result;
            result["cmd_vector"] = out.cmd_vector;
            if (out.mode_id.has_value()) {
                result["mode_id"] = out.mode_id.value();
            } else {
                result["mode_id"] = py::none();
            }
            result["estop"] = out.estop;
            result["wake"] = out.wake;
            result["sleep"] = out.sleep;
            return result;
        },
        R"pbdoc(
                Get the current joystick command state.

                Returns
                -------
                dict
                    A dictionary containing:
                      - cmd_vector: list of 6 floats (command values)
                      - mode_id: int or None (selected mode)
                      - estop: bool (emergency stop flag)
                      - wake: bool (wake flag; hold left trigger 2 seconds)
                      - sleep: bool (sleep flag; hold right trigger 2 seconds)

                Raises
                ------
                JoystickEstopError
                    If both triggers are pressed concurrently.
                JoystickSleepError
                    If the right trigger is held for more than 2 seconds.
                )pbdoc")
        .def("is_connected", &Joystick::isConnected,
             "Return True if the joystick device is connected.")
        .def("__repr__", [](const Joystick& js) {
            return std::string("<Joystick(connected=") +
                   (js.isConnected() ? "True" : "False") + ")>";
        });

    // Provide a simple version number to aid debugging.
    m.attr("__version__") = "1.0.0";
}