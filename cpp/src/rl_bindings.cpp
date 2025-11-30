#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "rl.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rl, m) {
    m.doc() = "RL bindings (pure C++ core; Python conversions only here)";

    py::class_<rl::RL>(m, "RL")
        .def(py::init<>())

        // Set ModeDesc from a python object
        .def("add_mode", [](rl::RL& self, py::object mode_obj) {
            rl::ModeDesc md;
            md.id                   = py::cast<int>(mode_obj.attr("id"));
            md.stacked_obs_order    = py::cast<std::vector<std::string>>(mode_obj.attr("stacked_obs_order"));
            md.non_stacked_obs_order= py::cast<std::vector<std::string>>(mode_obj.attr("non_stacked_obs_order"));
            md.action_scale         = py::cast<std::vector<float>>(mode_obj.attr("action_scale"));
            md.stack_size           = py::cast<int>(mode_obj.attr("stack_size"));
            md.cmd_vector_length    = py::cast<int>(mode_obj.attr("cmd_vector_length"));

            // obs_scale
            md.obs_scale.clear();
            py::dict obs_scale_py = mode_obj.attr("obs_scale").cast<py::dict>();
            for (auto item : obs_scale_py) {
                std::string k = py::cast<std::string>(item.first);
                md.obs_scale.emplace(k, py::cast<std::vector<float>>(item.second));
            }

            // inference: Python Mode.inference(state) 호출
            py::object infer_fun = mode_obj.attr("inference");
            md.inference = [infer_fun](const std::vector<float>& state)->std::vector<float> {
                py::gil_scoped_acquire gil;
                py::object out = infer_fun(state);
                return out.cast<std::vector<float>>();
            };

            self.addMode(md);
        }, py::arg("mode"))

        .def("set_mode", [](rl::RL& self, py::object mode_id) {
            if (!mode_id.is_none()) self.setMode(py::cast<int>(mode_id));
        }, py::arg("mode_id") = py::none())

        .def("build_state",
            [](rl::RL& self, py::dict obs, py::dict cmd, py::object last_action) {
                // 1) Set mode_id
                if (cmd.contains("mode_id") && !py::cast<py::object>(cmd["mode_id"]).is_none()) {
                    self.setMode(py::cast<int>(cmd["mode_id"]));
                }

                // 2) Convert obs [py:dict] -> [c++:unordered_map] 
                std::unordered_map<std::string, std::vector<float>> obs_map;
                for (auto item : obs) {
                    obs_map.emplace(py::cast<std::string>(item.first), py::cast<std::vector<float>>(item.second));
                }

                // 3) Convert cmd [py:dict] -> [c++:unordered_map] 
                std::unordered_map<std::string, std::vector<float>> cmd_map;
                if (cmd.contains("cmd_vector") && !py::cast<py::object>(cmd["cmd_vector"]).is_none()) {
                    cmd_map.emplace("cmd_vector", py::cast<std::vector<float>>(cmd["cmd_vector"]));
                }

                // 4) last_action (option)
                std::vector<float> la;
                const std::vector<float>* la_ptr = nullptr;
                if (!last_action.is_none()) {
                    la = last_action.cast<std::vector<float>>();
                    la_ptr = &la;
                }

                return self.buildState(obs_map, cmd_map, la_ptr, true);
            },
            py::arg("obs"), py::arg("cmd"), py::arg("last_action") = py::none()
        )

        .def("select_action", &rl::RL::selectAction, py::arg("state"));
}
