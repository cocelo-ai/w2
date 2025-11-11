// mode_bindings.cpp (float-only)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mode.hpp"

namespace py = pybind11;

// ---- Python value -> ScaleSpec (float ONLY) ----
static ScaleSpec py_to_scale_spec(const py::handle& h) {
    if (h.is_none()) {
        throw ModeConfigError("internal error: py_to_scale_spec called with None");
    }
    if (py::isinstance<py::bool_>(h)) {
        throw ModeConfigError("scale must be numeric, not bool");
    }
    if (py::isinstance<py::int_>(h) || py::isinstance<py::float_>(h)) {
        return py::cast<float>(h); // scalar(float)
    }
    if (py::isinstance<py::sequence>(h)) {
        std::vector<float> v;
        py::sequence seq = py::reinterpret_borrow<py::sequence>(h);
        v.reserve(seq.size());
        std::size_t idx = 0;
        for (py::handle item : seq) {
            if (py::isinstance<py::bool_>(item)) {
                throw ModeConfigError("scale sequence must not contain bool at index " + std::to_string(idx));
            }
            try {
                v.push_back(py::cast<float>(item));
            } catch (const py::cast_error&) {
                const std::string r = py::str(py::repr(item));
                throw ModeConfigError("scale sequence must contain only numbers; bad element at index "
                                      + std::to_string(idx) + ": " + r);
            }
            ++idx;
        }
        return v; // vector<float>
    }
    const std::string r = py::str(py::repr(h));
    throw ModeConfigError("scale must be a number or a sequence, got: " + r);
}

// ---- dict(mode_cfg) -> ModeConfig ----
static ModeConfig modeconfig_from_dict(const py::dict& d) {
    ModeConfig cfg;

    auto get_opt = [&](const char* key)->py::object {
        return d.contains(key) ? py::object(d[key]) : py::none();
    };

    if (!get_opt("id").is_none()) cfg.id = py::cast<int>(get_opt("id"));
    if (!get_opt("stacked_obs_order").is_none())
        cfg.stacked_obs_order = py::cast<std::vector<std::string>>(get_opt("stacked_obs_order"));
    if (!get_opt("non_stacked_obs_order").is_none())
        cfg.non_stacked_obs_order = py::cast<std::vector<std::string>>(get_opt("non_stacked_obs_order"));
    if (!get_opt("stack_size").is_none()) cfg.stack_size = py::cast<int>(get_opt("stack_size"));
    if (!get_opt("policy_path").is_none()) cfg.policy_path = py::cast<std::string>(get_opt("policy_path"));
    if (!get_opt("policy_type").is_none()) cfg.policy_type = py::cast<std::string>(get_opt("policy_type"));
    if (!get_opt("cmd_vector_length").is_none()) cfg.cmd_vector_length = py::cast<int>(get_opt("cmd_vector_length"));

    if (!get_opt("obs_scale").is_none()) {
        py::dict os = py::cast<py::dict>(get_opt("obs_scale"));
        for (auto item : os) {
            std::string key = py::cast<std::string>(item.first);
            cfg.obs_scale[key] = py_to_scale_spec(item.second);
        }
    }
    if (!get_opt("cmd_scale").is_none()) {
        // Treat cmd_scale as an alias for obs_scale["command"]
        cfg.obs_scale["command"] = py_to_scale_spec(get_opt("cmd_scale"));
    }
    if (!get_opt("action_scale").is_none())
        cfg.action_scale = py_to_scale_spec(get_opt("action_scale"));

    return cfg;
}

// ---- pybind module ----
PYBIND11_MODULE(mode, m) {
    m.doc() = "Mode (C++ core) + pybind11 bindings (float-only)";

    py::register_exception<ModeConfigError>(m, "ModeConfigError");

    // ModeConfig
    py::class_<ModeConfig>(m, "ModeConfig")
        .def(py::init<>())
        .def_readwrite("id", &ModeConfig::id)
        .def_readwrite("stacked_obs_order", &ModeConfig::stacked_obs_order)
        .def_readwrite("non_stacked_obs_order", &ModeConfig::non_stacked_obs_order)
        .def_readwrite("stack_size", &ModeConfig::stack_size)
        .def_readwrite("policy_path", &ModeConfig::policy_path)
        .def_readwrite("policy_type", &ModeConfig::policy_type)
        .def_readwrite("cmd_vector_length", &ModeConfig::cmd_vector_length)
        .def("set_obs_scale_item", [](ModeConfig& self, const std::string& key, py::object val){
            self.obs_scale[key] = py_to_scale_spec(val);
        })
        // set_cmd_scale is removed; command scale should be provided via obs_scale with key "command"
        .def("set_action_scale", [](ModeConfig& self, py::object val){
            self.action_scale = py_to_scale_spec(val);
        });

    // Mode
    py::class_<Mode>(m, "Mode")
        // C++ 스타일 생성자 (ModeConfig)
        .def(py::init<const ModeConfig&>(), py::arg("config"))

        // 파이썬 dict 기반 생성자: Mode(mode_cfg={...})
        .def(py::init([](py::object mode_cfg){
            if (mode_cfg.is_none() || !py::isinstance<py::dict>(mode_cfg)) {
                const std::string r = py::str(py::repr(mode_cfg));
                throw ModeConfigError("mode_cfg must be a dict, got: " + r);
            }
            ModeConfig cfg = modeconfig_from_dict(py::cast<py::dict>(mode_cfg));
            return Mode(cfg);
        }), py::arg("mode_cfg"))

        // inference(state): 1D list/array of float ONLY
        .def("inference", [](Mode& self, py::object state_obj) {
            // 1D만 허용
            if (py::isinstance<py::sequence>(state_obj)) {
                py::sequence seq = state_obj;
                if (py::len(seq) > 0 && py::isinstance<py::sequence>(seq[0])) {
                    throw std::runtime_error("state must be a 1D list/array");
                }
            }
            std::vector<float> state = state_obj.cast<std::vector<float>>();
            return self.inference(state);
        })

        // ===== 읽기 전용 프로퍼티들 (한 번에 전부 노출) =====
        .def_property_readonly("id", &Mode::id)
        .def_property_readonly("stack_size", &Mode::stack_size)
        .def_property_readonly("state_length", &Mode::state_length)
        .def_property_readonly("action_length", &Mode::action_length)
        .def_property_readonly("policy_type", &Mode::policy_type)
        .def_property_readonly("policy_path", &Mode::policy_path)
        .def_property_readonly("stacked_obs_order", &Mode::stacked_obs_order)
        .def_property_readonly("non_stacked_obs_order", &Mode::non_stacked_obs_order)
        .def_property_readonly("action_scale", &Mode::action_scale)
        // cmd_scale property removed; use obs_scale["command"]
        .def_property_readonly("obs_scale", &Mode::obs_scale)
        .def_property_readonly("obs_lengths", &Mode::obs_lengths)
        .def_property_readonly("cmd_vector_length", &Mode::cmd_vector_length);
}
