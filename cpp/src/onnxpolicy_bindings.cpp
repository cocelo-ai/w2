#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnxpolicy.hpp"

namespace py = pybind11;
using namespace onnxpolicy;

PYBIND11_MODULE(onnxpolicy, m) {
    m.doc() = "ONNX-based policies (MLP, LSTM) with ONNX Runtime";

    // MLPPolicy 
    py::class_<MLPPolicy>(m, "MLPPolicy")
        .def(py::init<const std::string&>(), py::arg("weightPath"),
             "Create MLPPolicy with an ONNX model file path")
        .def("inference",
            [](MLPPolicy& self, py::object stateObj) {
                // 1D 배열/리스트만 허용 (2D 이상이면 사용자 오류로 처리)
                if (py::isinstance<py::sequence>(stateObj)) {
                    py::sequence seq = stateObj;
                    if (py::len(seq) > 0 && py::isinstance<py::sequence>(seq[0])) {
                        throw std::runtime_error("state must be a 1D list/array");
                    }
                }
                std::vector<float> state = stateObj.cast<std::vector<float>>();
                return self.inference(state);
            },
            py::arg("state"),
            "Run inference: state -> action (values clipped to [-1,1])"
        );

    // LSTMPolicy
    py::class_<LSTMPolicy>(m, "LSTMPolicy")
        .def(py::init<const std::string&>(), py::arg("weightPath"),
             "Create LSTMPolicy with an ONNX model file path")
        .def(
            "inference",
            [](LSTMPolicy& self, py::object stateObj) {
                if (py::isinstance<py::sequence>(stateObj)) {
                    py::sequence seq = stateObj;
                    if (py::len(seq) > 0 && py::isinstance<py::sequence>(seq[0])) {
                        throw std::runtime_error("state must be a 1D list/array");
                    }
                }
                std::vector<float> state = stateObj.cast<std::vector<float>>();
                return self.inference(state);
            },
            py::arg("state"),
            "Run one-step inference (stateful; updates internal h/c)."
        );
}