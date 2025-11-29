#include "onnxpolicy.hpp"

namespace onnxpolicy {


// Query input/output tensor shape from session (dynamic dims may be returned as -1)
std::vector<int64_t> get_shape(const Ort::Session& session, size_t idx, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::TypeInfo ti = input ? session.GetInputTypeInfo(idx) : session.GetOutputTypeInfo(idx);
    auto tt = ti.GetTensorTypeAndShapeInfo();
    auto dims = tt.GetShape();
    return dims; // may contain -1 for dynamic dims
}

// Get input/output node name from session as string
std::string get_name(const Ort::Session& session, size_t idx, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto s = input ? session.GetInputNameAllocated(idx, alloc)
                   : session.GetOutputNameAllocated(idx, alloc);
    return std::string{s.get()};
}

// Fallback utility for dynamic dimensions
int64_t value_or(const int64_t x, int64_t fallback) {
    // ONNX uses -1 or 0 to denote dynamic. Treat <=0 as unknown.
    return (x > 0) ? x : fallback;
}


/*==========================
 * MLPPolicy (C++)
 *==========================*/
MLPPolicy::MLPPolicy(const std::string& weight_path)
: env(ORT_LOGGING_LEVEL_WARNING, "onnxpolicy"), session(nullptr)
{
    // Session options (single-threaded, sequential execution)
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session = Ort::Session(env, weight_path.c_str(), so);

    // Shared memory/options cache (run_opts stays default (empty options))
    mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    // Validate minimum number of inputs/outputs
    if (session.GetInputCount() < 1) {
        throw std::runtime_error("MLPPolicy: model has no inputs.");
    }
    if (session.GetOutputCount() < 1) {
        throw std::runtime_error("MLPPolicy: model has no outputs.");
    }

    // Cache first input/output names
    input_name  = get_name(session, 0, /*input=*/true);
    output_name = get_name(session, 0, /*input=*/false);
    input_name_c  = input_name.c_str();
    output_name_c = output_name.c_str();

    auto in_shape = get_shape(session, 0, /*input=*/true);
    batch_required = (!in_shape.empty() && in_shape[0] == 1);

    state_dim = (in_shape.empty() ? -1 : value_or(in_shape.back(), -1));
    if (state_dim <= 0) {
        throw std::runtime_error(
            "ONNX Error: dynamic or unknown state dimension detected. Export the model with a fixed last input dimension (>0)");
    } else {
        input_dims_template = {1, state_dim};
    }
}

// state: observation vector (shape: [state_dim])
// return: action vector (clip[-1,1])
std::vector<float> MLPPolicy::inference(const std::vector<float>& state) {
    if (state_dim > 0 && static_cast<int64_t>(state.size()) != state_dim) {
        throw std::runtime_error(
            "MLPPolicy: state size mismatch: expected " +
            std::to_string(state_dim) + " but got " +
            std::to_string(state.size())
        );
    }

    // Prepare input tensor: always [1, state_dim]
    std::vector<int64_t> input_dims;
    const float* src_ptr = state.data();

    input_dims = input_dims_template;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(src_ptr), state.size(),
        input_dims.data(), input_dims.size()
    );

    const char* input_names[]  = { input_name_c };
    const char* output_names[] = { output_name_c };

    auto outputs = session.Run(run_opts, input_names, &input_tensor, 1, output_names, 1);

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("MLPPolicy: unexpected output.");
    }

    auto& out = outputs[0];
    auto tt = out.GetTensorTypeAndShapeInfo();
    auto out_count = tt.GetElementCount();

    const float* out_data = out.GetTensorData<float>();
    std::vector<float> result(out_count);
    std::transform(out_data, out_data + out_count, result.begin(), clip_unit);

    if (batch_required && tt.GetShape().size() >= 2 && tt.GetShape()[0] == 1) {
        return result;
    }
    return result;
}


/*==========================
 * LSTMPolicy (C++) — final with default-name fallbacks
 *==========================*/
LSTMPolicy::LSTMPolicy(const std::string& weight_path)
: env(ORT_LOGGING_LEVEL_WARNING, "onnxpolicy"), session(nullptr)
{
    // Session options and session creation
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session = Ort::Session(env, weight_path.c_str(), so);

    // Shared memory/options cache
    mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    // Collect all input names (fixed order) + build index map
    const size_t n_inputs = session.GetInputCount();
    input_names.resize(n_inputs);
    input_cstrs.resize(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        input_names[i] = get_name(session, i, /*input=*/true);
        input_cstrs[i] = input_names[i].c_str();
        input_index_by_name[input_names[i]] = i;
    }

    // ---- Define candidates (place default names at the end) ----
    static const std::vector<std::string> state_candidates = {
        "state", "obs", "observation", "observations",
        "input", "input_0", "input0"
    };
    static const std::vector<std::string> h_in_candidates = {
        "h_in", "hidden_in", "h0", "h",
        "input_1", "input1"
    };
    static const std::vector<std::string> c_in_candidates = {
        "c_in", "cell_in", "c0", "c",
        "input_2", "input2"
    };

     // ---- Pick the first match among candidates, optionally checking rank (#dims) ----
    auto pick_with_rank = [&](const std::vector<std::string>& cands,
                              const char* role,
                              int expected_rank /* -1=ignore */) -> size_t {
        for (const auto& nm : cands) {
            auto it = input_index_by_name.find(nm);
            if (it == input_index_by_name.end()) continue;

            if (expected_rank >= 0) {
                auto shp = get_shape(session, it->second, /*input=*/true);
                int rank = static_cast<int>(shp.size());
                if (rank != expected_rank) continue; // select only when rank matches
            }
            return it->second;
        }

        // Error message
        std::string msg = std::string("Missing ") + role + " input. Tried {";
        for (size_t i = 0; i < cands.size(); ++i) {
            msg += cands[i];
            if (i + 1 < cands.size()) msg += ", ";
        }
        msg += "}. Available inputs: ";
        for (size_t i = 0; i < input_names.size(); ++i) {
            msg += input_names[i];
            if (i + 1 < input_names.size()) msg += ", ";
        }
        throw std::runtime_error(msg);
    };

    // ---- Finalize indices ----
    state_idx = pick_with_rank(state_candidates, "state", /*expected_rank=*/2);
    h_idx     = pick_with_rank(h_in_candidates, "hidden (h)", /*expected_rank=*/3);
    c_idx     = pick_with_rank(c_in_candidates, "cell (c)",   /*expected_rank=*/3);
    state_name = input_names[state_idx];

    // Estimate hidden/cell dimensions (based on input shape, typically [1,1,H])
    auto h_in_shape = get_shape(session, h_idx, true);
    auto c_in_shape = get_shape(session, c_idx, true);
    h_dim = (!h_in_shape.empty() ? value_or(h_in_shape.back(), 1) : 1);
    c_dim = (!c_in_shape.empty() ? value_or(c_in_shape.back(), 1) : 1);

    // Sequence/batch size (fixed to 1 in this implementation)
    batch_size = 1;
    seq_len    = 1;

    // Initialize internal state buffers (fixed size -> pointer stability)
    policy_h_in.assign(static_cast<size_t>(h_dim), 0.0f);
    policy_c_in.assign(static_cast<size_t>(c_dim), 0.0f);

    // Infer state dimension (throw if dynamic/unknown)
    auto state_shape = get_shape(session, state_idx, true);
    state_dim = (!state_shape.empty() ? value_or(state_shape.back(), -1) : -1);
    if (state_dim <= 0) {
        throw std::runtime_error(
            "ONNX Error: dynamic or unknown state dimension detected. "
            "Export the model with a fixed last input dimension (>0)"
        );
    }

    // Cache output names
    const size_t n_outputs = session.GetOutputCount();
    output_names.resize(n_outputs);
    output_cstrs.resize(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        output_names[i] = get_name(session, i, /*input=*/false);
        output_cstrs[i] = output_names[i].c_str();
    }

    // h/c tensor shape template ([1,1,H])
    hc_dims = {seq_len, batch_size, h_dim};
    cc_dims = {seq_len, batch_size, c_dim};

    // Prepare zero buffers for extra (unknown) inputs
    //  - For dynamic dims (-1/0), use materialized dims with 1 substituted, and create zero buffer of that size
    extra_input_dims.resize(n_inputs);
    zero_holders.resize(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        if (i == state_idx || i == h_idx || i == c_idx) continue;

        auto shp = get_shape(session, i, /*input=*/true);
        std::vector<int64_t> mat_dims = shp;
        if (mat_dims.empty()) mat_dims.push_back(1); // 스칼라 대비
        for (auto& d : mat_dims) d = value_or(d, 1);

        size_t cnt = 1;
        for (auto d : mat_dims) cnt *= static_cast<size_t>(d);

        extra_input_dims[i] = std::move(mat_dims);
        zero_holders[i]     = std::vector<float>(cnt, 0.0f);
    }
}

// Single time-step inference
std::vector<float> LSTMPolicy::inference(const std::vector<float>& state) {
    if (state_dim > 0 && static_cast<int64_t>(state.size()) != state_dim) {
        throw std::runtime_error(
            "LSTMPolicy: state size mismatch: expected " +
            std::to_string(state_dim) + " but got " +
            std::to_string(state.size())
        );
    }

    // state -> [1, state_dim]
    std::vector<int64_t> state_dims = {1, static_cast<int64_t>(state.size())};
    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(state.data()), state.size(),
        state_dims.data(), state_dims.size()
    );

    // h_in, c_in -> [1,1,H] (create Ort::Value per run; reuse buffers)
    Ort::Value h_in_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        policy_h_in.data(), policy_h_in.size(),
        hc_dims.data(), hc_dims.size()
    );
    Ort::Value c_in_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        policy_c_in.data(), policy_c_in.size(),
        cc_dims.data(), cc_dims.size()
    );

    // Build input binding vector (preserve original input order)
    std::vector<Ort::Value> in_vals(session.GetInputCount());
    for (size_t i = 0; i < in_vals.size(); ++i) {
        if (i == state_idx) {
            in_vals[i] = std::move(state_tensor);
        } else if (i == h_idx) {
            in_vals[i] = std::move(h_in_tensor);
        } else if (i == c_idx) {
            in_vals[i] = std::move(c_in_tensor);
        } else {
            // Reuse zero buffers (Ort::Value is recreated each call) with materialized dims
            auto& zeros = zero_holders[i];
            auto& dims  = extra_input_dims[i];
            Ort::Value zt = Ort::Value::CreateTensor<float>(
                mem_info,
                zeros.data(), zeros.size(),
                dims.data(), dims.size()
            );
            in_vals[i] = std::move(zt);
        }
    }

    auto outputs = session.Run(run_opts,
                               input_cstrs.data(), in_vals.data(), in_vals.size(),
                               output_cstrs.data(), output_cstrs.size());

    if (outputs.empty()) {
        throw std::runtime_error("LSTMPolicy: no outputs from session.");
    }

    // Find new h/c states in outputs and copy into internal buffers (without reallocation if possible)
    update_hidden_from_outputs(outputs);

    // Treat the first output tensor as the action, clip, and return
    auto& out0 = outputs[0];
    if (!out0.IsTensor()) throw std::runtime_error("LSTMPolicy: first output is not a tensor.");
    auto tt = out0.GetTensorTypeAndShapeInfo();
    size_t count = tt.GetElementCount();
    const float* data = out0.GetTensorData<float>();
    std::vector<float> action(count);
    std::transform(data, data + count, action.begin(), clip_unit);
    return action;
}

void LSTMPolicy::update_hidden_from_outputs(const std::vector<Ort::Value>& outs) {
    // Possible name candidates (in priority order) — append default name at the end
    static const std::vector<std::string> h_names = {
        "h_out", "hn", "hidden", "h", "output_1", "output1"
    };
    static const std::vector<std::string> c_names = {
        "c_out", "cn", "cell", "c", "output_2", "output2"
    };

    // Map: output name -> index
    std::unordered_map<std::string, size_t> out_idx;
    out_idx.reserve(output_names.size());
    for (size_t i = 0; i < output_names.size(); ++i) {
        out_idx[output_names[i]] = i;
    }

    auto try_update = [&](const std::vector<std::string>& names,
                          std::vector<float>& holder, int64_t expect_last) -> bool {
        for (const auto& nm : names) {
            auto it = out_idx.find(nm);
            if (it == out_idx.end()) continue;
            const auto& val = outs[it->second];
            if (!val.IsTensor()) continue;

            auto tt = val.GetTensorTypeAndShapeInfo();
            auto shp = tt.GetShape(); // Expected: [1,1,H]
            if (shp.size() != 3) continue;
            if (expect_last > 0 && value_or(shp.back(), expect_last) != expect_last) continue;

            size_t cnt = static_cast<size_t>(tt.GetElementCount()); // Typically H
            const float* data = val.GetTensorData<float>();

            if (holder.size() == cnt) {
                std::copy(data, data + cnt, holder.begin());        // no reallocation
            } else {
                holder.assign(data, data + cnt);                    // allow size change
            }
            return true;
        }
        return false;
    };

    (void)try_update(h_names, policy_h_in, h_dim);
    (void)try_update(c_names, policy_c_in, c_dim);
}

} // namespace onnxpolicy