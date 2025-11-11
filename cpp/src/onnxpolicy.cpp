#include "onnxpolicy.hpp"

namespace onnxpolicy {


// 세션에서 입력/출력 텐서의 shape를 조회(동적 차원은 -1 반환 가능)
std::vector<int64_t> get_shape(const Ort::Session& session, size_t idx, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::TypeInfo ti = input ? session.GetInputTypeInfo(idx) : session.GetOutputTypeInfo(idx);
    auto tt = ti.GetTensorTypeAndShapeInfo();
    auto dims = tt.GetShape();
    return dims; // may contain -1 for dynamic dims
}

// 세션에서 입력/출력 노드 이름을 문자열로 얻기
std::string get_name(const Ort::Session& session, size_t idx, bool input) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto s = input ? session.GetInputNameAllocated(idx, alloc)
                   : session.GetOutputNameAllocated(idx, alloc);
    return std::string{s.get()};
}

// 동적 차원용 fallback 유틸
int64_t value_or(const int64_t x, int64_t fallback) {
    // ONNX uses -1 or 0 to denote dynamic. Treat <=0 as unknown.
    return (x > 0) ? x : fallback;
}


/*==========================
 * MLPPolicy (C++)
 *==========================*/
MLPPolicy::MLPPolicy(const std::string& weight_path)
: env_(ORT_LOGGING_LEVEL_WARNING, "onnxpolicy"), session_(nullptr)
{
    // 세션 옵션(단일 스레드, 순차 실행)
    so_.SetIntraOpNumThreads(1);
    so_.SetInterOpNumThreads(1);
    so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_ = Ort::Session(env_, weight_path.c_str(), so_);

    // 공용 메모리/옵션 캐시
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    // run_opts_는 기본값(빈 옵션) 유지

    // 입력/출력 최소 개수 검증
    if (session_.GetInputCount() < 1) {
        throw std::runtime_error("MLPPolicy: model has no inputs.");
    }
    if (session_.GetOutputCount() < 1) {
        throw std::runtime_error("MLPPolicy: model has no outputs.");
    }

    // 첫 번째 입력/출력 이름 캐시
    input_name_  = get_name(session_, 0, /*input=*/true);
    output_name_ = get_name(session_, 0, /*input=*/false);
    input_name_c_  = input_name_.c_str();
    output_name_c_ = output_name_.c_str();

    auto in_shape = get_shape(session_, 0, /*input=*/true);
    batch_required_ = (!in_shape.empty() && in_shape[0] == 1);

    state_dim_ = (in_shape.empty() ? -1 : value_or(in_shape.back(), -1));
    if (state_dim_ <= 0) {
        throw std::runtime_error("ONNX Error: dynamic or unknown state dimension detected. Export the model with a fixed last input dimension (>0)");}
    else{
        input_dims_template_ = {1, state_dim_};
    }
}

// state: 관측값 벡터 (shape: [state_dim])
// 반환: 액션 벡터 (clip[-1,1])
std::vector<float> MLPPolicy::inference(const std::vector<float>& state) {
    if (state_dim_ > 0 && static_cast<int64_t>(state.size()) != state_dim_) {
        throw std::runtime_error(
            "MLPPolicy: state size mismatch: expected " +
            std::to_string(state_dim_) + " but got " +
            std::to_string(state.size())
        );
    }

    // 입력 텐서 준비: 항상 [1, state_dim]
    std::vector<int64_t> input_dims;
    const float* src_ptr = state.data();

    input_dims = input_dims_template_;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(src_ptr), state.size(),
        input_dims.data(), input_dims.size()
    );

    const char* input_names[]  = { input_name_c_ };
    const char* output_names[] = { output_name_c_ };

    auto outputs = session_.Run(run_opts_, input_names, &input_tensor, 1, output_names, 1);

    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("MLPPolicy: unexpected output.");
    }

    auto& out = outputs[0];
    auto tt = out.GetTensorTypeAndShapeInfo();
    auto out_count = tt.GetElementCount();

    const float* out_data = out.GetTensorData<float>();
    std::vector<float> result(out_count);
    std::transform(out_data, out_data + out_count, result.begin(), clip_unit);

    if (batch_required_ && tt.GetShape().size() >= 2 && tt.GetShape()[0] == 1) {
        return result;
    }
    return result;
}


/*==========================
 * LSTMPolicy (C++) — final with default-name fallbacks
 *==========================*/
LSTMPolicy::LSTMPolicy(const std::string& weight_path)
: env_(ORT_LOGGING_LEVEL_WARNING, "onnxpolicy"), session_(nullptr)
{
    // 세션 옵션 및 세션 생성
    so_.SetIntraOpNumThreads(1);
    so_.SetInterOpNumThreads(1);
    so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_ = Ort::Session(env_, weight_path.c_str(), so_);

    // 공용 메모리/옵션 캐시
    mem_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    // 모든 입력 이름 수집(순서 고정) + 인덱스 맵 구성
    const size_t n_inputs = session_.GetInputCount();
    input_names_.resize(n_inputs);
    input_cstrs_.resize(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        input_names_[i] = get_name(session_, i, /*input=*/true);
        input_cstrs_[i] = input_names_[i].c_str();
        input_index_by_name_[input_names_[i]] = i;
    }

    // ---- 후보 정의 (기본 이름을 맨 뒤에 배치) ----
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

    // ---- 후보 중 첫 매치를 고르되, 필요하면 랭크(차원 수) 검사 ----
    auto pick_with_rank = [&](const std::vector<std::string>& cands,
                              const char* role,
                              int expected_rank /* -1=ignore */) -> size_t {
        for (const auto& nm : cands) {
            auto it = input_index_by_name_.find(nm);
            if (it == input_index_by_name_.end()) continue;

            if (expected_rank >= 0) {
                auto shp = get_shape(session_, it->second, /*input=*/true);
                int rank = static_cast<int>(shp.size());
                if (rank != expected_rank) continue; // 랭크가 맞을 때만 선택
            }
            return it->second;
        }
        // 에러 메시지
        std::string msg = std::string("Missing ") + role + " input. Tried {";
        for (size_t i = 0; i < cands.size(); ++i) {
            msg += cands[i];
            if (i + 1 < cands.size()) msg += ", ";
        }
        msg += "}. Available inputs: ";
        for (size_t i = 0; i < input_names_.size(); ++i) {
            msg += input_names_[i];
            if (i + 1 < input_names_.size()) msg += ", ";
        }
        throw std::runtime_error(msg);
    };

    // ---- 인덱스 확정 ----
    state_idx_ = pick_with_rank(state_candidates, "state", /*expected_rank=*/2);
    h_idx_     = pick_with_rank(h_in_candidates, "hidden (h)", /*expected_rank=*/3);
    c_idx_     = pick_with_rank(c_in_candidates, "cell (c)",   /*expected_rank=*/3);
    state_name_ = input_names_[state_idx_];

    // 히든/셀 차원 추정(입력 shape 기준, 일반적으로 [1,1,H])
    auto h_in_shape = get_shape(session_, h_idx_, true);
    auto c_in_shape = get_shape(session_, c_idx_, true);
    h_dim_ = (!h_in_shape.empty() ? value_or(h_in_shape.back(), 1) : 1);
    c_dim_ = (!c_in_shape.empty() ? value_or(c_in_shape.back(), 1) : 1);

    // 시퀀스/배치 크기(본 구현은 1 고정)
    batch_size_ = 1;
    seq_len_    = 1;

    // 내부 상태 버퍼 초기화(크기 고정 -> 포인터 안정성 확보)
    policy_h_in_.assign(static_cast<size_t>(h_dim_), 0.0f);
    policy_c_in_.assign(static_cast<size_t>(c_dim_), 0.0f);

    // state 차원 추정 (동적/미지수면 예외)
    auto state_shape = get_shape(session_, state_idx_, true);
    state_dim_ = (!state_shape.empty() ? value_or(state_shape.back(), -1) : -1);
    if (state_dim_ <= 0) {
        throw std::runtime_error(
            "ONNX Error: dynamic or unknown state dimension detected. "
            "Export the model with a fixed last input dimension (>0)"
        );
    }

    // 출력 이름 캐시
    const size_t n_outputs = session_.GetOutputCount();
    output_names_.resize(n_outputs);
    output_cstrs_.resize(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        output_names_[i] = get_name(session_, i, /*input=*/false);
        output_cstrs_[i] = output_names_[i].c_str();
    }

    // h/c 텐서 shape 템플릿([1,1,H])
    hc_dims_ = {seq_len_, batch_size_, h_dim_};
    cc_dims_ = {seq_len_, batch_size_, c_dim_};

    // 추가 입력(모르는 입력)의 제로 버퍼 준비
    //  - 동적 차원(-1/0)은 1로 치환한 materialized dims 저장, 그 크기만큼 제로 버퍼 생성
    extra_input_dims_.resize(n_inputs);
    zero_holders_.resize(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        if (i == state_idx_ || i == h_idx_ || i == c_idx_) continue;

        auto shp = get_shape(session_, i, /*input=*/true);
        std::vector<int64_t> mat_dims = shp;
        if (mat_dims.empty()) mat_dims.push_back(1); // 스칼라 대비
        for (auto& d : mat_dims) d = value_or(d, 1);

        size_t cnt = 1;
        for (auto d : mat_dims) cnt *= static_cast<size_t>(d);

        extra_input_dims_[i] = std::move(mat_dims);
        zero_holders_[i]     = std::vector<float>(cnt, 0.0f);
    }
}

// 단일 타임스텝 추론
std::vector<float> LSTMPolicy::inference(const std::vector<float>& state) {
    if (state_dim_ > 0 && static_cast<int64_t>(state.size()) != state_dim_) {
        throw std::runtime_error(
            "LSTMPolicy: state size mismatch: expected " +
            std::to_string(state_dim_) + " but got " +
            std::to_string(state.size())
        );
    }

    // state -> [1, state_dim]
    std::vector<int64_t> state_dims = {1, static_cast<int64_t>(state.size())};
    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        const_cast<float*>(state.data()), state.size(),
        state_dims.data(), state_dims.size()
    );

    // h_in, c_in -> [1,1,H] (런타임마다 Ort::Value 생성; 버퍼는 재사용)
    Ort::Value h_in_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        policy_h_in_.data(), policy_h_in_.size(),
        hc_dims_.data(), hc_dims_.size()
    );
    Ort::Value c_in_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        policy_c_in_.data(), policy_c_in_.size(),
        cc_dims_.data(), cc_dims_.size()
    );

    // 입력 바인딩 벡터 구성(입력 순서를 그대로 따름)
    std::vector<Ort::Value> in_vals(session_.GetInputCount());
    for (size_t i = 0; i < in_vals.size(); ++i) {
        if (i == state_idx_) {
            in_vals[i] = std::move(state_tensor);
        } else if (i == h_idx_) {
            in_vals[i] = std::move(h_in_tensor);
        } else if (i == c_idx_) {
            in_vals[i] = std::move(c_in_tensor);
        } else {
            // 제로 버퍼 재사용(Ort::Value는 매 호출 생성), materialized dims 사용
            auto& zeros = zero_holders_[i];
            auto& dims  = extra_input_dims_[i];
            Ort::Value zt = Ort::Value::CreateTensor<float>(
                mem_info_,
                zeros.data(), zeros.size(),
                dims.data(), dims.size()
            );
            in_vals[i] = std::move(zt);
        }
    }

    auto outputs = session_.Run(run_opts_,
                                input_cstrs_.data(), in_vals.data(), in_vals.size(),
                                output_cstrs_.data(), output_cstrs_.size());

    if (outputs.empty()) {
        throw std::runtime_error("LSTMPolicy: no outputs from session.");
    }

    // 새로운 h/c 상태를 출력에서 찾아 내부 버퍼에 복사(가능하면 재할당 없이)
    update_hidden_from_outputs(outputs);

    // 첫 번째 출력 텐서를 액션으로 간주하고 clip하여 반환
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
    // 가능한 이름 후보(우선순위대로) — 기본 이름을 맨 뒤에 추가
    static const std::vector<std::string> h_names = {
        "h_out", "hn", "hidden", "h", "output_1", "output1"
    };
    static const std::vector<std::string> c_names = {
        "c_out", "cn", "cell", "c", "output_2", "output2"
    };

    // 맵: output name -> index
    std::unordered_map<std::string, size_t> out_idx;
    out_idx.reserve(output_names_.size());
    for (size_t i = 0; i < output_names_.size(); ++i) {
        out_idx[output_names_[i]] = i;
    }

    auto try_update = [&](const std::vector<std::string>& names,
                          std::vector<float>& holder, int64_t expect_last) -> bool {
        for (const auto& nm : names) {
            auto it = out_idx.find(nm);
            if (it == out_idx.end()) continue;
            const auto& val = outs[it->second];
            if (!val.IsTensor()) continue;

            auto tt = val.GetTensorTypeAndShapeInfo();
            auto shp = tt.GetShape(); // 기대: [1,1,H]
            if (shp.size() != 3) continue;
            if (expect_last > 0 && value_or(shp.back(), expect_last) != expect_last) continue;

            size_t cnt = static_cast<size_t>(tt.GetElementCount()); // 보통 H
            const float* data = val.GetTensorData<float>();

            if (holder.size() == cnt) {
                std::copy(data, data + cnt, holder.begin());     // 재할당 없음
            } else {
                holder.assign(data, data + cnt);                  // 사이즈 변동 시 갱신 허용
            }
            return true;
        }
        return false;
    };

    (void)try_update(h_names, policy_h_in_, h_dim_);
    (void)try_update(c_names, policy_c_in_, c_dim_);
}

} // namespace onnxpolicy
