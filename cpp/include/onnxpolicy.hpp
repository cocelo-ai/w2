#pragma once
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace onnxpolicy {

// [-1, 1] 구간으로 값 클리핑
inline float clip_unit(float x) noexcept {
    x = std::isfinite(x) ? x : 0.0f;
    return std::fmin(std::fmax(x, -1.0f), 1.0f);
}

// 세션에서 입력/출력 텐서의 shape를 조회(동적 차원은 -1 반환 가능)
std::vector<int64_t> get_shape(const Ort::Session& session, size_t idx, bool input=true);

// 세션에서 입력/출력 노드 이름을 문자열로 얻기
std::string get_name(const Ort::Session& session, size_t idx, bool input=true);

// 동적 차원용 fallback 유틸
int64_t value_or(const int64_t x, int64_t fallback);


/*==========================
 * MLPPolicy (C++)
 *==========================*/
class MLPPolicy {
public:
    explicit MLPPolicy(const std::string& weight_path);

    // state: 관측값 벡터 (shape: [state_dim])
    // 반환: 액션 벡터 (clip[-1,1])
    std::vector<float> inference(const std::vector<float>& state);

private:
    // ONNX Runtime 리소스
    Ort::Env env_;
    Ort::SessionOptions so_;
    Ort::Session session_;

    // 공용 캐시
    Ort::MemoryInfo mem_info_{nullptr}; // CPU Arena allocator
    Ort::RunOptions run_opts_{};        // 기본 옵션

    // 모델의 첫 번째 입/출력 이름(및 c_str 캐시)
    std::string input_name_;
    std::string output_name_;
    const char* input_name_c_{nullptr};
    const char* output_name_c_{nullptr};

    // 입력 구성/검증 메타데이터
    bool        batch_required_{false};
    int64_t     state_dim_{-1};

    // 입력 텐서 템플릿/버퍼(알려진 경우)
    std::vector<int64_t> input_dims_template_;
};

/*==========================
 * LSTMPolicy (C++) — final with default-name fallbacks
 * - 상태(state) 입력 이름 후보: {"state","obs","observation","observations","input","input_0","input0"} (2D 기대)
 * - 은닉/셀 입력 이름 후보:
 *     h={"h_in","hidden_in","h0","h","input_1","input1"} (3D 기대)
 *     c={"c_in","cell_in","c0","c","input_2","input2"}   (3D 기대)
 * - 출력 이름 후보(상태 갱신용):
 *     h={"h_out","hn","hidden","h","output_1","output1"} (3D 기대)
 *     c={"c_out","cn","cell","c","output_2","output2"}   (3D 기대)
 * - 추가 입력은 동적차원(-1/0)을 1로 치환해 materialized dims로 제로 텐서 바인딩
 *==========================*/
class LSTMPolicy {
public:
    explicit LSTMPolicy(const std::string& weight_path);

    // 단일 타임스텝 추론
    std::vector<float> inference(const std::vector<float>& state);

private:
    void update_hidden_from_outputs(const std::vector<Ort::Value>& outs);

private:
    // ONNX Runtime 리소스
    Ort::Env env_;
    Ort::SessionOptions so_;
    Ort::Session session_;

    // 공용 캐시
    Ort::MemoryInfo mem_info_{nullptr};
    Ort::RunOptions run_opts_{};

    // 입력 이름/인덱스 매핑 및 순서 보존용 캐시
    std::unordered_map<std::string, size_t> input_index_by_name_;
    std::vector<std::string> input_names_;
    std::vector<const char*> input_cstrs_;

    // 출력 이름(c_str 포함)
    std::vector<std::string> output_names_;
    std::vector<const char*> output_cstrs_;

    // 인덱스 캐시
    size_t state_idx_{static_cast<size_t>(-1)};
    size_t h_idx_{static_cast<size_t>(-1)};
    size_t c_idx_{static_cast<size_t>(-1)};

    // dims / states
    int64_t h_dim_{1}, c_dim_{1};
    int64_t batch_size_{1}, seq_len_{1};
    int64_t state_dim_{-1};

    // 상태 버퍼(크기 고정)
    std::string state_name_{"state"};
    std::vector<float> policy_h_in_;
    std::vector<float> policy_c_in_;

    // h/c 텐서용 dims 템플릿
    std::vector<int64_t> hc_dims_;
    std::vector<int64_t> cc_dims_;

    // 추가 입력용 materialized dims / 제로 버퍼
    std::vector<std::vector<int64_t>> extra_input_dims_;
    std::vector<std::vector<float>>   zero_holders_;
};

} // namespace onnxpolicy
