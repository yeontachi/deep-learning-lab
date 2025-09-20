/*
 quant/kernel.cu
 
 GPU에서 직접 동작하는 4비트 행렬-벡터 곱셈의 핵심 코드
 Unpacking이 바로 여기서 발생
*/

#include <cuda_runtime.h>
#include <cstding>

// 4비트 가중치와 FP32 입력을 사용하여 행렬-벡터 곱셈을 수행하는 CUDA 커널
__global__ void int4_gemv_kernel(
    const float* input,          // FP32 입력 벡터
    const int32_t* packed_w,     // 8개의 INT4가 패킹된 가중치 행렬
    const float* scale,          // 역양자화를 위한 스케일 값
    float* output,               // 결과 벡터(FP32)
    int in_features,
    int out_features
){
    // 현재 스레드가 담당할 출력 뉴런의 인덱스 계산
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= out_features) return;

    float acc = 0.0f;

    // 패킹된 가중치 행의 시작 위치
    const int32_t* packed_w_row = packed_w + (row * in_features/8);
    const float scale_val = scale[row];

    // 내적(Dot Product) 계산
    for(int i=0; i<in_features/8; ++i){
        // 1. 패킹된 가중치 한 덩어리(int32)를 로드
        int32_t packed_val = packed_w_row[i];

        // 2. 루프를 돌며 8개의 4비트 값을 언패킹(Unpacking) & 계산
        #pragma unroll
        for(int j=0; j<8; ++j){
            // Unpacking 발생 지점
            // j번쨰 4비트 값 추출(0~15)
            int weight_int4 = (packed_val >> (j*4)) & 0x0F;

            // 3. 역양자화 후 내적 계산
            float weight_fp32 = static_cast<float>(weight_int4) * scale_val;
            // 입력 벡터의 해당 값과 곱셈 후 누적
            acc += input[i*8+j] * weight_fp32;
        }
    }
    output[row] = acc;
}

// C++ 바인딩 파일에서 호출할 런처 함수
void int4_gemv_launcher(
    torch::Tensor input,
    torch::Tensor packed_w,
    torch::Tensor scale,
    torch::Tensor output
){
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = packed_w.size(0);

    // CUDA 커널 실행을 위한 스레드 블록/그리드 크기 설정
    const int threads_per_block = 256;
    const int blocks_per_grid = (out_features + threads_per_block - 1) / threads_per_block;

    // 배치 내 각 샘플에 대해 커널 실행
    for(int b=0; b<batch_size; ++b){
        int4_gemv_kernel<<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>() + b * in_features,
            packed_w.data_ptr<int32_t>(),
            scale.data_ptr<float>(),
            output.data_ptr<float>() + b*out_features,
            in_features,
            out_features
        )
    }
}