/*
kernel.cu에 정의된 int4_gemv_launcher 함수를 
Python에서 호출할 수 있도록 연결하는 접착제 코드
*/
#include <torch/extension.h>

// CUDA 파일에 선언된 런처 함수
void int4_gemv_launcher(
    torch::Tensor input,
    torch::Tensor packed_w,
    torch::Tensor scale,
    torch::Tensor output
);

// Python에 노출될 메인 함수
torch::Tensor int4_gemv(
    torch::Tensor input,
    torch::Tensor packed_w,
    torch::Tensor scale
){
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(intput.dim()==2, "Input must be 2-dimensional");

    auto output = torch::empty({input.size(0), packed_w.size(0)}, input.options());

    int4_gemv_launcher(intput, packed_w, scale, output);
    return output;
}

// "my_quant_lib" 라는 이름의 파이썬 모듈에 "int4_gemv" 함수를 등록
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("int4_gemv", &int4_gemv, "4-bit Integer GEMV kernel");
}