'''
컴파일된 커스텀 커널을 사용하는 QuantLinear 레이어를 정의한다.
'''
import torch
import torch.nn as nn 
import my_quant_lib # setup.py를 통해 컴파일된 라이브러리

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 패킹된 가중치는 int32 타입으로 저장됨(8개의 int4가 패킹)
        self.register_buffer('weight', torch.empty(out_features, in_features // 8, dtype = torch.int32))
        self.register_buffer('scale', torch.empty(out_features))
        
    def forward(self, x):
        # 컴파일된 C++ 라이브러리의 함수를 직접 호출
        return my_quant_lib.int4_gemv(x, self.weight, self.scale)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'