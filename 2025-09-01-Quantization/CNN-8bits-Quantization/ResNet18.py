# fake_quant_ste.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models

# -----------------------------
# 0) 공통 유틸
# -----------------------------
def qrange(num_bits=8, signed=True):
    if signed:      # int8 대칭: -127..127 (대칭성과 안정성을 위해 -128 제외)
        return -(2**(num_bits-1)-1), (2**(num_bits-1)-1)
    else:           # uint8 비대칭: 0..255
        return 0, (2**num_bits-1)

def get_scale_zero_point_asym(a, b, num_bits=8):
    """비대칭(주로 활성화) S,Z 계산"""
    qmin, qmax = qrange(num_bits=num_bits, signed=False)
    a, b = float(a), float(b)
    eps = 1e-12
    if b - a < eps:
        return 1.0, 0
    S = (b - a) / (qmax - qmin)
    Z = round(qmin - a / S)
    Z = int(max(qmin, min(qmax, Z)))
    return S, Z

def get_scale_zero_point_sym(max_abs, num_bits=8):
    """대칭(주로 가중치) S,Z 계산, Z=0"""
    qmin, qmax = qrange(num_bits=num_bits, signed=True)  # -127..127
    S = (max_abs / qmax) if max_abs > 0 else 1.0
    Z = 0
    return S, Z

def tensor_minmax(x, percentile=None, ch_axis=None):
    """퍼센타일 클리핑 옵션. per-tensor 또는 per-channel min/max 반환."""
    if ch_axis is None:
        flat = x.reshape(-1)
        if percentile is None:
            return flat.min().item(), flat.max().item()
        lo = torch.quantile(flat, percentile/100.0).item()
        hi = torch.quantile(flat, 1 - percentile/100.0).item()
        return lo, hi
    else:
        # per-channel min/max (채널 차원만 남기고 나머지 축 reduce)
        reduce_dims = tuple(d for d in range(x.ndim) if d != ch_axis)
        if percentile is None:
            lo = x.amin(dim=reduce_dims)
            hi = x.amax(dim=reduce_dims)
        else:
            # 간단히 정렬 근사(퍼센타일을 엄밀히 하고 싶으면 채널별로 quantile 호출)
            lo = x.movedim(ch_axis, 0).reshape(x.size(ch_axis), -1)
            hi = lo.clone()
            k_lo = max(0, int(percentile/100.0 * lo.size(1)) - 1)
            k_hi = min(lo.size(1)-1, int((1 - percentile/100.0) * lo.size(1)))
            lo, _ = lo.kthvalue(k_lo+1, dim=1)
            hi, _ = hi.kthvalue(k_hi+1, dim=1)
        return lo, hi  # 텐서 반환

# -----------------------------
# 1) 핵심: Fake-Quant + STE
# -----------------------------
class QuantDequantSTE(Function):
    @staticmethod
    def forward(ctx, x, S, Z, qmin, qmax):
        """
        x: FP32 텐서
        S: scale (broadcastable)
        Z: zero-point (broadcastable, 정수 취급)
        qmin/qmax: 정수 범위 (스칼라 int)
        """
        # 양자화
        q = torch.round(x / S) + Z
        q_clamped = torch.clamp(q, qmin, qmax)
        # 포워드 마스크: clamp 영역 안에 있는지(=기울기 통과)
        x_min = (qmin - Z) * S
        x_max = (qmax - Z) * S
        mask = (x >= x_min) & (x <= x_max)
        ctx.save_for_backward(mask)
        ctx.S = S
        ctx.Z = Z
        ctx.qmin = qmin
        ctx.qmax = qmax
        # 디양자화
        x_hat = S * (q_clamped - Z)
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        """
        STE: clamp 범위 내부는 grad 그대로 통과, 밖은 0으로 컷.
        (S, Z, qmin, qmax는 학습하지 않으므로 None)
        """
        (mask,) = ctx.saved_tensors
        grad_input = grad_output * mask.to(grad_output.dtype)
        return grad_input, None, None, None, None

class QATLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        # weight용 scale/zero-point 추정을 위해 min/max 러닝 값 추적 가능

    def forward(self, x):
        # weight quant
        w = self.fc.weight
        w_min, w_max = w.min(), w.max()
        S_w, Z_w = get_scale_zero_point_sym(max_abs=max(abs(float(w_min)), float(w_max)))
        w_q = QuantDequantSTE.apply(w, S_w, Z_w, -127, 127)

        # input quant
        a, b = tensor_minmax(x)
        S_x, Z_x = get_scale_zero_point_asym(a, b)
        x_q = QuantDequantSTE.apply(x, S_x, Z_x, 0, 255)

        # 선형 연산
        y = F.linear(x_q, w_q, self.fc.bias)

        # output quant (per-tensor 비대칭)
        a, b = tensor_minmax(y)
        S_y, Z_y = get_scale_zero_point_asym(a, b)
        y_q = QuantDequantSTE.apply(y, S_y, Z_y, 0, 255)
        return y_q

# ResNet18에 테스트해보기
from torchvision import models

if __name__ == "__main__":
    model = models.resnet18(weights=None, num_classes=10)

    # conv 레이어에 FakeQuant 래퍼를 적용해볼 수 있음
    # 예: conv1 레이어 테스트
    conv1 = model.conv1
    a, b = tensor_minmax(conv1.weight)
    S, Z = get_scale_zero_point_sym(max_abs=max(abs(float(a)), float(b)))
    w_q = QuantDequantSTE.apply(conv1.weight, S, Z, -127, 127)
    print("[ResNet18 conv1] 원래 weight 범위:", (conv1.weight.min().item(), conv1.weight.max().item()))
    print("[ResNet18 conv1] 양자화 weight 범위:", (w_q.min().item(), w_q.max().item()))

