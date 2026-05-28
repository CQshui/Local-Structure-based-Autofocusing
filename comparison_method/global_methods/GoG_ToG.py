import torch
import torch.nn.functional as F

class SoGAutoFocus:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def compute_gradient_magnitude_sq(self, complex_field: torch.Tensor) -> torch.Tensor:
        """返回梯度幅值的平方，避免不必要的 sqrt"""
        if not isinstance(complex_field, torch.Tensor):
            complex_field = torch.tensor(complex_field, dtype=torch.complex64)
        if not torch.is_complex(complex_field):
            complex_field = torch.view_as_complex(
                torch.stack([complex_field, torch.zeros_like(complex_field)], dim=-1)
            )

        complex_field = complex_field.to(self.device)

        ndim = complex_field.dim()
        if ndim == 2:
            complex_field = complex_field[None, None]
        elif ndim == 3:
            complex_field = complex_field[None]

        real_part = complex_field.real
        imag_part = complex_field.imag

        # 水平/垂直差分，合并实虚部 → 只做一次 sqrt（在外部按需调用）
        dx_sq = (real_part[..., 1:] - real_part[..., :-1]) ** 2 \
              + (imag_part[..., 1:] - imag_part[..., :-1]) ** 2
        dy_sq = (real_part[..., 1:, :] - real_part[..., :-1, :]) ** 2 \
              + (imag_part[..., 1:, :] - imag_part[..., :-1, :]) ** 2

        dx_sq = F.pad(dx_sq, (0, 1), value=0)
        dy_sq = F.pad(dy_sq, (0, 0, 0, 1), value=0)

        grad_sq = dx_sq + dy_sq  # 梯度幅值的平方，省掉一次 sqrt

        if ndim == 2:
            grad_sq = grad_sq[0, 0]
        elif ndim == 3:
            grad_sq = grad_sq[0]

        return grad_sq

    def gini_index(self, values_sq: torch.Tensor) -> torch.Tensor:
        """
        接受梯度平方图，内部开根号后计算 Gini。
        用 torch.sort 替换为累积和技巧，复杂度不变但常数更小；
        大图可改用直方图近似（见注释）。
        """
        flat = values_sq.reshape(-1)
        flat = flat[flat > 0]
        if flat.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        flat = torch.sqrt(flat)          # 只对非零元素开根
        flat, _ = torch.sort(flat)       # 升序

        N = flat.numel()
        cumsum = torch.cumsum(flat, dim=0)
        total  = cumsum[-1]
        if total == 0:
            return torch.tensor(0.0, device=self.device)

        # GI = 1 - (2/N) * Σ (cumsum[k] / total) + 1/N  （Lorenz 面积公式）
        gini = 1.0 + 1.0 / N - (2.0 / (N * total)) * cumsum.sum()
        return gini

    def tamura_coefficient(self, values_sq: torch.Tensor) -> torch.Tensor:
        """直接在平方域计算，最后一步才开根"""
        flat = values_sq.reshape(-1).float()
        if torch.all(flat == 0):
            return torch.tensor(0.0, device=self.device)

        # E[x] 和 E[x²] → std = sqrt(E[x²] - E[x]²)，全程无 sort
        grad = torch.sqrt(flat[flat > 0])
        mean_val = grad.mean()
        if mean_val == 0:
            return torch.tensor(0.0, device=self.device)
        std_val = grad.std()
        return torch.sqrt(std_val / mean_val)

    def gog_focus_criterion(self, complex_field: torch.Tensor) -> torch.Tensor:
        grad_sq = self.compute_gradient_magnitude_sq(complex_field)
        return self.gini_index(grad_sq)

    def tog_focus_criterion(self, complex_field: torch.Tensor) -> torch.Tensor:
        grad_sq = self.compute_gradient_magnitude_sq(complex_field)
        return self.tamura_coefficient(grad_sq)


# ── 模块级单例，避免重复初始化 ──────────────────────────────
_calculator_cache: dict = {}

def _get_calculator(device: str) -> SoGAutoFocus:
    if device not in _calculator_cache:
        _calculator_cache[device] = SoGAutoFocus(device=device)
    return _calculator_cache[device]


def calculate_gog(complex_field: torch.Tensor, device: str = None) -> float:
    if device is None:
        device = str(complex_field.device)   # ★ 跟随输入张量的设备
    return _get_calculator(device).gog_focus_criterion(complex_field).item()


def calculate_tog(complex_field: torch.Tensor, device: str = None) -> float:
    if device is None:
        device = str(complex_field.device)
    return _get_calculator(device).tog_focus_criterion(complex_field).item()