import torch
import torch.nn.functional as F


def variance_focus(U):
    """
    计算基于方差的聚焦度量值
    VF = variance(abs(U(z)))

    Args:
        U: 输入图像张量，可以是 (H, W) 或 (C, H, W) 或 (B, C, H, W)

    Returns:
        VF: 方差聚焦度量值
    """
    # 处理不同维度的输入
    original_dim = U.dim()

    # 统一转换为4D张量 (B, C, H, W)
    if U.dim() == 2:  # (H, W)
        U = U.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif U.dim() == 3:  # (C, H, W)
        U = U.unsqueeze(0)  # (1, C, H, W)
    elif U.dim() == 4:  # (B, C, H, W)
        pass  # 已经是4D
    else:
        raise ValueError(f"不支持的张量维度: {U.dim()}")

    batch_size, channels, h, w = U.shape

    # 如果是多通道，转换为灰度
    if channels > 1:
        U = U.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        channels = 1

    # 1. 计算绝对值
    abs_U = torch.abs(U)

    # 2. 计算方差
    # 沿着空间维度 (H, W) 计算方差
    variance = torch.var(abs_U, dim=(2, 3), unbiased=False)  # (B, 1)

    # 3. 获取方差值
    VF = variance.squeeze(1)  # (B,)

    # 根据原始维度返回适当的结果
    if original_dim == 2 or original_dim == 3:
        return VF.item()  # 单张图像返回标量
    else:
        return VF  # 批量图像返回张量


def variance_focus_advanced(U, method='standard'):
    """
    高级版本的方差聚焦度量，支持不同的方差计算方法

    Args:
        U: 输入图像张量
        method: 方差计算方法
               'standard' - 标准方差
               'normalized' - 归一化方差 (方差/均值)
               'log_variance' - 对数方差

    Returns:
        VF: 方差聚焦度量值
    """
    # 处理不同维度的输入
    original_dim = U.dim()

    # 统一转换为4D张量 (B, C, H, W)
    if U.dim() == 2:  # (H, W)
        U = U.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif U.dim() == 3:  # (C, H, W)
        U = U.unsqueeze(0)  # (1, C, H, W)
    elif U.dim() == 4:  # (B, C, H, W)
        pass  # 已经是4D
    else:
        raise ValueError(f"不支持的张量维度: {U.dim()}")

    batch_size, channels, h, w = U.shape

    # 如果是多通道，转换为灰度
    if channels > 1:
        U = U.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        channels = 1

    # 计算绝对值
    abs_U = torch.abs(U)

    if method == 'standard':
        # 标准方差
        variance = torch.var(abs_U, dim=(2, 3), unbiased=False)  # (B, 1)
        VF = variance.squeeze(1)  # (B,)

    elif method == 'normalized':
        # 归一化方差 (方差/均值)
        mean_val = torch.mean(abs_U, dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
        variance = torch.var(abs_U, dim=(2, 3), unbiased=False)  # (B, 1)
        # 避免除以零
        normalized_variance = variance / (mean_val.squeeze(2).squeeze(2) + 1e-8)
        VF = normalized_variance.squeeze(1)  # (B,)

    elif method == 'log_variance':
        # 对数方差
        variance = torch.var(abs_U, dim=(2, 3), unbiased=False)  # (B, 1)
        log_variance = torch.log(variance + 1e-8)  # 避免log(0)
        VF = log_variance.squeeze(1)  # (B,)

    else:
        raise ValueError(f"不支持的方差计算方法: {method}")

    # 根据原始维度返回适当的结果
    if original_dim == 2 or original_dim == 3:
        return VF.item()  # 单张图像返回标量
    else:
        return VF  # 批量图像返回张量


def simple_variance_focus(U):
    """
    简化版本的方差聚焦度量，假设输入是2D或3D张量
    适用于自动聚焦场景
    """
    # 确保是2D张量
    if U.dim() == 3:
        if U.shape[0] == 1:  # (1, H, W)
            U = U.squeeze(0)  # (H, W)
        else:  # (C, H, W)
            U = U.mean(dim=0)  # (H, W)

    # 添加批次和通道维度
    U_4d = U.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 计算绝对值
    abs_U = torch.abs(U_4d)

    # 计算方差
    variance = torch.var(abs_U, dim=(2, 3), unbiased=False)  # (1, 1)
    VF = variance.squeeze(0).squeeze(0)  # 标量

    return VF.item()


def variance_focus_with_mask(U, mask_ratio=0.1):
    """
    带掩码的方差聚焦度量，只计算图像中心区域
    可以减少边缘效应的影响

    Args:
        U: 输入图像张量
        mask_ratio: 中心区域的比例 (0-1)

    Returns:
        VF: 方差聚焦度量值
    """
    # 处理不同维度的输入
    original_dim = U.dim()

    # 统一转换为4D张量 (B, C, H, W)
    if U.dim() == 2:  # (H, W)
        U = U.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif U.dim() == 3:  # (C, H, W)
        U = U.unsqueeze(0)  # (1, C, H, W)
    elif U.dim() == 4:  # (B, C, H, W)
        pass  # 已经是4D
    else:
        raise ValueError(f"不支持的张量维度: {U.dim()}")

    batch_size, channels, h, w = U.shape

    # 如果是多通道，转换为灰度
    if channels > 1:
        U = U.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        channels = 1

    # 计算绝对值
    abs_U = torch.abs(U)

    # 创建中心掩码
    center_h_start = int(h * (1 - mask_ratio) / 2)
    center_h_end = int(h * (1 + mask_ratio) / 2)
    center_w_start = int(w * (1 - mask_ratio) / 2)
    center_w_end = int(w * (1 + mask_ratio) / 2)

    # 提取中心区域
    center_region = abs_U[:, :, center_h_start:center_h_end, center_w_start:center_w_end]

    # 计算中心区域的方差
    variance = torch.var(center_region, dim=(2, 3), unbiased=False)  # (B, 1)
    VF = variance.squeeze(1)  # (B,)

    # 根据原始维度返回适当的结果
    if original_dim == 2 or original_dim == 3:
        return VF.item()  # 单张图像返回标量
    else:
        return VF  # 批量图像返回张量


if __name__ == "__main__":
    # 使用示例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 单张图像
    U_single = torch.randn(256, 256, device=device)  # (H, W)
    vf_value = variance_focus(U_single)
    print(f"单张图像的 VF = {vf_value}")

    # 多通道图像
    U_rgb = torch.randn(3, 256, 256, device=device)  # (C, H, W)
    vf_value_rgb = variance_focus(U_rgb)
    print(f"RGB图像的 VF = {vf_value_rgb}")

    # 批量图像
    U_batch = torch.randn(4, 3, 256, 256, device=device)  # (B, C, H, W)
    vf_batch = variance_focus(U_batch)
    print(f"批量图像的 VF = {vf_batch}")

    # 简化版本测试
    vf_simple = simple_variance_focus(U_single)
    print(f"简化版本的 VF = {vf_simple}")

    # 高级版本测试
    vf_normalized = variance_focus_advanced(U_single, method='normalized')
    print(f"归一化方差 VF = {vf_normalized}")

    vf_log = variance_focus_advanced(U_single, method='log_variance')
    print(f"对数方差 VF = {vf_log}")

    # 带掩码版本测试
    vf_masked = variance_focus_with_mask(U_single, mask_ratio=0.5)
    print(f"带掩码的 VF = {vf_masked}")