import torch
import numpy as np


def eigenvalue_focus_metric(U, k_percent=0.01):
    def compute_eig_metric(image_2d, k_percent):
        M, N = image_2d.shape

        energy = torch.sqrt(torch.sum(image_2d ** 2))
        if energy == 0:
            return 0.0
        A_bar = image_2d / energy

        mu = torch.mean(A_bar)
        A_centered = A_bar - mu

        # ★ 原来：显式构造 M×M 协方差矩阵再 SVD → O(M³)
        # Q = torch.mm(A_centered, A_centered.T) / (M - 1)
        # _, S, _ = torch.svd(Q)
        # eigenvalues = S
        #
        # ★ 现在：直接对 A_centered 做 thin-SVD，奇异值²/(M-1) = 协方差矩阵特征值
        # 数学完全等价，但跳过了 M×M 矩阵的构造和分解
        S = torch.linalg.svdvals(A_centered)          # shape: (min(M,N),)
        eigenvalues = S ** 2 / (M - 1)

        # 原来 eigvalsh 返回升序，svd 返回降序，需要手动升序保持一致
        eigenvalues, _ = torch.sort(eigenvalues)

        k = int(k_percent * M)
        L = torch.sum(eigenvalues[:M - k])

        return L.item()

    if U.dim() == 4:
        batch_size, channels, h, w = U.shape
        eig_values = []
        for i in range(batch_size):
            single_image = U[i]
            if channels > 1:
                single_image = single_image.mean(dim=0)
            eig_val = eigenvalue_focus_metric(single_image, k_percent)
            eig_values.append(eig_val)
        return torch.tensor(eig_values)

    elif U.dim() == 3:
        if U.shape[0] > 1:
            U_gray = U.mean(dim=0)
        else:
            U_gray = U.squeeze(0)
        return eigenvalue_focus_metric(U_gray, k_percent)

    elif U.dim() == 2:
        return compute_eig_metric(U, k_percent)

    else:
        raise ValueError(f"不支持的张量维度: {U.dim()}")


def eigenvalue_focus_metric_complex(U, k_percent=0.01):
    if not torch.is_complex(U):
        raise ValueError("输入必须是复数张量")

    U_real = U.real
    U_imag = U.imag
    U_combined = torch.stack([U_real, U_imag], dim=0)

    return eigenvalue_focus_metric(U_combined, k_percent)


def eigenvalue_focus_metric_block(U, k_percent=0.01, num_blocks=4):
    if U.dim() != 2:
        if U.dim() == 3:
            U = U.mean(dim=0) if U.shape[0] > 1 else U.squeeze(0)
        elif U.dim() == 4:
            U = U[0].mean(dim=0) if U.shape[1] > 1 else U[0].squeeze(0)

    H, W = U.shape
    block_h = H // num_blocks
    block_w = W // num_blocks

    metrics = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            h_start = i * block_h
            h_end = (i + 1) * block_h if i < num_blocks - 1 else H
            w_start = j * block_w
            w_end = (j + 1) * block_w if j < num_blocks - 1 else W

            block = U[h_start:h_end, w_start:w_end]

            if block.numel() > 0:
                block_metric = eigenvalue_focus_metric(block, k_percent)
                metrics.append(block_metric)

    return np.mean(metrics) if metrics else 0.0


def calculate_eig_focus_curve(reconstructed_images, k_percent=0.02, use_blocks=False):
    focus_curve = 0

    for i, image in enumerate(reconstructed_images):
        if use_blocks:
            eig_val = eigenvalue_focus_metric_block(image, k_percent)
        else:
            eig_val = eigenvalue_focus_metric(image, k_percent)
        focus_curve += eig_val

    return focus_curve