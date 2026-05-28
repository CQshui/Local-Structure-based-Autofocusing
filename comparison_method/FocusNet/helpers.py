from tensorflow.keras.backend import tanh
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import math
import os
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf


class RegressionSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, fx_ref, fy_ref, d_path, b_size, color_mode='grayscale',
                 target_size=(256, 256), output_size=(256, 256),
                 save_dir=r'F:\dongjiayao\Pycharm\DEP-AF\FocusNet\tmp',
                 # -------------------- 新增散斑增强参数 --------------------
                 speckle_aug=False,          # 是否启用散斑增强
                 speckle_prob=0.5,           # 每张图应用增强的概率
                 speckle_contrast=(0.3, 1.0),# 散斑对比度范围 [min, max] 或标量
                 speckle_size=(2.0, 6.0),    # 散斑颗粒尺寸范围（像素）[min, max] 或标量
                 speckle_method='intensity'  # 保留扩展接口，当前固定 intensity
                 ):
        self.x = x
        self.y = y
        self.batch_size = b_size
        self.dir = d_path
        self.color_mode = color_mode
        self.target_height, self.target_width = target_size
        self.output_size = output_size
        self.pix = 1e-6
        self.save_dir = save_dir

        # -------------------- 散斑增强配置 --------------------
        self.speckle_aug      = speckle_aug
        self.speckle_prob     = speckle_prob
        self.speckle_contrast = speckle_contrast
        self.speckle_size     = speckle_size
        self.speckle_method   = speckle_method

        # 处理 fx_ref 和 fy_ref（支持标量或列表）
        def process_freq(freq, name):
            if isinstance(freq, (list, tuple, np.ndarray)):
                arr = np.asarray(freq, dtype=np.float32)
                if len(arr) != len(x):
                    raise ValueError(f"{name} list length must match x length")
                return arr
            else:
                return np.full(len(x), freq, dtype=np.float32)

        self.fx_ref = process_freq(fx_ref, 'fx_ref')
        self.fy_ref = process_freq(fy_ref, 'fy_ref')

    # ==================== 散斑增强辅助方法 ====================

    @staticmethod
    def _simulate_speckle_pattern(size, speckle_size=3.0, target_contrast=1.0):
        """生成符合目标对比度的散斑强度图样（均值=1）"""
        h, w = size
        # 随机复场
        random_complex = np.random.randn(h, w) + 1j * np.random.randn(h, w)
        # 频域高斯低通滤波（控制颗粒尺寸）
        y_grid, x_grid = np.ogrid[-h // 2:h // 2, -w // 2:w // 2]
        mask = np.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * (speckle_size ** 2)))
        mask = np.fft.fftshift(mask)
        # 频域滤波
        speckle_fft   = np.fft.fft2(random_complex) * mask
        speckle_field = np.fft.ifft2(speckle_fft)
        # 强度（完全发展散斑，对比度≈1）
        speckle_intensity = np.abs(speckle_field) ** 2
        speckle_intensity /= np.mean(speckle_intensity)   # 均值归一到 1
        # 调整对比度（与均匀背景线性混合）
        if target_contrast < 1.0:
            speckle_intensity = (target_contrast * speckle_intensity
                                 + (1 - target_contrast))
        return speckle_intensity

    def _speckle_augmentation(self, image_uint8):
        """
        对单张重建振幅图（uint8, H×W）应用散斑增强。
        按 self.speckle_prob 概率决定是否执行；标签不参与修改。

        Parameters
        ----------
        image_uint8 : np.ndarray, dtype=uint8, shape (H, W)

        Returns
        -------
        np.ndarray, dtype=uint8, shape (H, W)
        """
        if np.random.random() > self.speckle_prob:
            return image_uint8

        # 解析对比度
        c = (np.random.uniform(self.speckle_contrast[0], self.speckle_contrast[1])
             if isinstance(self.speckle_contrast, (list, tuple))
             else float(self.speckle_contrast))

        # 解析颗粒尺寸
        s = (np.random.uniform(self.speckle_size[0], self.speckle_size[1])
             if isinstance(self.speckle_size, (list, tuple))
             else float(self.speckle_size))

        # 生成散斑图样并相乘
        speckle   = self._simulate_speckle_pattern(image_uint8.shape[:2],
                                                    speckle_size=s,
                                                    target_contrast=c)
        img_float = image_uint8.astype(np.float32) / 255.0
        img_out   = np.clip(img_float * speckle * 255.0, 0, 255).astype(np.uint8)
        return img_out

    # ==========================================================

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def _calculate_cut_position(self, fx_val, fy_val, width, height):
        dfx = 1.0 / (width * self.pix)
        dfy = 1.0 / (height * self.pix)
        dx = int(round(fx_val / dfx))
        dy = int(round(fy_val / dfy))
        x_peak = (width // 2 + dx) % width
        y_peak = (height // 2 + dy) % height
        rect_w = width // 3
        rect_h = height // 3
        min_x = max(0, x_peak - rect_w // 2)
        min_y = max(0, y_peak - rect_h // 2)
        max_x = min(width, min_x + rect_w)
        max_y = min(height, min_y + rect_h)
        rect_w = max_x - min_x
        rect_h = max_y - min_y
        return min_x, min_y, rect_w, rect_h

    def visualize_first_sample(self, idx=0, save_dir=None):
        """
        可视化第 idx 个样本的预处理中间结果，并将图像保存到指定目录。
        与 __getitem__ 的处理逻辑完全一致，只是增加了可视化和保存步骤。
        """
        if idx >= len(self.x):
            print(f"Index {idx} out of range")
            return

        if save_dir is None:
            save_dir = self.save_dir or '.'
        os.makedirs(save_dir, exist_ok=True)

        file_name = self.x[idx]
        fx_val = self.fx_ref[idx]
        fy_val = self.fy_ref[idx]

        # 1. 加载图像
        img = load_img(f"{self.dir}/{file_name}", color_mode=self.color_mode)
        img_array = img_to_array(img).astype(np.float32) / 255.0
        if img_array.shape[-1] == 3:
            img_array = (0.299 * img_array[..., 0] +
                         0.587 * img_array[..., 1] +
                         0.114 * img_array[..., 2])[..., np.newaxis]
        gray = img_array.squeeze()
        orig_height, orig_width = gray.shape

        # 2. 傅里叶变换
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        # 3. 截取物光频谱
        min_x, min_y, w, h = self._calculate_cut_position(fx_val, fy_val, orig_width, orig_height)
        cropped_complex = fft_shift[min_y:min_y + h, min_x:min_x + w]

        # 4. 峰值定位和中心化
        width, height = orig_width, orig_height
        dfx = 1.0 / (width * self.pix)
        dfy = 1.0 / (height * self.pix)
        dx = int(round(fx_val / dfx))
        dy = int(round(fy_val / dfy))
        x_peak = (width // 2 + dx) % width
        y_peak = (height // 2 + dy) % height
        peak_in_crop_x = x_peak - min_x
        peak_in_crop_y = y_peak - min_y
        crop_h, crop_w = cropped_complex.shape
        shift_y = crop_h // 2 - peak_in_crop_y
        shift_x = crop_w // 2 - peak_in_crop_x
        centered = np.roll(cropped_complex, shift=(shift_y, shift_x), axis=(0, 1))

        # 5. 粘贴到全尺寸频谱
        full_spectrum = np.zeros((orig_height, orig_width), dtype=np.complex64)
        start_y = (orig_height - crop_h) // 2
        start_x = (orig_width - crop_w) // 2
        full_spectrum[start_y:start_y + crop_h, start_x:start_x + crop_w] = centered

        # 6. 逆傅里叶变换重建
        recon = np.fft.ifft2(np.fft.ifftshift(full_spectrum))
        recon_amp = np.abs(recon)

        # 7. resize
        if self.output_size != (orig_height, orig_width):
            recon_amp_pil     = Image.fromarray((recon_amp / recon_amp.max() * 255).astype(np.uint8))
            recon_amp_resized = recon_amp_pil.resize(self.output_size, Image.BILINEAR)
            recon_amp_final   = np.array(recon_amp_resized).astype(np.float32) / 255.0
        else:
            recon_amp_final = recon_amp / recon_amp.max()

        # ========== 可视化额外步骤 ==========
        magnitude     = np.abs(fft_shift)
        log_magnitude = np.log1p(magnitude)
        centered_mag  = np.abs(centered)
        centered_mag_norm = ((centered_mag - centered_mag.min()) /
                             (centered_mag.max() - centered_mag.min() + 1e-8))
        recon_norm    = ((recon_amp - recon_amp.min()) /
                         (recon_amp.max() - recon_amp.min() + 1e-8))

        base_name = os.path.splitext(os.path.basename(file_name))[0]

        gray_uint8 = (gray * 255).astype(np.uint8)
        Image.fromarray(gray_uint8).save(
            os.path.join(save_dir, f'01_gray_{base_name}_{idx}.png'))

        log_min, log_max = log_magnitude.min(), log_magnitude.max()
        log_norm = (((log_magnitude - log_min) / (log_max - log_min) * 255).astype(np.uint8)
                    if log_max > log_min else np.zeros_like(log_magnitude, dtype=np.uint8))
        Image.fromarray(log_norm).save(
            os.path.join(save_dir, f'02_log_magnitude_{base_name}_{idx}.png'))

        cropped_mag      = np.abs(cropped_complex)
        cropped_mag_norm = ((cropped_mag - cropped_mag.min()) /
                            (cropped_mag.max() - cropped_mag.min() + 1e-8))
        Image.fromarray((cropped_mag_norm * 255).astype(np.uint8)).save(
            os.path.join(save_dir, f'03_cropped_mag_{base_name}_{idx}.png'))

        Image.fromarray((centered_mag_norm * 255).astype(np.uint8)).save(
            os.path.join(save_dir, f'04_centered_mag_{base_name}_{idx}.png'))

        full_spectrum_mag  = np.abs(full_spectrum)
        full_spectrum_norm = ((full_spectrum_mag - full_spectrum_mag.min()) /
                              (full_spectrum_mag.max() - full_spectrum_mag.min() + 1e-8))
        Image.fromarray((full_spectrum_norm * 255).astype(np.uint8)).save(
            os.path.join(save_dir, f'05_full_spectrum_{base_name}_{idx}.png'))

        Image.fromarray((recon_norm * 255).astype(np.uint8)).save(
            os.path.join(save_dir, f'06_reconstructed_original_{base_name}_{idx}.png'))

        recon_final_uint8 = (recon_amp_final * 255).astype(np.uint8)
        Image.fromarray(recon_final_uint8).save(
            os.path.join(save_dir, f'07_reconstructed_resized_{base_name}_{idx}.png'))

        # -------------------- 可视化散斑增强效果（如已启用）--------------------
        if self.speckle_aug:
            aug_uint8 = self._speckle_augmentation(recon_final_uint8.copy())
            Image.fromarray(aug_uint8).save(
                os.path.join(save_dir, f'08_speckle_augmented_{base_name}_{idx}.png'))

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 4, 1);  plt.imshow(gray, cmap='gray');            plt.title(f'01. Original hologram\n{gray.shape}');           plt.colorbar()
        plt.subplot(2, 4, 2);  plt.imshow(log_magnitude, cmap='gray');   plt.title(f'02. Log magnitude spectrum\n{log_magnitude.shape}'); plt.colorbar()
        plt.subplot(2, 4, 3);  plt.imshow(cropped_mag_norm, cmap='gray');plt.title(f'03. Cropped spectrum\n{crop_h}x{crop_w}');         plt.colorbar()
        plt.subplot(2, 4, 4);  plt.imshow(centered_mag_norm, cmap='gray');plt.title(f'04. Centered spectrum\n{crop_h}x{crop_w}');       plt.colorbar()
        plt.subplot(2, 4, 5);  plt.imshow(full_spectrum_norm, cmap='gray');plt.title(f'05. Full spectrum (padded)\n{orig_height}x{orig_width}'); plt.colorbar()
        plt.subplot(2, 4, 6);  plt.imshow(recon_norm, cmap='gray');      plt.title(f'06. Reconstructed (original)\n{recon_amp.shape}'); plt.colorbar()
        plt.subplot(2, 4, 7);  plt.imshow(recon_amp_final, cmap='gray'); plt.title(f'07. Reconstructed (resized)\n{self.output_size}'); plt.colorbar()
        if self.speckle_aug:
            plt.subplot(2, 4, 8)
            plt.imshow(aug_uint8, cmap='gray')
            plt.title(f'08. Speckle augmented\ncontrast={self.speckle_contrast}, size={self.speckle_size}')
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end   = min((idx + 1) * self.batch_size, len(self.x))
        batch_indices = list(range(start, end))

        batch_files  = [self.x[i] for i in batch_indices]
        batch_labels = [self.y[i] for i in batch_indices]
        batch_fx     = self.fx_ref[batch_indices]
        batch_fy     = self.fy_ref[batch_indices]

        images = []
        for file_name, fx_val, fy_val in zip(batch_files, batch_fx, batch_fy):
            # 1. 加载图像（原始尺寸，不 resize）
            img = load_img(f"{self.dir}/{file_name}", color_mode=self.color_mode)
            img_array = img_to_array(img).astype(np.float32) / 255.0
            if img_array.shape[-1] == 3:
                img_array = (0.299 * img_array[..., 0] +
                             0.587 * img_array[..., 1] +
                             0.114 * img_array[..., 2])[..., np.newaxis]
            gray = img_array.squeeze()
            orig_height, orig_width = gray.shape

            # 2. 傅里叶变换
            fft       = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)

            # 3. 截取物光频谱
            min_x, min_y, w, h = self._calculate_cut_position(
                fx_val, fy_val, orig_width, orig_height)
            cropped_complex = fft_shift[min_y:min_y + h, min_x:min_x + w]

            # 4. 峰值定位和中心化
            width, height = orig_width, orig_height
            dfx = 1.0 / (width * self.pix)
            dfy = 1.0 / (height * self.pix)
            dx      = int(round(fx_val / dfx))
            dy      = int(round(fy_val / dfy))
            x_peak  = (width  // 2 + dx) % width
            y_peak  = (height // 2 + dy) % height
            peak_in_crop_x = x_peak - min_x
            peak_in_crop_y = y_peak - min_y
            crop_h, crop_w = cropped_complex.shape
            shift_y  = crop_h // 2 - peak_in_crop_y
            shift_x  = crop_w // 2 - peak_in_crop_x
            centered = np.roll(cropped_complex, shift=(shift_y, shift_x), axis=(0, 1))

            # 5. 粘贴到全尺寸频谱
            full_spectrum = np.zeros((orig_height, orig_width), dtype=np.complex64)
            start_y = (orig_height - crop_h) // 2
            start_x = (orig_width  - crop_w) // 2
            full_spectrum[start_y:start_y + crop_h, start_x:start_x + crop_w] = centered

            # 6. 逆傅里叶变换重建
            recon     = np.fft.ifft2(np.fft.ifftshift(full_spectrum))
            recon_amp = np.abs(recon)

            # 7. 空间域 resize → uint8（增强在 uint8 域操作，语义一致）
            if self.output_size != (orig_height, orig_width):
                recon_amp_pil     = Image.fromarray(
                    (recon_amp / recon_amp.max() * 255).astype(np.uint8))
                recon_amp_resized = recon_amp_pil.resize(self.output_size, Image.BILINEAR)
                recon_amp_uint8   = np.array(recon_amp_resized)          # uint8, H×W
            else:
                recon_amp_uint8 = (recon_amp / recon_amp.max() * 255).astype(np.uint8)

            # -------------------- 8. 散斑数据增强（仅训练期间建议启用）--------------------
            if self.speckle_aug:
                recon_amp_uint8 = self._speckle_augmentation(recon_amp_uint8)
            # ---------------------------------------------------------------------------

            recon_amp_final = recon_amp_uint8.astype(np.float32)   # 保持原始值域 [0,255]
            images.append(recon_amp_final[..., np.newaxis])

        batch_x = np.array(images)
        batch_y = np.array(batch_labels)
        return batch_x, batch_y

class RegressionSequence_exp(tf.keras.utils.Sequence):
    def __init__(self, x, y, cut_size, d_path, b_size, color_mode='grayscale',
                 target_size=(256, 256), output_size=(256, 256),
                 save_dir=r'F:\dongjiayao\Pycharm\DEP-AF\FocusNet\tmp',
                 num_workers=4,
                 device='cuda'):
        self.x = x
        self.y = y
        self.batch_size = b_size
        self.dir = d_path
        self.color_mode = color_mode
        self.target_height, self.target_width = target_size
        self.output_size = output_size   # (W, H) for cv2.resize
        self.save_dir = save_dir
        self.num_workers = num_workers
        # self.device = torch.device(device)

        if len(cut_size) == 4:
            self.cut_size = list(cut_size)
        else:
            raise ValueError("cut_size 必须为长度为 4 的列表: [min_x, min_y, rect_w, rect_h]")

        # ---------- 缓存 ----------
        self._cached_delta  = None   # (delta_x, delta_y)，整个训练只算一次
        self._gpu_mask      = None   # torch bool tensor on GPU
        self._gpu_mask_shape = None  # (H, W)

    # 新增一个属性，每次调用时动态获取当前 CUDA device
    @property
    def _current_device(self):
        return torch.device(f'cuda:{torch.cuda.current_device()}')

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # ------------------------------------------------------------------
    # 读图（CPU，cv2）
    # ------------------------------------------------------------------
    def _load_gray_cv2(self, file_name):
        path = os.path.join(self.dir, file_name)
        if self.color_mode == 'grayscale':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法读取: {path}")
            return img.astype(np.float32) / 255.0
            # return img.astype(np.float32)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"无法读取: {path}")
            img = img.astype(np.float32) / 255.0
            return (0.299 * img[..., 0] +
                    0.587 * img[..., 1] +
                    0.114 * img[..., 2])

    # ------------------------------------------------------------------
    # GPU 掩膜缓存
    # ------------------------------------------------------------------
    # _get_gpu_mask：每次检查 mask 是否在正确设备上
    def _get_gpu_mask(self, h, w):
        dev = self._current_device
        # 形状或设备任一不符都重建
        if (self._gpu_mask is not None
                and self._gpu_mask_shape == (h, w)
                and self._gpu_mask.device == dev):
            return self._gpu_mask
        min_x, min_y, rect_w, rect_h = self.cut_size
        mask_cpu = np.zeros((h, w), dtype=bool)
        mask_cpu[min_y:min_y + rect_h, min_x:min_x + rect_w] = True
        self._gpu_mask = torch.from_numpy(mask_cpu).to(dev)
        self._gpu_mask_shape = (h, w)
        return self._gpu_mask

    # ------------------------------------------------------------------
    # Delta 计算（CPU + OpenCV，整个训练只跑一次）
    # ------------------------------------------------------------------
    def _compute_delta_cpu(self, gray_cpu):
        min_x, min_y, rect_w, rect_h = self.cut_size
        fft_shift_cpu = np.fft.fftshift(np.fft.fft2(gray_cpu))

        log_mag = np.log1p(np.abs(fft_shift_cpu))
        log_mag_uint8 = (
            (log_mag - log_mag.min()) /
            (log_mag.max() - log_mag.min() + 1e-8) * 255
        ).astype(np.uint8)

        fft_h, fft_w = log_mag_uint8.shape
        fft_img = log_mag_uint8.copy()
        mask = np.zeros_like(fft_img)
        mask[min_y:min_y + rect_h, min_x:min_x + rect_w] = 1
        fft_img = fft_img * mask

        _, binary = cv2.threshold(fft_img, 180, 255, cv2.THRESH_BINARY)
        binary = cv2.GaussianBlur(binary, (1, 1), 50)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0, 0, log_mag_uint8

        bboxes = [cv2.boundingRect(c) for c in contours]
        x_c, y_c, w_c, h_c = max(bboxes, key=lambda b: b[2])
        delta_x = int(0.5 * w_c + x_c - 0.5 * fft_w)
        delta_y = int(0.5 * h_c + y_c - 0.5 * fft_h)
        return delta_x, delta_y, log_mag_uint8

    # ------------------------------------------------------------------
    # 核心：torch FFT 全程在 GPU
    # ------------------------------------------------------------------
    # _process_one_image：所有 GPU tensor 都用 dev 变量，不用 self.device
    def _process_one_image(self, file_name, return_intermediates=False):
        dev = self._current_device  # ← 动态获取，与当前 worker 一致

        gray_cpu = self._load_gray_cv2(file_name)
        orig_h, orig_w = gray_cpu.shape

        if self._cached_delta is None:
            dx, dy, log_mag_uint8 = self._compute_delta_cpu(gray_cpu)
            self._cached_delta = (dx, dy)
        else:
            dx, dy = self._cached_delta
            log_mag_uint8 = None

        gray_gpu = torch.from_numpy(gray_cpu).to(dev)
        fft_gpu = torch.fft.fftshift(torch.fft.fft2(gray_gpu))

        mask = self._get_gpu_mask(orig_h, orig_w)  # 已保证与 dev 一致
        # zeros 也用 dev，不再硬编码
        u0 = torch.where(mask, fft_gpu,
                         torch.zeros(1, dtype=torch.complex64, device=dev))

        u0 = torch.roll(u0, shifts=-dx, dims=1)
        u0 = torch.roll(u0, shifts=-dy, dims=0)

        recon = torch.fft.ifft2(torch.fft.ifftshift(u0))
        amp_gpu = torch.abs(recon)
        amp_cpu = amp_gpu.cpu().numpy()

        # amp_norm = amp_cpu
        amp_max = amp_cpu.max()
        amp_norm = amp_cpu / (amp_max + 1e-8)

        out_w, out_h = self.output_size
        if (out_h, out_w) != (orig_h, orig_w):
            amp_final_0 = cv2.resize(amp_norm, (out_w, out_h), interpolation=cv2.INTER_LINEAR) * 255
            # resize 后重新归一化，确保最大值严格为 1.0
            amp_max_resized = amp_final_0
            # if amp_max_resized. > 1e-8:
            amp_final = amp_final_0 / amp_max_resized.max() * 255
            # amp_final = amp_max_resized
        else:
            amp_final = amp_norm * 255

        if return_intermediates:
            return amp_final, fft_gpu.cpu().numpy(), log_mag_uint8, u0.cpu().numpy(), amp_cpu, dx, dy
        return amp_final, None, None, None, None, dx, dy

    # ------------------------------------------------------------------
    # __getitem__：多线程读图 + 逐张 GPU 处理
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # if idx == 0:
        #     self.visualize_first_sample()

        start = idx * self.batch_size
        end   = min((idx + 1) * self.batch_size, len(self.x))
        batch_files  = self.x[start:end]
        batch_labels = self.y[start:end]

        # 多线程并行读图，消除磁盘 IO 等待
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            grays = list(ex.map(self._load_gray_cv2, batch_files))

        # 逐张送 GPU 处理（串行避免显存溢出）
        images = []
        for fn in batch_files:
            amp_final, *_ = self._process_one_image(fn)
            images.append(amp_final[..., np.newaxis])

        batch_x = np.array(images)
        batch_y = np.array(batch_labels)

        return batch_x, batch_y

    # ------------------------------------------------------------------
    # 可视化（调试用）
    # ------------------------------------------------------------------
    def visualize_first_sample(self, idx=0, save_dir=None):
        if idx >= len(self.x):
            print(f"Index {idx} out of range")
            return
        if save_dir is None:
            save_dir = self.save_dir or '.'
        os.makedirs(save_dir, exist_ok=True)
        self._cached_delta = None
        file_name = self.x[idx]
        min_x, min_y, rect_w, rect_h = self.cut_size

        (amp_final, fft_cpu, log_mag_uint8,
         u0_cpu, amp_cpu, dx, dy) = self._process_one_image(
            file_name, return_intermediates=True
        )
        gray = self._load_gray_cv2(file_name)

        def norm_u8(arr):
            arr = np.abs(arr).astype(np.float32)
            mn, mx = arr.min(), arr.max()
            return ((arr - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)

        log_vis = np.log1p(np.abs(fft_cpu))
        base = os.path.splitext(os.path.basename(file_name))[0]

        # ---- 保存各步骤单张图 ----
        for fname, arr in [
            (f'01_gray_{base}_{idx}.png', (gray * 255).astype(np.uint8)),
            (f'02_log_magnitude_{base}_{idx}.png', norm_u8(log_vis)),
            (f'03_u0_masked_rolled_{base}_{idx}.png', norm_u8(np.abs(u0_cpu))),
            (f'04_reconstructed_{base}_{idx}.png', norm_u8(amp_cpu)),
            (f'05_resized_{base}_{idx}.png', (amp_final * 255).astype(np.uint8)),
        ]:
            Image.fromarray(arr).save(os.path.join(save_dir, fname))

        print(f"[Visualize] delta=({dx},{dy})  cut_size={self.cut_size}")

        # ---- 绘制拼图并保存 ----
        imgs = [gray, log_vis, np.abs(u0_cpu), norm_u8(amp_cpu), amp_final]
        titles = [
            f'01 Original\n{gray.shape}',
            f'02 Log magnitude\ndelta=({dx},{dy})',
            '03 u0 masked+rolled',
            '04 Reconstructed',
            f'05 Resized {self.output_size}',
        ]

        fig = plt.figure(figsize=(20, 8))
        for i, (img, title) in enumerate(zip(imgs, titles), 1):
            ax = fig.add_subplot(2, 3, i)
            im = ax.imshow(img, cmap='gray')
            ax.set_title(title)
            if i == 2:
                ax.add_patch(patches.Rectangle(
                    (min_x, min_y), rect_w, rect_h,
                    linewidth=2, edgecolor='red', facecolor='none'))
            plt.colorbar(im, ax=ax)

        plt.tight_layout()

        # 保存拼图
        overview_path = os.path.join(save_dir, f'00_overview_{base}_{idx}.png')
        fig.savefig(overview_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] 拼图已保存: {overview_path}")

        plt.show()
        plt.close(fig)  # 释放内存

class UnalRegressionSequence(RegressionSequence):
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            np.array(load_img(f"{self.dir}/{file_name}", color_mode=self.color_mode))
            for file_name in batch_x]), np.array(batch_y)


# Perform holors augmentation on holograms (Only 90°, 180° and 270° rotations)
class Rotate90Randomly(tf.keras.layers.Layer):

    @staticmethod
    def call(x, training=False):
        def random_rotate():
            rotation_factor = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
            return tf.image.rot90(x, k=rotation_factor)

        training = tf.constant(training, dtype=tf.bool)

        rotated = tf.cond(training, random_rotate, lambda: x)
        rotated.set_shape(rotated.shape)
        return rotated


# Add Fourier transform to the tensor
class Fourier2D(tf.keras.layers.Layer):
    def __init__(self, *args, sl=slice(0, 1), **kwargs):
        super().__init__(*args, **kwargs)
        self.slice = sl

    def call(self, x: tf.Tensor):
        # def fourier(hologram):  # ToDo: Test with log only
            # return tf.concat([
            #     tf.math.real(hologram[:, :, self.slice]),
            #     tf.expand_dims(tf.math.log(tf.math.square(
            #         tf.abs(tf.signal.fftshift(tf.signal.fft2d(hologram[:, :, 0])))
            #     )), -1)],
            #     axis=-1
            # )  # Use the hologram and the log abs fourier transform

        def fourier(hologram):  # There are nan problems with log(0)
            return tf.concat([
                tf.math.real(hologram[:, :, self.slice]),
                # tf.math.real(hologram),
                tf.expand_dims(
                    tf.abs(tf.signal.fftshift(tf.signal.fft2d(hologram[:, :, 0]))), -1)],
                axis=-1
            )  # Use the hologram and the log abs fourier transform

        return tf.vectorized_map(fourier, x)  # Performs tf ops in max parallelism


class Scheduler:
    def __init__(self, changing_period, sprint_epoch, decrease_ratio=2, decay=0.1):
        self.changing_period = changing_period
        self.sprint_epoch = sprint_epoch
        self.decrease_ratio = decrease_ratio
        self.decay = decay

    def schedule(self, epoch, lr):
        if epoch % self.changing_period == 0 and 0 < epoch < self.sprint_epoch:
            return lr / self.decrease_ratio
        elif epoch < self.sprint_epoch:
            return lr
        else:
            return lr * tf.math.exp(-self.decay)


# Custom activation function to limit regression output possibilities
def holo(d, target_min=0.1, target_max=10):
    d = tanh(d) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.
    return d * scale + target_min
