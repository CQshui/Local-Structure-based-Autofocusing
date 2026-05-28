from matplotlib import pyplot as plt


def prepare_image_PIL(im):
    im = im[:, :, ::-1] - np.zeros_like(im)  # rgb to bgr
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def prepare_image_torch_gpu(im_tensor):
    # 输入 im_tensor 形状 (H, W, 3)，位于 GPU
    # BGR 转换（替代 im[:, :, ::-1]）
    im_tensor = im_tensor[:, :, [2, 1, 0]]  # RGB -> BGR

    # 减去 BGR 均值 (GPU 张量)
    mean = torch.tensor([104.00698793, 116.66876762, 122.67891434],
                       dtype=torch.float32, device=im_tensor.device)
    im_tensor = im_tensor - mean

    # 调整维度顺序为 (C, H, W)
    im_tensor = im_tensor.permute(2, 0, 1)
    return im_tensor


def prepare_image_cv2(im):
    im -= np.array((104.00698793, 116.66876762, 122.67891434))
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


def is_grayscale(img):
    # 判断图像是否为灰度图
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def convert_to_rgb(img):
    # 将灰度图转换为RGB
    if is_grayscale(img):
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
    return img


import numpy as np
import cv2
import torch
from torch.utils import data
from PIL import Image
import os
from os.path import join


def angular_spectrum(field, wavelength, pixel_size, z):
    H, W = field.shape
    k = 2.0 * np.pi / wavelength

    fx = np.fft.fftfreq(W, d=pixel_size)
    fy = np.fft.fftfreq(H, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    sqrt_term = np.sqrt(np.maximum(arg, 0.0))
    H_transfer = np.exp(1j * k * z * sqrt_term)
    H_transfer[arg < 0] = 0.0
    F = np.fft.fft2(field)
    F_prop = F * H_transfer
    return np.fft.ifft2(F_prop)


class BSDS_RCFLoader(data.Dataset):
    def __init__(self, root=r'E:\DongJiayao\Data\PASCAL', split='train', transform=False,
                 k_size=10, channel=3, resize=None, get_pil=False,
                 ):

        self.root = root
        self.split = split
        self.transform = transform
        self.k_size = k_size
        self.channel = channel
        self.resize = resize
        self.get_pil = get_pil

        if self.split == 'train':
            self.filelist = join(self.root, 'train_pair.lst')
            with open(self.filelist, 'r') as f:
                self.filelist = f.readlines()
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
            with open(self.filelist, 'r') as f:
                self.filelist = f.readlines()
        elif self.split == 'predict':
            self.namelist = os.listdir(root)
            self.filelist = self.namelist
        else:
            self.namelist = os.listdir(root)
            self.filelist = self.namelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            try:
                img_file, lb_file = self.filelist[index].split()
            except ValueError:
                print(f"Error parsing line {index}: {self.filelist[index]}")
                img_file = "aug_data/0.0_0/2010_005297.jpg"  # 替换为默认路径或跳过
                lb_file = "aug_gt/0.0_0/2010_005297.png"

            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            img0 = cv2.imread(join(self.root, img_file), 0)  # 0表示灰度模式

            if self.transform:  # 确保只有在训练且开启transform时才进行增强
                import random
                # 可以考虑使用np.random代替random模块，有时更快
                scale_factor = np.random.uniform(0.5, 2.0)  # 缩放比例范围可根据任务调整
                # 计算新的尺寸
                new_width = int(img0.shape[1] * scale_factor)
                new_height = int(img0.shape[0] * scale_factor)

                # 使用INTER_NEAREST插值缩放标签，保持边缘清晰度
                lb = cv2.resize(lb, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                img0 = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2

            if self.resize:
                lb = cv2.resize(lb, (320, 320), interpolation=cv2.INTER_NEAREST)  # 关键调整
                img0 = cv2.resize(img0, (320, 320), interpolation=cv2.INTER_LINEAR)

            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 128)] = 2
            lb[lb >= 128] = 1

            # 读取灰度图并调整尺寸
            img0 = np.array(img0, dtype=np.float32)
            img = convert_to_rgb(img0)  # 转为三通道 (224,224,3)
            img = prepare_image_cv2(img)
            if self.channel == 3:
                return img, lb

            else:
                return img0, lb[0]  # 单通道 (224,224)

        else:
            try:
                # 其他情况，调整尺寸并转换
                img = Image.open(join(self.root, self.namelist[index]))
                img = np.array(img, dtype=np.float32)
                img = convert_to_rgb(img)  # 确保RGB格式
                img = cv2.resize(img, (img.shape[1] // self.k_size, img.shape[0] // self.k_size),
                                 interpolation=cv2.INTER_NEAREST)
                img = prepare_image_PIL(img)
                return img, self.namelist[index]
            except:
                pass

    def visualize(self, num_samples=20, indices=None, figsize=(15, 10),
                  save_dir=r'F:\dongjiayao\Data\COCO\val\article\speckle_comparison\tmp', dpi=300, show=True):
        """
        可视化数据集中的样本，并可选择将每个样本保存为高分辨率图像。

        参数:
            num_samples (int): 要显示的样本数量（当 indices 为 None 时使用）
            indices (list[int]): 指定的样本索引列表，若提供则忽略 num_samples
            figsize (tuple): matplotlib 图像大小
            save_dir (str, optional): 若指定，则将每个样本的图像和标签保存到此目录（PNG格式）
            dpi (int): 保存整个图表时的分辨率（若同时显示和保存图表）
            show (bool): 是否显示 matplotlib 窗口（若为 False 且 save_dir 不为空，则只保存）
        """
        if indices is None:
            indices = list(range(min(num_samples, len(self))))
        else:
            indices = [i for i in indices if 0 <= i < len(self)]

        if len(indices) == 0:
            print("没有有效的索引可显示。")
            return

        has_label = (self.split == 'train')
        cols = 2 if has_label else 1
        rows = len(indices)

        # 创建图表（如果需要显示或保存整体图表）
        if show or (save_dir and not show):
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            # 统一 axes 为二维数组
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, cols)
            elif cols == 1:
                axes = axes.reshape(rows, 1)

        # 如果指定保存目录，则创建目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for row, idx in enumerate(indices):
            data_item = self[idx]

            if has_label:
                img, lb = data_item
            else:
                img = data_item
                lb = None

            # ----- 将图像转换为可显示的 RGB uint8 -----
            if torch.is_tensor(img):
                img_np = img.cpu().numpy().transpose(1, 2, 0)
            else:
                img_np = img.transpose(1, 2, 0)

            mean = np.array([104.00698793, 116.66876762, 122.67891434])
            img_np = img_np + mean
            img_np = img_np[:, :, ::-1]  # BGR -> RGB
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # ----- 处理标签（如果有）-----
            if has_label:
                if torch.is_tensor(lb):
                    lb_np = lb.cpu().numpy().squeeze()
                else:
                    lb_np = lb.squeeze()
                # 将标签映射为彩色图
                lb_viz = np.zeros((lb_np.shape[0], lb_np.shape[1], 3), dtype=np.uint8)
                lb_viz[lb_np == 1] = [255, 255, 255]  # 边缘为白色
                lb_viz[lb_np == 2] = [128, 128, 128]  # 不确定为灰色
                # 背景保持黑色

            # ----- 保存单个样本的高分辨率图像 -----
            if save_dir:
                # 保存图像
                img_filename = os.path.join(save_dir, f"img_{idx:04d}.png")
                Image.fromarray(img_np).save(img_filename)
                if has_label:
                    # 保存标签彩色图
                    lb_filename = os.path.join(save_dir, f"label_{idx:04d}.png")
                    Image.fromarray(lb_viz).save(lb_filename)

            # ----- 如果显示图表，则绘制到对应子图 -----
            if show or (save_dir and not show):
                axes[row, 0].imshow(img_np)
                axes[row, 0].set_title(f'Image {idx}')
                axes[row, 0].axis('off')

                if has_label:
                    axes[row, 1].imshow(lb_viz)
                    axes[row, 1].set_title(f'Label {idx}')
                    axes[row, 1].axis('off')

        # 最终显示或保存整个图表
        if show or (save_dir and not show):
            plt.tight_layout()
            if save_dir and not show:
                # 仅保存图表而不显示
                chart_path = os.path.join(save_dir, "visualization_grid.png")
                plt.savefig(chart_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
            elif show:
                plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建带有孪生像增强的数据加载器
    train_dataset = BSDS_RCFLoader(
        root=r'F:\DongJiayao\Data\PASCAL',
        split='train',
    )

    train_dataset.visualize()
