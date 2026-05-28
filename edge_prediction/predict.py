#!/user/bin/python
# coding=utf-8
import os
import re
import time

import numpy as np
from PIL import Image
import cv2
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew

from edge_prediction.model.RCF import RCF

matplotlib.use('Agg')

from edge_prediction.dataloader.data_loader import BSDS_RCFLoader


# 计时器
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)  # 执行原函数并保留返回值
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result  # 返回原函数的执行结果
    return wrapper

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int,
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default=r'F:\dongjiayao\Pycharm\Holo-Track\rcf\tmp\113328\checkpoint_epoch4.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset',
                    default=r'F:\dongjiayao\Data\AutoFocusDatabase\MultyDistance_v1\Image__2024-04-26__20-23-57.bmp')

# ================ save
parser.add_argument('--save', default=r'F:\dongjiayao\Data\holograms\Focus\edge', type=str,)

# ================ scale
parser.add_argument('--scale', default=8,)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    args.cuda = True
    # dataset
    predict_dataset = BSDS_RCFLoader(root=args.dataset, split="predict", k_size=args.scale)
    predict_loader = DataLoader(
        predict_dataset, batch_size=args.batch_size,
        num_workers=1, drop_last=True, shuffle=True)

    # model
    model = RCF()
    model.cuda()

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'"
          .format(args.resume))

    rcf_predict(model, predict_loader, save_dir=args.save)


def rcf_predict(model, predict_loader, save_dir=''):
    model.eval()

    if save_dir:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # 存储文件名和对应的熵值
    names = []
    concentrations = []
    max_concentration = -1
    best_focus_image = None
    best_focus_name = None

    for idx, (image, name) in enumerate(predict_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        # 边缘检测图像保存
        # result_img.save(os.path.join(save_dir, name[0]))

        # 计算 concentration 和 non_zero_values
        concentration, non_zero_values = calculate_brightness_concentration(
            np.array(result_img), tensor=results[-1], return_non_zero=True
        )

        # 保存分布图
        if save_dir:
            plot_path = os.path.join(save_dir, f"{os.path.splitext(name[0])[0]}_distribution.png")
            plt.figure(figsize=(8, 6))

            # 获取图像总像素数量
            total_pixels = result_img.size[0] * result_img.size[1]

            # 创建横坐标 0-255
            bins = np.arange(0, 256)

            # 绘制直方图，纵坐标为每个像素值的数量
            # 使用 density=False 确保计数而不是密度
            counts, bins, patches = plt.hist(non_zero_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')

            # 设置横坐标固定为0-255
            plt.xlim(0, 255)

            # 设置纵坐标固定为图像总像素数量
            plt.ylim(0, 15000)

            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(f'Non-zero Pixel Distribution: {name[0]}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Distribution plot saved to {plot_path}")

        concentrations.append(concentration)
        names.append(name[0].split('_')[0])  # 提取文件名 todo

        # 更新最大 concentration 的图像
        if concentration > max_concentration:
            max_concentration = concentration
            best_focus_image = image
            best_focus_name = name

        print("Running test [%d/%d]" % (idx + 1, len(predict_loader)))
        print(name, concentration)

    # 按文件名中的编号排序
    # 假设文件名格式为 image_{number}.jpg
    # 使用正则表达式提取编号
    sorted_indices = sorted(range(len(names)), key=lambda x: int(re.findall(r'\d+', names[x])[0]))
    sorted_names = [names[i] for i in sorted_indices]
    sorted_concentrations = [concentrations[i] for i in sorted_indices]

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_names, sorted_concentrations, marker='.', linestyle='-', color='b')
    plt.xlabel('Image Name (Sorted by Number)')
    plt.ylabel('Concentration')
    plt.title('Concentration of Predicted Images')
    plt.xticks(rotation=45)  # 旋转横坐标标签
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局

    if save_dir:
        # 保存折线图
        plot_path = os.path.join(save_dir, 'concentration_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Concentration plot saved to {plot_path}")

    return best_focus_image, best_focus_name


# @timer
def calculate_brightness_concentration(
    image,
    tensor=None,
    siamese_model=None,
    return_non_zero=False,
    return_all_metrics=False
):
    # =========================
    # 输入处理
    # =========================
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)

    # 转灰度
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    # =========================
    # 基础统计
    # =========================
    # total_brightness = image.sum()
    nonzero_pixels = np.count_nonzero(image)

    if nonzero_pixels == 0:
        if return_non_zero:
            return 0, np.array([])
        return 0

    # concentration = np.log(total_brightness / nonzero_pixels)

    non_zero_values = image[image != 0]

    # =========================
    # 已有指标
    # =========================
    # variance = np.var(image)
    variance = np.var(non_zero_values, ddof=1)
    # entropy_val = shannon_entropy(non_zero_values)

    # =========================
    # 新增指标
    # =========================

    # ---- 1. Gini 系数 ----
    # 问题：标准公式对排序敏感
    # 解决：使用稳定实现（避免数值问题）
    # sorted_vals = np.sort(non_zero_values)
    # n = len(sorted_vals)
    # index = np.arange(1, n + 1)

    # 防止除0
    # if sorted_vals.sum() == 0:
    #     gini = 0
    # else:
    #     gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

    # ---- 2. 峰度（Kurtosis）----
    # 问题：默认是Pearson，需要转为Fisher
    # 解决：fisher=True（正态=0）
    # kurt = kurtosis(non_zero_values, fisher=True, bias=False)

    # ---- 3. Fisher偏度（Skewness）----
    # 问题：分布非对称性
    # skewness_val = skew(non_zero_values, bias=False)

    # =========================
    # 输出控制
    # =========================
    # if return_all_metrics:
    #     return {
    #         "concentration": concentration,
    #         "variance": variance,
    #         "entropy": entropy_val,
    #         "gini": gini,
    #         "kurtosis": -kurt,
    #         "skewness": skewness_val
    #     }

    if return_non_zero:
        return variance, non_zero_values

    return variance


if __name__ == '__main__':
    main()
