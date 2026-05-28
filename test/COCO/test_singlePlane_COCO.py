# #!/usr/bin/env python
# # coding=utf-8
# import os
# import argparse
#
# # ── 必须在所有 torch/cuda import 之前设置 ──────────────────────
# _pre_parser = argparse.ArgumentParser(add_help=False)
# _pre_parser.add_argument('--gpu', default='0', type=str)   # ← 改这里指定 GPU
# _pre_args, _ = _pre_parser.parse_known_args()
# os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu
# # ────────────────────────────────────────────────────────────────

import csv
import math
import re
import time
import random
import cv2
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
import os

import torch.nn.functional as F
from tqdm import tqdm

from edge_prediction.model.RCF import RCF
from peak_finder.peakfinder import PeakFinder

from edge_prediction.predict import calculate_brightness_concentration
from edge_prediction.dataloader.data_loader import convert_to_rgb, prepare_image_PIL, prepare_image_cv2, prepare_image_torch_gpu

from propagate.angular_torch import AngularBatch

def set_deterministic():
    """设置确定性运行环境"""
    import random
    import numpy as np
    import torch

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"设置确定性环境，随机种子: {seed}")

# 在main函数开头调用
set_deterministic()

# 计时器
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)  # 执行原函数并保留返回值
        end = time.time()
        print(f"{func.__name__} 执行耗时: {end - start:.4f} 秒")
        return result  # 返回原函数的执行结果
    return wrapper


# @timer
# 初始化函数，重建第一张图，并获取频谱滤波后全息图
def initial(root, fx_ref=None, fy_ref=None):
    hologram = Image.open(root).convert('L')
    filtered = AngularBatch(lam=532e-9,
                            pix=1e-6,
                            z_initial=0.00030,
                            input_img=hologram,
                            inline=False,
                            get_filtered=True,
                            fx_ref=fx_ref,
                            fy_ref=fy_ref
                            ).start()  # filtered处于空间域，获取频谱滤波结果，位于torch_gpu上

    return filtered


class focus_and_z():
    def __init__(self, filtered_tensor, model):
        self.filtered_tensor = filtered_tensor
        self.model = model
        self.device = next(model.parameters()).device

    # 定义聚焦分数和z的函数，输入z，输出score
    # @timer
    def func(self, z, **kwargs):
        in_brent = kwargs['in_brent']
        propped = AngularBatch(lam=532e-9,
                               pix=1e-6,
                               z_initial=z,
                               input_img=self.filtered_tensor,
                               inline=True,
                               get_filtered=False
                               ).start()  # filtered处于空间域，获取频谱滤波结果，位于torch_gpu上

        # GPU版本处理
        propped = torch.abs(propped)  # 直接取绝对值，无需转 CPU
        # 归一化到 [0, 255] 并保持 GPU 张量
        propped = (propped / propped.max() * 255).to(torch.float32)
        # 单通道扩展为三通道 (替代 np.stack)
        propped_3d = propped.unsqueeze(2).repeat(1, 1, 3)  # (H, W, 3)

        score = focus_value(self.model, propped_3d, scale=1)  # 聚焦分数初始化

        if "get_img" in kwargs:
            propped_np = torch.abs(propped).cpu().numpy()
            propped_np = (propped_np / propped_np.max() * 255).astype(np.uint8)
            propped_pil = Image.fromarray(propped_np, mode="L").convert('RGB')
            return propped_pil
        else:
            return score


# @timer
def safe_crop(image, left, top, right, bottom):
    """
    安全裁剪图像，确保裁剪区域不会超出图像边界。

    :param image: PIL.Image 对象
    :param left: 裁剪区域的左边界
    :param top: 裁剪区域的上边界
    :param right: 裁剪区域的右边界
    :param bottom: 裁剪区域的下边界
    :return: 裁剪后的图像
    """
    # 获取图像的宽度和高度
    width, height = image.size

    # 调整裁剪区域，确保不超出图像边界
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    # 如果裁剪区域无效（宽度或高度为0），返回原始图像
    if right <= left or bottom <= top:
        return image

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


# @timer
def focus_value(model, image, scale=8):
    model.eval()

    # GPU版本处理
    # 下采样（替代 OpenCV 的 resize）f
    h, w = image.shape[0], image.shape[1]
    new_h, new_w = h // scale, w // scale
    # todo 测试固定分辨率
    # new_h, new_w = 256, 256
    image = F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0),  # 转为 (1, C, H, W)
        size=(new_h, new_w),
        mode='nearest'
    ).squeeze(0).permute(1, 2, 0)  # 恢复为 (H', W', C)

    # GPU 预处理 (替代 prepare_image_PIL)
    image = prepare_image_torch_gpu(image)

    image = image.unsqueeze(0)

    image = image.cuda()
    _, _, H, W = image.shape
    results = model(image)

    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    result = Image.fromarray((result * 255).astype(np.uint8))
    # 边缘检测图像保存
    # result.save(r'F:\dongjiayao\Data\COCO\val\edge\edge_{:.5f}.jpg'.format(time.time()))

    concentration = calculate_brightness_concentration(np.array(result), tensor=results[-1])

    return concentration


# @timer
def main(root, save_dir, lb, ub, model=None, fx_ref=None, fy_ref=None):
    # 模型加载逻辑调整
    filtered = initial(root=root, fx_ref=fx_ref, fy_ref=fy_ref)
    focus_z = focus_and_z(filtered, model)     # todo get_score_directly表示是否用模型直接得到聚焦分数

    def max_func(z, scale=None):
        # in_brent很重要，防止在优化时在小区间重复迭代，导致误识别颗粒
        score = focus_z.func(z, in_brent=True, scale=scale)
        return -score.item() if isinstance(score, torch.Tensor) else -score

    finder = PeakFinder(max_func, lb=lb, ub=ub, max_evals=200, target='min', precision_factor=1, initial_points_factor=5)
    z_result = finder.find_optimum()

    # img_result = focus_z.func(z_result, in_brent=True, get_img=True)
    # save_path = os.path.join(save_dir, os.path.basename(root))
    # img_result.save(save_path)      # todo 不保存了，省一点时间

    return z_result


def batch_main(root_dir, save_dir, test_csv=None):
    """
    批量处理目录中的全息图，并可选择进行测试

    :param root_dir: 包含全息图的目录
    :param save_dir: 保存重建结果的目录
    :param test_csv: 可选，包含真实z值的CSV文件路径
    """
    lb = 0.00005          # todo
    ub = 0.00085

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f)) and
           os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ]
    image_files = natsorted(image_files)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 如果有提供测试CSV文件，读取真实值
    ground_truth = {}
    fx = {}
    fy = {}
    if test_csv and os.path.exists(test_csv):
        print(f"加载测试数据: {test_csv}")
        with open(test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hologram_name = row['hologram_name']
                z_gt = float(row['z'])
                ground_truth[hologram_name] = z_gt

                # 尝试读取 fx_ref 和 fy_ref，若不存在则为 None
                fx_ref = float(row['fx_ref']) if 'fx_ref' in row else None
                fy_ref = float(row['fy_ref']) if 'fy_ref' in row else None
                fx[hologram_name] = fx_ref
                fy[hologram_name] = fy_ref

    # 初始化结果存储
    results = []

    # 创建结果CSV文件并立即写入表头
    results_csv = os.path.join(save_dir, "focusing_results.csv")
    fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error']

    # 创建CSV文件并写入表头
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # 模型加载
    model = RCF()
    model.to(device)
    checkpoint = torch.load(r'F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing\edge_prediction\tmp\rcf\speckle_aug_true/checkpoint_epoch10.pth',
                            weights_only=True, )
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # 创建进度条
    pbar = tqdm(image_files, desc="Processing holograms", unit="image")

    # 初始化误差统计变量
    total_abs_error = 0.0
    total_rel_error = 0.0
    valid_rel_count = 0
    processed_count = 0

    for image_file in pbar:
        image_path = os.path.join(root_dir, image_file)
        pbar.set_description(f"Processing: {image_file}")

        # 如果csv文件中存在离轴信息
        fx_ref = None
        fy_ref = None
        if image_file in fx:
            fx_ref = fx[image_file]
            fy_ref = fy[image_file]

        # 处理图像并获取预测的z值
        z_pred = main(lb=lb, ub=ub, root=image_path, save_dir=save_dir, model=model, fx_ref=fx_ref, fy_ref=fy_ref)

        # 准备结果行
        result_row = {
            'hologram_name': image_file,
            'z_gt': '',
            'z_pred': z_pred,
            'abs_error': '',
            'rel_error': ''
        }

        # 如果有真实值，记录误差
        if image_file in ground_truth:
            z_gt = ground_truth[image_file]

            abs_error = abs(z_pred - z_gt)
            rel_error = abs_error / abs(ub - lb) if abs(ub - lb) > 1e-9 else float('inf')

            # 更新累计误差
            processed_count += 1
            total_abs_error += abs_error
            if not math.isinf(rel_error):
                total_rel_error += rel_error
                valid_rel_count += 1

            # 计算实时平均误差
            avg_abs_error = total_abs_error / processed_count
            avg_rel_error = (total_rel_error / valid_rel_count) * 100 if valid_rel_count > 0 else float('nan')

            # 更新结果行
            result_row.update({
                'z_gt': z_gt,
                'abs_error': abs_error,
                'rel_error': rel_error
            })

            # 存储结果到内存列表
            results.append(result_row.copy())

            # 更新进度条显示
            pbar.set_postfix({
                'z_gt': f"{z_gt:.7f}",
                'z_pred': f"{z_pred:.7f}",
                'abs_err': f"{abs_error:.7f}",
                'rel_err(%)': f"{rel_error * 100:.2f}" if not math.isinf(rel_error) else "Inf",
                'avg_abs': f"{avg_abs_error:.7f}",
                'avg_rel(%)': f"{avg_rel_error:.2f}" if not math.isnan(avg_rel_error) else "NaN"
            })
        else:
            # 没有真实值的情况
            results.append(result_row.copy())
            pbar.set_postfix({
                'z_pred': f"{z_pred:.7f}"
            })

        # 关键修改：每处理一个图像就立即写入CSV
        with open(results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result_row)

    # 如果进行了测试，输出统计结果
    if results:
        print("\n" + "=" * 50)
        print("聚焦距离预测误差统计")
        print("=" * 50)

        # 计算统计指标
        abs_errors = [r['abs_error'] for r in results]
        rel_errors = [r['rel_error'] for r in results if not math.isinf(r['rel_error'])]

        avg_abs_error = sum(abs_errors) / len(abs_errors)
        min_abs_error = min(abs_errors)
        max_abs_error = max(abs_errors)

        if rel_errors:
            avg_rel_error = sum(rel_errors) / len(rel_errors) * 100
            min_rel_error = min(rel_errors) * 100
            max_rel_error = max(rel_errors) * 100
        else:
            avg_rel_error = min_rel_error = max_rel_error = float('nan')

        # 打印统计信息
        print(f"处理图像数量: {len(results)}")
        print(f"平均绝对误差: {avg_abs_error:.7f} m")
        print(f"最小绝对误差: {min_abs_error:.7f} m")
        print(f"最大绝对误差: {max_abs_error:.7f} m")

        if not math.isnan(avg_rel_error):
            print(f"平均相对误差: {avg_rel_error:.2f}%")
            print(f"最小相对误差: {min_rel_error:.2f}%")
            print(f"最大相对误差: {max_rel_error:.2f}%")

        # 保存详细结果
        results_csv = os.path.join(save_dir, "focusing_results.csv")
        with open(results_csv, 'w', newline='') as f:
            fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n详细结果已保存至: {results_csv}")


def test_focusing_accuracy(root_dir, test_csv, save_dir):
    """
    测试聚焦算法的准确度

    :param test_csv: 包含真实z值的CSV文件路径
    :param save_dir: 保存重建结果和测试报告的目录
    """
    # 从CSV文件中提取图像目录
    with open(test_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("测试CSV文件为空")
            return

    # 运行批量处理
    print(f"开始测试聚焦算法准确度")
    print(f"测试数据集: {test_csv}")
    print(f"图像目录: {root_dir}")
    print(f"保存目录: {save_dir}")

    batch_main(root_dir, save_dir, test_csv=test_csv)


if __name__ == "__main__":
    test_focusing_accuracy(root_dir=r'F:\dongjiayao\Data\COCO\val\holograms_amp_phase\amplitude',
                           save_dir=r'F:\dongjiayao\Data\COCO\val\article\comparison\ours',
                           test_csv=r'F:\dongjiayao\Data\COCO\val\holograms_amp_phase\amplitude\AutoFocusDatabase.csv')
