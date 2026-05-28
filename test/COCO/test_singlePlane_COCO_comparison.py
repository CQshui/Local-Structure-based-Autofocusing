#!/usr/bin/env python
# coding=utf-8
import os
import argparse

from comparison_method.global_methods.eig import calculate_eig_focus_curve

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', default='0', type=str)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

import csv
import math
import time
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
import os
from tqdm import tqdm

from comparison_method.global_methods.GoG_ToG import calculate_tog
from peak_finder.peakfinder import PeakFinder
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

    # 定义聚焦分数和z的函数，输入z，输出score
    # @timer
    def func(self, z, **kwargs):
        in_brent = kwargs['in_brent']
        propped = AngularBatch(lam=532e-9,
                               # pix=0.342e-6,
                               pix=1e-6,
                               z_initial=z,
                               input_pth=r'F:\dongjiayao\Data\holograms',
                               output_pth=r'F:\dongjiayao\Data\tmp',
                               input_img=self.filtered_tensor,
                               inline=True,
                               get_filtered=False
                               ).start()  # filtered处于空间域，获取频谱滤波结果，位于torch_gpu上

        # GPU版本处理
        # 假设 propped 是 GPU 上的 Tensor，形状为 [H, W]
        propped = torch.abs(propped)
        propped = (propped / propped.max()).clamp(0, 1)  # 归一化到 [0, 1]
        propped = (propped * 255).to(torch.uint8)  # 转成 0-255，和原始行为一致

        propped_copy = propped

        # 转为 [1, 1, H, W] 的形状，直接用于模型推理
        propped = propped.unsqueeze(0).unsqueeze(0).to(torch.float32)  # 加 batch 和 channel 维度

        # 推理
        score = calculate_eig_focus_curve(propped)
        # score = variance_focus(propped)
        # score = calculate_tog(propped)
        # score = calculate_gog(propped)
        # print(score)

        # 若需要可视化图像
        if "get_img" in kwargs:
            propped_np = torch.abs(propped_copy).cpu().numpy()
            propped_np = (propped_np / propped_np.max() * 255).astype(np.uint8)
            propped_pil = Image.fromarray(propped_np, mode="L").convert('RGB')
            return propped_pil
        else:
            return score

# @timer
def main(root, save_dir, lb, ub, model=None, fx_ref=None, fy_ref=None):
    filtered = initial(root=root, fx_ref=fx_ref, fy_ref=fy_ref)
    focus_z = focus_and_z(filtered, model)     # todo get_score_directly表示是否用模型直接得到聚焦分数

    def max_func(z, scale=None):
        # in_brent很重要，防止在优化时在小区间重复迭代，导致误识别颗粒
        score = focus_z.func(z, in_brent=True, scale=scale)
        return -score.item() if isinstance(score, torch.Tensor) else -score

    finder = PeakFinder(max_func, lb=lb, ub=ub, max_evals=200, target='min', precision_factor=5)
    z_result = finder.find_optimum()
    img_result = focus_z.func(z_result, in_brent=True, get_img=True)

    save_path = os.path.join(save_dir, os.path.basename(root))
    img_result.save(save_path)      # todo 不保存了，省一点时间

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    choice = ['mobilercf', 'rcf', 'focus_score'][1]

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
        z_pred = main(lb=lb, ub=ub, root=image_path, save_dir=save_dir, model=None, fx_ref=fx_ref, fy_ref=fy_ref)

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
    test_focusing_accuracy(root_dir=r'F:\dongjiayao\Data\COCO\val\holograms',
                           save_dir=r'F:\dongjiayao\Data\COCO\val\article\comparison\gog',
                           test_csv=r'F:\dongjiayao\Data\COCO\val\holograms\AutoFocusDatabase.csv')
