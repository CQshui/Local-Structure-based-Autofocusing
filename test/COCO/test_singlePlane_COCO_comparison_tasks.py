#!/usr/bin/env python
# coding=utf-8
import sys
import os

# ========== 保存原始命令行参数 ==========
_original_argv = sys.argv[:]

# ========== 构造用于预解析的最小参数列表 ==========
# 只保留脚本名和 --gpu 参数（如果存在）
_temp_argv = [_original_argv[0]]
for i, arg in enumerate(_original_argv):
    if arg == '--gpu' and i + 1 < len(_original_argv):
        _temp_argv.extend(['--gpu', _original_argv[i + 1]])
        break
if '--gpu' not in _temp_argv:
    _temp_argv.extend(['--gpu', '0'])

# 临时替换 sys.argv，确保后续所有导入的模块不会看到用户的其他参数
sys.argv = _temp_argv

# 现在可以安全地导入 argparse 并设置 CUDA_VISIBLE_DEVICES
import argparse

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument('--gpu', default='0', type=str)
_pre_args, _ = _pre_parser.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

# ========== 安全导入所有模块（此时 sys.argv 只有最小参数） ==========
import csv
import math
import time
import numpy as np
import torch
from PIL import Image
from natsort import natsorted
import torch.nn.functional as F
from tqdm import tqdm

from edge_prediction.model.RCF import RCF
from peak_finder.peakfinder import PeakFinder
from edge_prediction.predict import calculate_brightness_concentration
from edge_prediction.dataloader.data_loader import prepare_image_torch_gpu
from propagate.angular_torch import AngularBatch

from comparison_method.global_methods.eig import calculate_eig_focus_curve
from comparison_method.global_methods.GoG_ToG import calculate_tog, calculate_gog
from comparison_method.global_methods.variance import variance_focus

# ========== 恢复原始命令行参数 ==========
sys.argv = _original_argv

# ── 方法注册表 ────────────────────────────────────────────────
METHOD_REGISTRY = {
    'eigen': calculate_eig_focus_curve,
    'tog': calculate_tog,
    'gog': calculate_gog,
    'var': variance_focus,
}
MODEL_METHODS = {'ours_rcf', 'ours_dlep'}
MODEL_CKPT = {
    'ours_rcf': r'F:\dongjiayao\Pycharm\DEP-AF\edge_prediction\tmp\rcf\speckle_aug_true\checkpoint_epoch10.pth',
    'ours_dlep': None,
}


# ─────────────────────────────────────────────────────────────


def set_deterministic():
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设置确定性环境，随机种子: {seed}")


set_deterministic()


def initial(root, fx_ref=None, fy_ref=None):
    hologram = Image.open(root).convert('L')
    filtered = AngularBatch(
        lam=532e-9, pix=1e-6, z_initial=0.00030,
        input_img=hologram, inline=False,
        get_filtered=True, fx_ref=fx_ref, fy_ref=fy_ref,
    ).start()
    return filtered


def focus_value_model(model, image_3d, scale=8):
    """image_3d: GPU tensor (H, W, 3) float32 [0,255]"""
    model.eval()
    h, w = image_3d.shape[0], image_3d.shape[1]
    image = F.interpolate(
        image_3d.permute(2, 0, 1).unsqueeze(0),
        size=(h // scale, w // scale),
        mode='nearest'
    ).squeeze(0).permute(1, 2, 0)
    image = prepare_image_torch_gpu(image).unsqueeze(0).cuda()
    results = model(image)
    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    result = Image.fromarray((result * 255).astype(np.uint8))
    return calculate_brightness_concentration(np.array(result), tensor=results[-1])


class focus_and_z:
    def __init__(self, filtered_tensor, method, model=None):
        self.filtered_tensor = filtered_tensor
        self.method = method
        self.model = model

    def func(self, z, **kwargs):
        propped = AngularBatch(
            lam=532e-9, pix=1e-6, z_initial=z,
            input_img=self.filtered_tensor,
            inline=True, get_filtered=False,
        ).start()

        propped_abs = torch.abs(propped)
        propped_norm = (propped_abs / propped_abs.max()).clamp(0, 1)
        propped_uint8 = (propped_norm * 255).to(torch.uint8)

        if self.method in METHOD_REGISTRY:
            propped_4d = propped_uint8.unsqueeze(0).unsqueeze(0).to(torch.float32)
            score = METHOD_REGISTRY[self.method](propped_4d)

        elif self.method in MODEL_METHODS:
            propped_float = propped_uint8.to(torch.float32)
            propped_3d = propped_float.unsqueeze(2).repeat(1, 1, 3)
            score = focus_value_model(self.model, propped_3d, scale=1)

        else:
            raise ValueError(f"未知方法: {self.method}")

        if "get_img" in kwargs:
            propped_np = propped_abs.cpu().numpy()
            propped_np = (propped_np / propped_np.max() * 255).astype(np.uint8)
            return Image.fromarray(propped_np, mode="L").convert('RGB')

        return score


def main(root, save_dir, lb, ub, method, model=None, fx_ref=None, fy_ref=None):
    filtered = initial(root=root, fx_ref=fx_ref, fy_ref=fy_ref)
    fz = focus_and_z(filtered, method=method, model=model)

    def max_func(z, scale=None):
        score = fz.func(z, in_brent=True, scale=scale)
        return -score.item() if isinstance(score, torch.Tensor) else -score

    finder = PeakFinder(max_func, lb=lb, ub=ub, max_evals=200,
                        target='min', precision_factor=1,
                        initial_points_factor=5)
    z_result = finder.find_optimum()
    return z_result


def batch_main(root_dir, save_dir, method, test_csv=None):
    lb, ub = 0.00005, 0.00085
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}  (物理 GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"聚焦方法: {method}")

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = natsorted([
        f for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
           and os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS
    ])
    os.makedirs(save_dir, exist_ok=True)

    ground_truth, fx, fy = {}, {}, {}
    if test_csv and os.path.exists(test_csv):
        print(f"加载测试数据: {test_csv}")
        with open(test_csv, 'r') as f:
            for row in csv.DictReader(f):
                name = row['hologram_name']
                ground_truth[name] = float(row['z'])
                fx[name] = float(row['fx_ref']) if 'fx_ref' in row else None
                fy[name] = float(row['fy_ref']) if 'fy_ref' in row else None

    # ── 模型加载 ─────────────────────────────────────────────
    model = None
    if method in MODEL_METHODS:
        print(f"加载模型: {method}")
        model = RCF()
        model.to(device)
        ckpt_path = MODEL_CKPT.get(method)
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=True)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            print(f"  权重加载成功: {ckpt_path}")
        else:
            print(f"  未找到权重，使用随机初始化")
        model.eval()
    # ─────────────────────────────────────────────────────────

    results = []
    results_csv = os.path.join(save_dir, "focusing_results.csv")
    fieldnames = ['hologram_name', 'z_gt', 'z_pred', 'abs_error', 'rel_error']
    with open(results_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    pbar = tqdm(image_files, desc="Processing holograms", unit="image")
    total_abs = total_rel = 0.0
    valid_rel = processed = 0

    for image_file in pbar:
        pbar.set_description(f"Processing: {image_file}")
        image_path = os.path.join(root_dir, image_file)
        fx_ref = fx.get(image_file)
        fy_ref = fy.get(image_file)

        z_pred = main(lb=lb, ub=ub, root=image_path,
                      save_dir=save_dir, method=method,
                      model=model, fx_ref=fx_ref, fy_ref=fy_ref)

        result_row = {'hologram_name': image_file, 'z_gt': '',
                      'z_pred': z_pred, 'abs_error': '', 'rel_error': ''}

        if image_file in ground_truth:
            z_gt = ground_truth[image_file]
            abs_error = abs(z_pred - z_gt)
            rel_error = abs_error / abs(ub - lb) if abs(ub - lb) > 1e-9 else float('inf')

            processed += 1;
            total_abs += abs_error
            if not math.isinf(rel_error):
                total_rel += rel_error;
                valid_rel += 1

            avg_abs = total_abs / processed
            avg_rel = total_rel / valid_rel * 100 if valid_rel > 0 else float('nan')

            result_row.update({'z_gt': z_gt, 'abs_error': abs_error, 'rel_error': rel_error})
            pbar.set_postfix({
                'z_gt': f"{z_gt:.7f}",
                'z_pred': f"{z_pred:.7f}",
                'abs_err': f"{abs_error:.7f}",
                'rel_err(%)': f"{rel_error * 100:.2f}" if not math.isinf(rel_error) else "Inf",
                'avg_abs': f"{avg_abs:.7f}",
                'avg_rel(%)': f"{avg_rel:.2f}" if not math.isnan(avg_rel) else "NaN",
            })
        else:
            pbar.set_postfix({'z_pred': f"{z_pred:.7f}"})

        results.append(result_row.copy())
        with open(results_csv, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(result_row)

    valid_results = [r for r in results if r['abs_error'] != '']
    if valid_results:
        print("\n" + "=" * 50)
        print(f"聚焦距离预测误差统计  [{method}]")
        print("=" * 50)
        abs_errs = [r['abs_error'] for r in valid_results]
        rel_errs = [r['rel_error'] for r in valid_results if not math.isinf(r['rel_error'])]
        print(f"处理图像数量: {len(valid_results)}")
        print(f"平均/最小/最大绝对误差: "
              f"{np.mean(abs_errs):.7f} / {min(abs_errs):.7f} / {max(abs_errs):.7f} m")
        if rel_errs:
            print(f"平均/最小/最大相对误差: "
                  f"{np.mean(rel_errs) * 100:.2f}% / "
                  f"{min(rel_errs) * 100:.2f}% / "
                  f"{max(rel_errs) * 100:.2f}%")
        print(f"\n详细结果已保存至: {results_csv}")


# ── 命令行入口 ────────────────────────────────────────────────
if __name__ == "__main__":
    all_methods = list(METHOD_REGISTRY.keys()) + list(MODEL_METHODS)

    parser = argparse.ArgumentParser(description='Digital Hologram Auto-Focus Evaluation')
    parser.add_argument('--gpu', default='0', type=str,
                        help='物理 GPU ID')
    parser.add_argument('--method', default='eigen', type=str,
                        choices=all_methods,
                        help=f'聚焦方法，可选: {all_methods}')
    parser.add_argument('--root_dir', required=True, type=str,
                        help='全息图目录')
    parser.add_argument('--save_dir', required=True, type=str,
                        help='结果保存目录')
    parser.add_argument('--test_csv', default=None, type=str,
                        help='含真实z值的CSV文件路径')
    args = parser.parse_args()

    batch_main(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        method=args.method,
        test_csv=args.test_csv,
    )
