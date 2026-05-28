import os

import torch

from edge_prediction.model.BDCN import BDCN
from edge_prediction.model.CATS import CATS
from edge_prediction.model.CHRNet import CHRNet
from edge_prediction.model.HED import HED
from edge_prediction.model.MUGE import MUGE
from edge_prediction.model.RCF import RCF


def print_all_models_params(models_dict, detailed=False):
    """
    打印多个模型的参数量统计
    Args:
        models_dict: dict, {模型名称: 模型实例}
        detailed: bool, 是否打印每个子模块的详细参数量（默认 False 只打印总计）
    """

    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    print("\n" + "=" * 70)
    print(f"{'Model Name':<25} | {'Total Params (M)':>15} | {'Trainable (M)':>15}")
    print("-" * 70)

    for name, model in models_dict.items():
        total, trainable = count_params(model)
        print(f"{name:<25} | {total / 1e6:>15.2f} | {trainable / 1e6:>15.2f}")

    print("=" * 70 + "\n")

    if detailed:
        # 可选：打印第一个模型的子模块详情（或分别打印）
        for name, model in models_dict.items():
            print(f"\n--- Detailed breakdown for {name} ---")
            for module_name, module in model.named_children():
                total, trainable = count_params(module)
                print(f"  {module_name:<20} | {total / 1e6:>10.2f} M | {trainable / 1e6:>10.2f} M")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 实例化你的所有模型
    rcf = RCF(device=device).to(device)
    muge = MUGE(encoder_name="efficientnet-b7").to(device)
    cats = CATS().to(device)
    bdcn = BDCN().to(device)
    hed = HED().to(device)
    chrnet = CHRNet().to(device)


    # 放入字典
    models = {
        "BDCN": bdcn,
        "HED": hed,
        "CHRNet": chrnet,
        "RCF": rcf,
        "MUGE": muge,
        "CATS": cats,
    }

    # 打印统计
    print_all_models_params(models, detailed=False)