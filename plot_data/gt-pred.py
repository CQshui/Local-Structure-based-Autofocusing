import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter

# 设置全局样式 - 适合学术论文的简洁风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


def create_combined_comparison_figure(csv_paths, method_names, output_dir, colors=None):
    """
    创建组合的预测结果图，将所有方法显示在一张大图中

    :param csv_paths: 各方法CSV文件路径列表
    :param method_names: 各方法名称列表
    :param output_dir: 输出目录
    :param colors: 各方法颜色列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置颜色
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 计算子图布局
    n_methods = len(csv_paths)
    n_cols = min(3, n_methods)  # 最多3列
    n_rows = (n_methods + n_cols - 1) // n_cols

    # 设置图形大小 - 根据子图数量调整
    fig_width = 10  # 整个大图的宽度（英寸）
    fig_height = 3 * n_rows  # 高度根据行数调整

    # 创建大图和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 如果只有一行，确保axes是数组
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # 存储所有方法的统计指标
    metrics = []

    # 首先计算所有数据的全局最小值和最大值
    all_gt_min = float('inf')
    all_gt_max = float('-inf')
    all_pred_min = float('inf')
    all_pred_max = float('-inf')

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        all_gt_min = min(all_gt_min, df['z_gt'].min())
        all_gt_max = max(all_gt_max, df['z_gt'].max())
        all_pred_min = min(all_pred_min, df['z_pred'].min())
        all_pred_max = max(all_pred_max, df['z_pred'].max())

    # 确定统一的坐标轴范围
    global_min = min(all_gt_min, all_pred_min)
    global_max = max(all_gt_max, all_pred_max)

    # 稍微扩展范围以美观
    margin = (global_max - global_min) * 0.02
    global_min -= margin
    global_max += margin

    # 为每个方法创建子图
    for i, (csv_path, method_name) in enumerate(zip(csv_paths, method_names)):
        # 读取数据
        df = pd.read_csv(csv_path)

        # 计算关键统计指标
        mae = df['abs_error'].mean()
        rmse = np.sqrt((df['abs_error'] ** 2).mean())
        r2 = 1 - np.sum((df['z_gt'] - df['z_pred']) ** 2) / np.sum((df['z_gt'] - df['z_gt'].mean()) ** 2)

        # 确定子图位置
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]

        # 散点图 - 使用单一颜色，更小的点
        color = colors[i % len(colors)]
        ax.scatter(df['z_gt'], df['z_pred'],
                   c=color, s=15, alpha=0.7, linewidths=0)

        # 添加参考线 (y=x)
        ax.plot([global_min, global_max], [global_min, global_max],
                'k--', alpha=0.5, linewidth=0.8)

        # 设置统一的坐标轴范围
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        # 确保坐标轴比例一致
        ax.set_aspect('equal', adjustable='box')

        # 只在底部和左侧显示坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 设置坐标轴标签
        ax.set_xlabel('GT (m)', fontsize=10)
        ax.set_ylabel('Pred (m)', fontsize=10)

        # 设置刻度
        ax.tick_params(axis='both', which='major', labelsize=9)

        # 确保刻度数量和位置一致
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # 添加方法名称作为标题
        ax.set_title(method_name, fontsize=11, pad=10)

        # 在图表内部添加关键统计指标
        # stats_text = f'MAE: {mae:.3f}m\nRMSE: {rmse:.3f}m\nR²: {r2:.3f}'
        # ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        #         fontsize=9, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))

        # 存储指标
        metrics.append({
            'Method': method_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

    # 隐藏多余的子图
    for i in range(n_methods, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        if n_rows > 1:
            axes[row_idx][col_idx].set_visible(False)
        else:
            axes[col_idx].set_visible(False)

    # 调整子图间距
    plt.tight_layout()

    # 保存组合图像
    plt.savefig(os.path.join(output_dir, 'combined_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 创建指标表格
    metrics_df = pd.DataFrame(metrics)

    # 创建性能比较条形图
    fig_width = 8
    fig_height = 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # MAE和RMSE比较
    x_pos = np.arange(len(method_names))
    width = 0.35

    ax1.bar(x_pos - width / 2, metrics_df['MAE'], width,
            label='MAE', color='#1f77b4', alpha=0.7)
    ax1.bar(x_pos + width / 2, metrics_df['RMSE'], width,
            label='RMSE', color='#ff7f0e', alpha=0.7)

    ax1.set_ylabel('Error (m)', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names, fontsize=9, rotation=45)
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # R²比较
    ax2.bar(x_pos, metrics_df['R²'], color='#2ca02c', alpha=0.7)
    ax2.set_ylabel('R²', fontsize=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(method_names, fontsize=9, rotation=45)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 保存指标表格
    metrics_df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)

    return metrics_df


# 使用示例
if __name__ == "__main__":
    output_directory = r'F:\dongjiayao\Data\COCO\val\article\comparison\000'
    # output_directory = r'F:\dongjiayao\Data\holograms\Focus\results\CSVs'

    # 假设你有多个方法的CSV文件
    csv_files = [
        output_directory + r'\var.csv',
        output_directory + r'\tog.csv',
        output_directory + r'\gog.csv',
        output_directory + r'\eigen.csv',
        output_directory + r'\focusnet.csv',
        output_directory + r'\ours.csv',
    ]

    method_names = [
        'Variance',
        'ToG',
        'GoG',
        'Eigen',
        'FocusNet',
        'Proposed Method',
    ]

    # 创建组合比较图表
    metrics_df = create_combined_comparison_figure(csv_files, method_names, output_directory)

    print("比较完成！生成的指标：")
    print(metrics_df)