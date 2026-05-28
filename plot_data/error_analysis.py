import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter

# 设置全局样式
plt.rcParams['font.sans-serif'] = 'SimHei'  # 使用更友好的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
根据获得的误差CSV文件作图
"""

def visualize_focusing_errors(csv_path, output_dir=None):
    """
    可视化聚焦距离预测误差

    :param csv_path: 包含误差数据的CSV文件路径
    :param output_dir: 保存图表的目录（默认为CSV文件所在目录）
    """
    # 读取数据
    df = pd.read_csv(csv_path)

    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # 计算统计信息
    abs_error_mean = df['abs_error'].mean()
    abs_error_median = df['abs_error'].median()
    abs_error_std = df['abs_error'].std()

    rel_error = df['rel_error'] * 100  # 转换为百分比
    rel_error_mean = rel_error.mean()
    rel_error_median = rel_error.median()
    rel_error_std = rel_error.std()

    num_samples = len(df)

    # 打印基本统计信息
    print("=" * 60)
    print(f"聚焦距离预测误差分析 (样本数: {num_samples})")
    print("=" * 60)
    print(f"平均绝对误差: {abs_error_mean:.6f} m")
    print(f"绝对误差中位数: {abs_error_median:.6f} m")
    print(f"绝对误差标准差: {abs_error_std:.6f} m")
    print(f"平均相对误差: {rel_error_mean:.2f}%")
    print(f"相对误差中位数: {rel_error_median:.2f}%")
    print(f"相对误差标准差: {rel_error_std:.2f}%")
    print("=" * 60)

    # 1. 预测值 vs 真实值散点图
    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(data=df, x='z_gt', y='z_pred',
                         hue='abs_error', palette='viridis',
                         size='abs_error', sizes=(20, 200),
                         alpha=0.8)

    # 添加参考线
    min_val = min(df['z_gt'].min(), df['z_pred'].min())
    max_val = max(df['z_gt'].max(), df['z_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

    plt.title('真实值 vs 预测值', fontsize=18)
    plt.xlabel('真实聚焦距离 (m)', fontsize=14)
    plt.ylabel('预测聚焦距离 (m)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 添加颜色条
    norm = plt.Normalize(df['abs_error'].min(), df['abs_error'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('绝对误差 (m)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_truth.png'), dpi=300)
    plt.close()

    # 2. 误差分布直方图
    plt.figure(figsize=(14, 6))

    # 绝对误差分布
    plt.subplot(1, 2, 1)
    sns.histplot(df['abs_error'], bins=30, kde=True, color='royalblue')
    plt.axvline(abs_error_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'均值: {abs_error_mean:.4f}')
    plt.axvline(abs_error_median, color='green', linestyle='dashed', linewidth=1.5,
                label=f'中位数: {abs_error_median:.4f}')
    plt.title('绝对误差分布', fontsize=16)
    plt.xlabel('绝对误差 (m)', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 相对误差分布
    plt.subplot(1, 2, 2)
    sns.histplot(rel_error, bins=30, kde=True, color='coral')
    plt.axvline(rel_error_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'均值: {rel_error_mean:.2f}%')
    plt.axvline(rel_error_median, color='green', linestyle='dashed', linewidth=1.5,
                label=f'中位数: {rel_error_median:.2f}%')
    plt.title('相对误差分布', fontsize=16)
    plt.xlabel('相对误差 (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
    plt.close()

    # 3. 误差箱线图 - 修正为分开绘制
    plt.figure(figsize=(14, 8))

    # 绝对误差箱线图
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['abs_error'], color='royalblue', width=0.5)
    plt.title('绝对误差箱线图', fontsize=16)
    plt.ylabel('绝对误差 (m)', fontsize=12)

    # 添加统计值标注
    abs_stats = df['abs_error'].describe()
    plt.text(0.5, 0.9, f'中位数: {abs_stats["50%"]:.6f}m',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.text(0.5, 0.8, f'均值: {abs_error_mean:.6f}m',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.grid(axis='y', alpha=0.3)

    # 相对误差箱线图
    plt.subplot(1, 2, 2)
    sns.boxplot(y=rel_error, color='coral', width=0.5)
    plt.title('相对误差箱线图', fontsize=16)
    plt.ylabel('相对误差 (%)', fontsize=12)

    # 添加统计值标注
    rel_stats = rel_error.describe()
    plt.text(0.5, 0.9, f'中位数: {rel_stats["50%"]:.2f}%',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.text(0.5, 0.8, f'均值: {rel_error_mean:.2f}%',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.grid(axis='y', alpha=0.3)

    plt.suptitle('误差分布箱线图', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_boxplot.png'), dpi=300)
    plt.close()

    # 4. 误差随真实值变化的趋势图
    plt.figure(figsize=(14, 10))

    # 绝对误差趋势
    plt.subplot(2, 1, 1)
    sns.regplot(data=df, x='z_gt', y='abs_error',
                scatter_kws={'alpha': 0.6, 'color': 'steelblue'},
                line_kws={'color': 'red', 'linewidth': 2.5})
    plt.title('绝对误差随真实值变化趋势', fontsize=16)
    plt.xlabel('真实聚焦距离 (m)', fontsize=14)
    plt.ylabel('绝对误差 (m)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 相对误差趋势
    plt.subplot(2, 1, 2)
    sns.regplot(data=df, x='z_gt', y='rel_error',
                scatter_kws={'alpha': 0.6, 'color': 'salmon'},
                line_kws={'color': 'red', 'linewidth': 2.5})
    plt.title('相对误差随真实值变化趋势', fontsize=16)
    plt.xlabel('真实聚焦距离 (m)', fontsize=14)
    plt.ylabel('相对误差', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_trend.png'), dpi=300)
    plt.close()

    # 5. 综合误差热力图
    plt.figure(figsize=(12, 10))

    # 计算相关系数矩阵
    corr_matrix = df[['z_gt', 'z_pred', 'abs_error', 'rel_error']].corr()

    # 创建热力图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5, cbar_kws={'label': '相关系数'},
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('变量间相关性热力图', fontsize=18)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # 6. 累积误差分布图（分开绘制）
    plt.figure(figsize=(14, 10))

    # 绝对误差累积分布
    plt.subplot(2, 1, 1)
    abs_error_sorted = np.sort(df['abs_error'])
    y_abs = np.arange(1, len(abs_error_sorted) + 1) / len(abs_error_sorted)
    plt.plot(abs_error_sorted, y_abs, linewidth=3, color='royalblue')

    # 添加关键点标记
    for percentile in [0.5, 0.9, 0.95, 0.99]:
        abs_thresh = np.percentile(abs_error_sorted, percentile * 100)
        plt.axvline(abs_thresh, color='gray', linestyle=':', alpha=0.7)
        plt.text(abs_thresh, percentile + 0.02, f'{abs_thresh:.6f}m ({percentile:.0%})',
                 ha='left', va='bottom', fontsize=10)

    plt.title('绝对误差累积分布', fontsize=16)
    plt.xlabel('绝对误差 (m)', fontsize=12)
    plt.ylabel('累积比例', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 相对误差累积分布
    plt.subplot(2, 1, 2)
    rel_error_sorted = np.sort(rel_error)
    y_rel = np.arange(1, len(rel_error_sorted) + 1) / len(rel_error_sorted)
    plt.plot(rel_error_sorted, y_rel, linewidth=3, color='coral')

    # 添加关键点标记
    for percentile in [0.5, 0.9, 0.95, 0.99]:
        rel_thresh = np.percentile(rel_error_sorted, percentile * 100)
        plt.axvline(rel_thresh, color='gray', linestyle=':', alpha=0.7)
        plt.text(rel_thresh, percentile + 0.02, f'{rel_thresh:.2f}% ({percentile:.0%})',
                 ha='left', va='bottom', fontsize=10)

    plt.title('相对误差累积分布 (%)', fontsize=16)
    plt.xlabel('相对误差 (%)', fontsize=12)
    plt.ylabel('累积比例', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_error.png'), dpi=300)
    plt.close()

    # 7. 误差分布小提琴图 - 修正为分开绘制
    plt.figure(figsize=(14, 8))

    # 绝对误差小提琴图
    plt.subplot(1, 2, 1)
    sns.violinplot(y=df['abs_error'], color='royalblue', inner="quartile")
    plt.title('绝对误差分布', fontsize=16)
    plt.ylabel('绝对误差 (m)', fontsize=12)

    # 添加统计值标注
    abs_stats = df['abs_error'].describe()
    plt.text(0.5, 0.9, f'中位数: {abs_stats["50%"]:.6f}m',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.text(0.5, 0.8, f'均值: {abs_error_mean:.6f}m',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.grid(axis='y', alpha=0.3)

    # 相对误差小提琴图
    plt.subplot(1, 2, 2)
    sns.violinplot(y=rel_error, color='coral', inner="quartile")
    plt.title('相对误差分布', fontsize=16)
    plt.ylabel('相对误差 (%)', fontsize=12)

    # 添加统计值标注
    rel_stats = rel_error.describe()
    plt.text(0.5, 0.9, f'中位数: {rel_stats["50%"]:.2f}%',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.text(0.5, 0.8, f'均值: {rel_error_mean:.2f}%',
             transform=plt.gca().transAxes, fontsize=12, ha='center')
    plt.grid(axis='y', alpha=0.3)

    plt.suptitle('误差分布小提琴图', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_violin.png'), dpi=300)
    plt.close()

    print(f"所有图表已保存至: {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 替换为你想保存图表的目录（可选）
    output_directory = r'F:\dongjiayao\Data\AutoFocusDatabase\article\comparison\var'

    csv_file = output_directory + r'\var.csv'

    # 生成所有图表
    visualize_focusing_errors(csv_file, output_directory)
