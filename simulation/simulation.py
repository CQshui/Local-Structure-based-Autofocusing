import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import os
import random
import csv


def forward_propagation(image_array, z, lam, pix, propagation_type='inline', fx_ref=0.0, fy_ref=0.0):
    """
    正向传播：从物体平面到全息图平面
    :param image_array: 输入图像数组
    :param z: 传播距离 (米)
    :param lam: 波长 (米)
    :param pix: 像素大小 (米)
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param fx_ref: 离轴参考光x方向空间频率 (1/米)
    :param fy_ref: 离轴参考光y方向空间频率 (1/米)
    :return: 全息图强度
    """
    # 归一化图像
    obj_amplitude = image_array / 255.0

    # 获取图像尺寸
    height, width = obj_amplitude.shape

    # 计算空间频率
    fx = np.linspace(-1 / (2 * pix), 1 / (2 * pix), width)
    fy = np.linspace(-1 / (2 * pix), 1 / (2 * pix), height)
    FX, FY = np.meshgrid(fx, fy)

    # 计算传递函数
    k = 2 * np.pi / lam  # 波数
    temp = 1 - (lam ** 2) * (FX ** 2 + FY ** 2)
    temp[temp < 0] = 0  # 消除倏逝波

    H = np.exp(1j * k * z * np.sqrt(temp))
    H = fftshift(H)  # 移动零频率到中心

    # 正向传播
    U1 = fft2(fftshift(obj_amplitude))  # FFT
    U2 = U1 * H  # 频域相乘
    U3 = ifftshift(ifft2(U2))  # IFFT

    # 创建参考光
    if propagation_type == 'inline':
        # 同轴平面波
        reference = 1.0
    else:  # off_axis
        # 构建空间坐标网格 (以图像中心为原点)
        x = (np.arange(width) - width // 2) * pix
        y = (np.arange(height) - height // 2) * pix
        X, Y = np.meshgrid(x, y)

        # 离轴平面波 (倾斜平面波)
        reference = np.exp(1j * 2 * np.pi * (fx_ref * X + fy_ref * Y))

    # 计算全息图场分布和强度
    hologram_field = reference + U3
    hologram_intensity = np.abs(hologram_field) ** 2

    return hologram_intensity


def generate_hologram(image_path, output_path, lam, pix, propagation_type='inline', fx_ref=0.0, fy_ref=0.0):
    """
    生成单个全息图
    :param image_path: 输入图像路径
    :param output_path: 输出文件夹
    :param lam: 波长
    :param pix: 像素大小
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param fx_ref: 离轴参考光x方向空间频率 (1/米)
    :param fy_ref: 离轴参考光y方向空间频率 (1/米)
    :return: 全息图文件名
    """
    # 加载图像
    # image = Image.open(image_path).convert("L")
    # img_array = np.asarray(image)

    image = Image.open(image_path).convert("L")
    img_resized = image.resize((260, 260), Image.BILINEAR)  # 或 Image.LANCZOS
    img_array = np.asarray(image)

    # 随机传播距离 (0.01-0.05米)
    z = random.uniform(0.0001, 0.0008)
    # z = random.uniform(0.0010, 0.0040)

    # 正向传播
    hologram = forward_propagation(img_array, z, lam, pix, propagation_type, fx_ref, fy_ref)

    # 创建输出文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    hologram_name = f"{base_name}_{propagation_type}_holo.jpg"
    hologram_path = os.path.join(output_path, hologram_name)

    # 归一化并保存
    hologram_normalized = (hologram - hologram.min()) / (hologram.max() - hologram.min())
    plt.imsave(hologram_path, hologram_normalized, cmap='gray')

    return hologram_name, z


def batch_process(input_dir, output_dir, propagation_type='inline'):
    """
    批量处理VOC数据集
    :param input_dir: 输入图像目录
    :param output_dir: 输出目录
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :return: 所有全息图信息
    """
    all_info = []
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 计算最大空间频率 (奈奎斯特频率)
    max_freq = 1 / (2 * pix)

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)

        # 设置参考光参数
        fx_ref = 0.0
        fy_ref = 0.0

        if propagation_type == 'off_axis':
            # 确保频谱峰值位置合理分布，将频谱图四等分
            # 目标：零级在中心，两个一级频谱在1/4和3/4位置

            # 计算合适的参考光频率
            target_fx = max_freq / 2  # 目标频率为奈奎斯特频率的一半

            # 随机选择方向（左或右）
            if random.random() > 0.5:
                fx_ref = target_fx
            else:
                fx_ref = -target_fx

            # y方向保持为0，避免复杂化
            fy_ref = 0.0

            # 确保有足够的离轴分量
            while abs(fx_ref) < 0.1 * max_freq:
                fx_ref = random.choice([target_fx, -target_fx])

        # 生成全息图
        hologram_name, z = generate_hologram(
            img_path, output_dir, lam, pix, propagation_type, fx_ref, fy_ref
        )

        # 收集信息 - 按照要求的表头格式
        all_info.append({
            'wavelength': lam * 1e9,  # 转换为nm
            'pix': pix * 1e6,  # 转换为um
            'z': z,  # 保持米单位
            'hologram_name': hologram_name,
            'reconstruction_name': hologram_name,  # 与hologram_name相同
            'fx_ref': fx_ref,  # 新增
            'fy_ref': fy_ref  # 新增
        })

    return all_info


def save_to_csv(data, csv_path):
    """
    保存全息图信息到CSV
    :param data: 全息图信息列表
    :param csv_path: CSV文件路径
    """
    with open(csv_path, 'w', newline='') as csvfile:
        # 按照要求的表头顺序
        fieldnames = ['wavelength', 'pix', 'z', 'hologram_name', 'reconstruction_name', 'fx_ref', 'fy_ref']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    # 参数设置
    lam = 532e-9
    pix = 1e-6

    # 输入输出路径
    input_path = r'F:\dongjiayao\Data\PASCAL\aug_data\0.0_1'
    output_path = r'F:\dongjiayao\Data\PASCAL\phase_amp_train\0.0_1\A-P'
    csv_path = os.path.join(output_path, 'AutoFocusDatabase.csv')

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 选择传播类型 ('inline' 或 'off_axis')
    propagation_type = 'off_axis'

    # 批量处理
    hologram_data = batch_process(input_path, output_path, propagation_type)

    # 保存CSV
    save_to_csv(hologram_data, csv_path)

    print(f"处理完成! 共生成 {len(hologram_data)} 个{propagation_type}全息图")
    print(f"全息图保存至: {output_path}")
    print(f"CSV文件保存至: {csv_path}")
