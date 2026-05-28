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


def preprocess_image_to_match_B(img_array, compress_factor=0.4):
    """
    预处理图像，使其统计特征接近真实全息重建图B。
    :param img_array: 原始灰度图像数组 (0~255 uint8)
    :param compress_factor: 灰度范围压缩比例 (0~1)，越小对比度越低 (建议0.3~0.5)
    :return: 调整后的图像数组 (0~255 uint8)
    """
    img_float = img_array.astype(np.float32)
    mean_val = np.mean(img_float)

    compressed = mean_val + compress_factor * (img_float - mean_val)
    compressed = np.clip(compressed, 0, 255)

    return compressed.astype(np.uint8)


def generate_hologram(image_path, output_path, lam, pix, propagation_type='inline', fx_ref=0.0, fy_ref=0.0,
                      apply_preprocess=True, compress_factor=0.4, target_size=None):
    """
    生成单个全息图（先对原图做预处理，再进行传播）
    :param image_path: 输入图像路径
    :param output_path: 输出文件夹
    :param lam: 波长
    :param pix: 像素大小
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param fx_ref: 离轴参考光x方向空间频率 (1/米)
    :param fy_ref: 离轴参考光y方向空间频率 (1/米)
    :param apply_preprocess: 是否对原图应用预处理（灰度压缩+散斑噪声）
    :param compress_factor: 灰度压缩因子 (建议0.3~0.5)
    :param target_size: 统一尺寸 (宽, 高)，例如 (256, 256)，None 则保持原尺寸
    :return: 全息图文件名
    """
    image = Image.open(image_path).convert("L")

    # 可选：resize 到统一尺寸
    if target_size is not None:
        image = image.resize(target_size, Image.BILINEAR)

    img_array = np.asarray(image)

    # 在传播前对原图进行预处理，模拟全息重建图的统计特征
    if apply_preprocess:
        img_array = preprocess_image_to_match_B(img_array, compress_factor)

    # 随机传播距离 (0.0001~0.0008米)
    z = random.uniform(0.0001, 0.0008)

    # 正向传播
    hologram = forward_propagation(img_array, z, lam, pix, propagation_type, fx_ref, fy_ref)

    # 创建输出文件名 - 文件名中体现预处理参数
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if apply_preprocess:
        hologram_name = f"{base_name}_{propagation_type}_holo.jpg"
    else:
        hologram_name = f"{base_name}_{propagation_type}_holo.jpg"
    hologram_path = os.path.join(output_path, hologram_name)

    # 归一化并保存
    hologram_normalized = (hologram - hologram.min()) / (hologram.max() - hologram.min())
    plt.imsave(hologram_path, hologram_normalized, cmap='gray')

    return hologram_name, z


def batch_process(input_dir, output_dir, lam, pix, propagation_type='inline',
                  apply_preprocess=True, compress_factor=0.4, target_size=None):
    """
    批量处理图像，生成全息图（先预处理原图，再传播）
    :param input_dir: 输入图像目录
    :param output_dir: 输出目录
    :param lam: 波长 (米)
    :param pix: 像素大小 (米)
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param apply_preprocess: 是否对原图应用预处理
    :param compress_factor: 灰度压缩因子
    :param target_size: 统一尺寸 (宽, 高)，例如 (256, 256)，None 则保持原尺寸
    :return: 所有全息图信息
    """
    all_info = []
    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    # 计算最大空间频率 (奈奎斯特频率)
    max_freq = 1 / (2 * pix)

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)

        # 设置参考光参数
        fx_ref = 0.0
        fy_ref = 0.0

        if propagation_type == 'off_axis':
            target_fx = max_freq / 2

            if random.random() > 0.5:
                fx_ref = target_fx
            else:
                fx_ref = -target_fx

            fy_ref = 0.0

            while abs(fx_ref) < 0.1 * max_freq:
                fx_ref = random.choice([target_fx, -target_fx])

        # 生成全息图（传播前先对原图做预处理）
        hologram_name, z = generate_hologram(
            img_path, output_dir, lam, pix, propagation_type, fx_ref, fy_ref,
            apply_preprocess, compress_factor, target_size
        )

        all_info.append({
            'wavelength': lam * 1e9,
            'pix': pix * 1e6,
            'z': z,
            'hologram_name': hologram_name,
            'reconstruction_name': hologram_name,
            'fx_ref': fx_ref,
            'fy_ref': fy_ref
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
    lam = 532e-9  # 波长 (532nm)
    pix = 1e-6  # 像素大小 (1μm)

    # 预处理参数：灰度压缩0.4，散斑对比度0.5（可按需修改）
    apply_preprocess = True
    compress_factor = 0.2    # 灰度压缩因子，越小对比度越低

    # resize 参数：是否统一缩放到 256x256，None 则不缩放
    target_size = (256, 256)

    # 输入输出路径
    input_path = r'F:\dongjiayao\Data\COCO\val\images'  # 原始清晰图像
    output_path = r'F:\dongjiayao\Data\COCO\val\holograms_gray_compress_256\{}'.format(compress_factor)  # 输出全息图
    csv_path = os.path.join(output_path, 'AutoFocusDatabase.csv')

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 选择传播类型 ('inline' 或 'off_axis')
    propagation_type = 'off_axis'  # 可修改为需要的传播类型

    # 批量处理（先对原图做预处理，再进行传播）
    hologram_data = batch_process(input_path, output_path, lam, pix, propagation_type,
                                  apply_preprocess=apply_preprocess,
                                  compress_factor=compress_factor,
                                  target_size=target_size)

    # 保存CSV
    save_to_csv(hologram_data, csv_path)

    print(f"处理完成! 共生成 {len(hologram_data)} 个{propagation_type}全息图")
    print(f"全息图保存至: {output_path}")
    print(f"CSV文件保存至: {csv_path}")
    print(f"预处理参数: compress_factor={compress_factor}")
