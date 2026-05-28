import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import os
import random
import csv
from scipy.ndimage import gaussian_filter


def calculate_speckle_contrast(intensity_pattern):
    """
    计算散斑对比度 C = σ_I / <I>
    :param intensity_pattern: 散斑强度图样
    :return: 散斑对比度
    """
    mean_intensity = np.mean(intensity_pattern)
    std_intensity = np.std(intensity_pattern)

    if mean_intensity == 0:
        return 0

    contrast = std_intensity / mean_intensity
    return contrast


def simulate_speckle_pattern(size, speckle_size=3.0, target_contrast=1.0):
    """
    模拟符合定义的散斑图样
    :param size: 图像尺寸 (height, width)
    :param speckle_size: 散斑颗粒尺寸
    :param target_contrast: 目标散斑对比度 (理论值为1.0)
    :return: 散斑强度图样
    """
    height, width = size

    # 基于傅里叶变换的散斑生成
    # 生成随机复数场（模拟相干光散射）
    random_complex = np.random.randn(height, width) + 1j * np.random.randn(height, width)

    # 应用低通滤波控制散斑尺寸
    y, x = np.ogrid[-height // 2:height // 2, -width // 2:width // 2]
    mask = np.exp(-(x ** 2 + y ** 2) / (2 * (speckle_size ** 2)))
    mask = fftshift(mask)

    # 频域滤波
    speckle_fft = fft2(random_complex) * mask
    speckle_field = ifft2(speckle_fft)

    # 计算强度（完全发展的散斑，对比度≈1）
    speckle_intensity = np.abs(speckle_field) ** 2

    # 归一化到均值为1
    speckle_intensity = speckle_intensity / np.mean(speckle_intensity)

    # 调整到目标对比度
    if target_contrast < 1.0:
        # 通过与均匀背景混合来降低对比度
        # I_adjusted = α*I_speckle + (1-α)*<I_speckle>
        # 其中 α 由目标对比度决定
        mean_val = 1.0  # 已经归一化
        speckle_intensity = target_contrast * speckle_intensity + (1 - target_contrast) * mean_val

    return speckle_intensity


def add_speckle_noise(field, speckle_contrast=0.0, speckle_size=2.0):
    """
    在全息图场中模拟散斑噪声（修正版本）
    :param field: 全息图复振幅场
    :param speckle_contrast: 目标散斑对比度 (0-1，理论最大值为1)
    :param speckle_size: 散斑颗粒尺寸 (像素)
    :return: 添加散斑噪声后的场
    """
    if speckle_contrast <= 0:
        return field

    # 获取图像尺寸
    height, width = field.shape

    # 生成符合定义的散斑图样
    speckle_pattern = simulate_speckle_pattern(
        (height, width),
        speckle_size=speckle_size,
        target_contrast=speckle_contrast
    )

    # 将散斑应用到场（振幅调制）
    # 使用sqrt是因为散斑图样是强度，而field是振幅
    speckled_field = field * np.sqrt(speckle_pattern)

    return speckled_field


def forward_propagation_with_speckle(image_array, z, lam, pix,
                                     propagation_type='inline',
                                     fx_ref=0.0, fy_ref=0.0,
                                     speckle_params=None):
    """
    正向传播：从物体平面到全息图平面（包含散斑噪声）
    :param image_array: 输入图像数组
    :param z: 传播距离 (米)
    :param lam: 波长 (米)
    :param pix: 像素大小 (米)
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param fx_ref: 离轴参考光x方向空间频率 (1/米)
    :param fy_ref: 离轴参考光y方向空间频率 (1/米)
    :param speckle_params: 散斑参数字典
        - enabled: 是否启用散斑噪声
        - contrast: 散斑对比度 (0-1)
        - size: 散斑颗粒尺寸
        - method: 散斑生成方法 ('phase' 或 'intensity')
    :return: 全息图强度
    """
    # 默认散斑参数
    if speckle_params is None:
        speckle_params = {
            'enabled': False,
            'contrast': 0.3,
            'size': 2.0,
            'method': 'intensity'
        }

    # 归一化图像
    obj_amplitude = image_array / 255.0

    # 添加散斑效应（如果启用）
    if speckle_params.get('enabled', False):
        # 方法1：在物体表面添加随机相位（模拟粗糙表面）
        if speckle_params.get('method', 'intensity') == 'phase':
            height, width = obj_amplitude.shape
            # 生成随机相位（模拟表面高度变化）
            random_phase = np.random.uniform(0, 2 * np.pi, (height, width))
            # 对相位进行平滑以控制散斑尺寸
            if speckle_params.get('size', 2.0) > 0:
                random_phase = gaussian_filter(random_phase, sigma=speckle_params['size'])
            # 将物体转换为复振幅场，添加随机相位
            obj_field = obj_amplitude * np.exp(1j * random_phase)
            # 计算传播后的强度得到散斑
            obj_amplitude = np.abs(obj_field)

        # 方法2：生成符合定义的散斑图样并叠加
        else:
            speckle_pattern = simulate_speckle_pattern(
                obj_amplitude.shape,
                speckle_size=speckle_params.get('size', 2.0),
                target_contrast=speckle_params.get('contrast', 0.3)
            )
            # 散斑图样是强度，应用到振幅
            obj_amplitude = obj_amplitude * np.sqrt(speckle_pattern)

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

    # 在全息图平面添加散斑噪声（如果启用）
    # if speckle_params.get('enabled', False) and speckle_params.get('add_at_hologram', False):
    #     hologram_field = add_speckle_noise(
    #         hologram_field,
    #         speckle_contrast=speckle_params.get('contrast', 0.3),
    #         speckle_size=speckle_params.get('size', 2.0)
    #     )

    hologram_intensity = np.abs(hologram_field) ** 2

    return hologram_intensity


def generate_hologram_with_speckle(image_path, output_path, lam, pix,
                                   propagation_type='inline',
                                   fx_ref=0.0, fy_ref=0.0,
                                   speckle_params=None):
    """
    生成单个全息图（包含散斑噪声）
    :param image_path: 输入图像路径
    :param output_path: 输出文件夹
    :param lam: 波长
    :param pix: 像素大小
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param fx_ref: 离轴参考光x方向空间频率 (1/米)
    :param fy_ref: 离轴参考光y方向空间频率 (1/米)
    :param speckle_params: 散斑参数
    :return: 全息图文件名, 传播距离
    """
    # 加载图像
    image = Image.open(image_path).convert("L")
    # image = Image.open(image_path).convert("L").resize((256, 256))
    img_array = np.asarray(image)

    # 随机传播距离
    z = random.uniform(0.00010, 0.00080)

    # 正向传播（带散斑）
    hologram = forward_propagation_with_speckle(
        img_array, z, lam, pix, propagation_type, fx_ref, fy_ref, speckle_params
    )

    # 创建输出文件名
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 根据是否添加散斑来命名
    if speckle_params and speckle_params.get('enabled', False):
        hologram_name = f"{base_name}.jpg"
    else:
        hologram_name = f"{base_name}_{propagation_type}_holo.jpg"

    hologram_path = os.path.join(output_path, hologram_name)

    # 归一化并保存
    hologram_normalized = (hologram - hologram.min()) / (hologram.max() - hologram.min())
    plt.imsave(hologram_path, hologram_normalized, cmap='gray')

    return hologram_name, z


def batch_process_with_speckle(input_dir, output_dir, propagation_type='inline',
                               speckle_enabled=True, speckle_variation=True,
                               fixed_contrast=0.5, fixed_size=3.0, fixed_method='intensity'):
    """
    批量处理VOC数据集（包含散斑噪声）
    :param input_dir: 输入图像目录
    :param output_dir: 输出目录
    :param propagation_type: 传播类型 ('inline' 或 'off_axis')
    :param speckle_enabled: 是否启用散斑噪声
    :param speckle_variation: 是否在图像间变化散斑参数
    :param fixed_contrast: 固定散斑对比度 (仅当speckle_variation=False时使用)
    :param fixed_size: 固定散斑尺寸 (仅当speckle_variation=False时使用)
    :param fixed_method: 固定散斑方法 (仅当speckle_variation=False时使用)
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
            target_fx = max_freq / 2
            if random.random() > 0.5:
                fx_ref = target_fx
            else:
                fx_ref = -target_fx
            fy_ref = 0.0

            while abs(fx_ref) < 0.1 * max_freq:
                fx_ref = random.choice([target_fx, -target_fx])

        # 设置散斑参数
        speckle_params = None
        if speckle_enabled:
            if speckle_variation:
                # 随机散斑参数，模拟不同粗糙度的表面
                speckle_contrast = random.uniform(0.1, 0.8)  # 对比度变化 (最大0.8接近理想值1)
                speckle_size = random.uniform(1.0, 4.0)  # 散斑尺寸变化
                speckle_method = random.choice(['phase', 'intensity'])  # 方法变化
            else:
                # 使用固定散斑参数（从主函数传入）
                speckle_contrast = fixed_contrast
                speckle_size = fixed_size
                speckle_method = fixed_method

            speckle_params = {
                'enabled': True,
                'contrast': speckle_contrast,
                'size': speckle_size,
                'method': speckle_method,
                'add_at_hologram': False  # 建议在物体平面添加散斑
            }

        # 生成全息图
        hologram_name, z = generate_hologram_with_speckle(
            img_path, output_dir, lam, pix, propagation_type, fx_ref, fy_ref, speckle_params
        )

        # 收集信息
        info = {
            'wavelength': lam * 1e9,
            'pix': pix * 1e6,
            'z': z,
            'hologram_name': hologram_name,
            'reconstruction_name': hologram_name,
            'fx_ref': fx_ref,
            'fy_ref': fy_ref
        }

        # 添加散斑参数到记录
        if speckle_params:
            info.update({
                'speckle_contrast': speckle_params['contrast'],
                'speckle_size': speckle_params['size'],
                'speckle_method': speckle_params['method']
            })

        all_info.append(info)

    return all_info


def visualize_speckle_effect(test_contrasts=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], speckle_size=3.0):
    """
    可视化散斑效应并验证对比度
    :param test_contrasts: 要测试的对比度列表
    :param speckle_size: 散斑颗粒尺寸
    """
    # 创建测试图像
    test_image = np.ones((512, 512)) * 128

    # 生成不同对比度的散斑
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    print("\n========== 散斑对比度验证 ==========")
    for idx, target_contrast in enumerate(test_contrasts):
        ax = axes.flatten()[idx]

        # 生成散斑图样
        speckle_pattern = simulate_speckle_pattern(
            test_image.shape,
            speckle_size=speckle_size,
            target_contrast=target_contrast
        )

        # 应用散斑到图像（强度调制）
        speckled_image = test_image * speckle_pattern
        speckled_image = np.clip(speckled_image, 0, 255).astype(np.uint8)

        # 计算实际对比度
        actual_contrast = calculate_speckle_contrast(speckled_image)

        # 显示
        ax.imshow(speckled_image, cmap='gray')
        ax.set_title(f'目标C={target_contrast:.1f}, 实际C={actual_contrast:.3f}')
        ax.axis('off')

        # 打印验证信息
        print(f"目标对比度: {target_contrast:.2f} -> 实际对比度: {actual_contrast:.3f}")

    print("====================================\n")

    plt.tight_layout()
    plt.savefig('speckle_effect_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # ==================== 散斑参数设置区域 ====================
    # 散斑对比度
    SPECKLE_CONTRAST = 0.9

    # 散斑颗粒尺寸
    SPECKLE_SIZE = 100

    # 散斑生成方法 ('phase': 相位法, 'intensity': 强度法)
    SPECKLE_METHOD = 'intensity'

    # 是否启用散斑噪声
    ENABLE_SPECKLE = True

    # 是否在图像间变化散斑参数 (True: 每张图随机参数, False: 使用上面的固定参数)
    SPECKLE_VARIATION = False

    # 可视化测试的对比度列表
    TEST_CONTRASTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # =======================================================

    # 基本参数设置
    lam = 532e-9
    pix = 1e-6

    # 输入输出路径
    input_path = r'F:\dongjiayao\Data\COCO\val\images'
    output_path = r'F:\dongjiayao\Data\COCO\val\holograms_with_speckle\contrast_{}_size100'.format(SPECKLE_CONTRAST)
    csv_path = os.path.join(output_path, 'AutoFocusDatabase.csv')

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 选择传播类型
    propagation_type = 'off_axis'  # 可修改为 'inline'

    # 可视化散斑效应
    if ENABLE_SPECKLE:
        print("正在生成散斑效应可视化...")
        print(f"使用散斑参数: 对比度={SPECKLE_CONTRAST}, 尺寸={SPECKLE_SIZE}, 方法={SPECKLE_METHOD}")
        visualize_speckle_effect(test_contrasts=TEST_CONTRASTS, speckle_size=SPECKLE_SIZE)

    # 批量处理（带散斑）
    print("\n开始批量处理全息图...")
    print(f"散斑参数变化模式: {'随机变化' if SPECKLE_VARIATION else '固定参数'}")
    if not SPECKLE_VARIATION:
        print(f"  - 散斑对比度: {SPECKLE_CONTRAST}")
        print(f"  - 散斑尺寸: {SPECKLE_SIZE}")
        print(f"  - 散斑方法: {SPECKLE_METHOD}")

    hologram_data = batch_process_with_speckle(
        input_path, output_path,
        propagation_type=propagation_type,
        speckle_enabled=ENABLE_SPECKLE,
        speckle_variation=SPECKLE_VARIATION,
        fixed_contrast=SPECKLE_CONTRAST,
        fixed_size=SPECKLE_SIZE,
        fixed_method=SPECKLE_METHOD
    )

    # 保存CSV（扩展字段以包含散斑参数）
    with open(csv_path, 'w', newline='') as csvfile:
        if ENABLE_SPECKLE:
            fieldnames = ['wavelength', 'pix', 'z', 'hologram_name', 'reconstruction_name',
                          'fx_ref', 'fy_ref', 'speckle_contrast', 'speckle_size', 'speckle_method']
        else:
            fieldnames = ['wavelength', 'pix', 'z', 'hologram_name', 'reconstruction_name',
                          'fx_ref', 'fy_ref']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in hologram_data:
            writer.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"处理完成! 共生成 {len(hologram_data)} 个{propagation_type}全息图")
    print(f"散斑噪声: {'已启用' if ENABLE_SPECKLE else '未启用'}")
    if ENABLE_SPECKLE and not SPECKLE_VARIATION:
        print(f"散斑对比度: {SPECKLE_CONTRAST}")
    print(f"全息图保存至: {output_path}")
    print(f"CSV文件保存至: {csv_path}")
    print(f"{'=' * 60}")
