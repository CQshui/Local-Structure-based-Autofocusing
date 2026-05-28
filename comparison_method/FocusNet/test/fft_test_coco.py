import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import csv

# 屏蔽 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== GPU 显存按需增长 ====================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"已为 {len(physical_devices)} 个 GPU 启用显存增长")
else:
    print("未检测到 GPU，使用 CPU")

# 导入自定义层（确保与训练脚本中的路径一致）
from comparison_method.FocusNet.helpers import Rotate90Randomly, Fourier2D

# ==================== 配置参数 ====================
model_path = '../models/2headed_extra_layer_log_abs_e150_ce120_lr0.0005_d0.1_bs64_dr0.05_cp_30_x4_pascal256_cp.keras'  # 根据实际最佳模型路径修改
data_dir = r'F:\dongjiayao\Data\COCO\val\holograms_256'
output_csv = r'F:\dongjiayao\Data\COCO\val\article\hf_ratio_comparison\test.csv'

output_size = (256, 256)
pix = 1e-6

# ==================== 加载测试集数据 ====================
df = pd.read_csv(f'{data_dir}/AutoFocusDatabase.csv')
x_set = df['hologram_name'].tolist()
y_set = (df['z'].values * 1e5).tolist()  # 训练时乘以 1e5 的缩放
fx_ref = df['fx_ref'].tolist()
fy_ref = df['fy_ref'].tolist()

x_test = x_set
y_test = y_set
fx_ref_test = fx_ref
fy_ref_test = fy_ref

print(f"预测集样本数: {len(x_test)}")

# ==================== 加载模型 ====================
custom_objects = {
    'Rotate90Randomly': Rotate90Randomly,
    'Fourier2D': Fourier2D
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print("模型加载成功")


# ==================== 定义预处理函数（与 RegressionSequence 逻辑一致）====================
def preprocess_image(image_path, fx_val, fy_val):
    """
    输入：图像路径，对应的 fx_ref 和 fy_ref
    输出：形状为 (1, output_size[0], output_size[1], 1) 的模型输入张量
    """
    # 1. 加载图像（原始尺寸，不resize）
    img = load_img(image_path, color_mode='grayscale')
    img_array = img_to_array(img).astype(np.float32) / 255.0
    gray = img_array.squeeze()  # (H, W)
    orig_height, orig_width = gray.shape

    # 2. 计算傅里叶变换
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    # 3. 计算截取位置（使用原始尺寸）
    width, height = orig_width, orig_height
    dfx = 1.0 / (width * pix)
    dfy = 1.0 / (height * pix)
    dx = int(round(fx_val / dfx))
    dy = int(round(fy_val / dfy))
    x_peak = (width // 2 + dx) % width
    y_peak = (height // 2 + dy) % height
    rect_w = width // 3
    rect_h = height // 3
    min_x = max(0, x_peak - rect_w // 2)
    min_y = max(0, y_peak - rect_h // 2)
    max_x = min(width, min_x + rect_w)
    max_y = min(height, min_y + rect_h)
    rect_w = max_x - min_x
    rect_h = max_y - min_y

    # 4. 截取复数频谱
    cropped_complex = fft_shift[min_y:min_y + rect_h, min_x:min_x + rect_w]

    # 5. 居中峰值
    shift_y = rect_h // 2 - (y_peak - min_y)
    shift_x = rect_w // 2 - (x_peak - min_x)
    centered = np.roll(cropped_complex, shift=(shift_y, shift_x), axis=(0, 1))

    # 6. 嵌入全尺寸数组
    full_spectrum = np.zeros((orig_height, orig_width), dtype=np.complex64)
    start_y = (orig_height - rect_h) // 2
    start_x = (orig_width - rect_w) // 2
    full_spectrum[start_y:start_y + rect_h, start_x:start_x + rect_w] = centered

    # 7. 回到空间域
    recon = np.fft.ifft2(np.fft.ifftshift(full_spectrum))
    recon_amp = np.abs(recon).astype(np.float32)

    max_val = recon_amp.max()
    if max_val > 0:
        recon_uint8 = (recon_amp / max_val * 255).astype(np.uint8)
    else:
        recon_uint8 = np.zeros_like(recon_amp, dtype=np.uint8)

    pil_img = Image.fromarray(recon_uint8)
    pil_img = pil_img.resize((256, 256), Image.BILINEAR)
    final_amp = np.array(pil_img).astype(np.float32)  # [0,1]

    # 9. 构建模型输入张量 (1,256,256,1)
    input_tensor = final_amp[np.newaxis, ..., np.newaxis]

    return input_tensor


# ==================== 初始化 CSV 文件并写入表头 ====================
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'true_value_scaled', 'predicted_scaled',
                     'true_value_original', 'error'])

# ==================== 执行预测并实时保存 ====================
predictions_buffer = []
batch_size_log = 100  # 每处理 100 个样本写入一次

for i, fname in enumerate(x_test):
    fx = fx_ref_test[i]
    fy = fy_ref_test[i]
    full_path = os.path.join(data_dir, fname)
    input_data = preprocess_image(full_path, fx, fy)
    pred = model.predict(input_data, verbose=0)[0, 0]  # 假设输出为标量

    # 准备待写入的一行数据
    true_original = y_test[i] / 1e5
    pred_original = pred / 1e5
    error = abs(true_original - pred_original)
    predictions_buffer.append([
        fname,
        y_test[i],
        pred,
        true_original,
        error
    ])

    # 每 batch_size_log 个样本或最后一个样本时写入文件
    if (i + 1) % batch_size_log == 0 or (i + 1) == len(x_test):
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(predictions_buffer)
        predictions_buffer = []  # 清空缓冲区
        print(f"已处理并保存 {i + 1}/{len(x_test)} 个样本")

print(f"预测结果已实时保存至 {output_csv}")
