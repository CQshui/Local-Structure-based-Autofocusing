import numpy as np
import tensorflow as tf
import pandas as pd
import os

tf.config.optimizer.set_experimental_options({"disable_loop_optimization": True})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from comparison_method.FocusNet.helpers import Rotate90Randomly, Fourier2D, RegressionSequence_exp

# ==================== 配置 ====================
name            = '2headed_extra_layer_log_abs'
epochs          = 150
batch_size      = 64
changing_epoch  = 120
changing_period = 30
learning_rate   = 0.0005
decay           = 0.1
factor          = 4
dropout         = 0.05
title = (f'{name}_e{epochs}_ce{changing_epoch}_lr{learning_rate}_d{decay}_bs{batch_size}_'
         f'dr{dropout}_cp_{changing_period}_x{factor}_exp256')

data_dir       = r'F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\hologram'
model_path     = f'../models/{title}_cp.keras'
test_split_csv = f'../hist/{title}_test_split.csv'
output_csv     = f'../hist/{title}_test_results_use_pascal256model.csv'

# ==================== 加载 test split ====================
assert os.path.exists(test_split_csv), f"找不到 test split 文件: {test_split_csv}"
test_df  = pd.read_csv(test_split_csv)
x_test   = test_df['hologram_name'].tolist()
y_test   = test_df['y_true'].tolist()
print(f"测试集样本数: {len(x_test)}")

# ==================== 加载模型 ====================
print(f"加载模型: {model_path}")
model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'Rotate90Randomly': Rotate90Randomly,
        'Fourier2D':        Fourier2D,
    }
)
model.summary()

# ==================== 构建测试序列（batch_size=1，保证顺序对齐）====================
sequence_test = RegressionSequence_exp(
    x_test, y_test, [2025, 2148, 466, 180], data_dir, b_size=1
)

# ==================== 逐样本推理 ====================
y_pred_list = []
y_true_list = []

print("开始推理...")
for i in range(len(sequence_test)):
    batch_x, batch_y = sequence_test[i]
    pred = model.predict(batch_x, verbose=0)          # shape: (1, 1)
    y_pred_list.append(float(pred[0, 0]))
    y_true_list.append(float(batch_y[0]))

    if (i + 1) % 100 == 0:
        print(f"  已完成 {i+1}/{len(sequence_test)}")

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

# ==================== 计算指标 ====================
mae  = np.mean(np.abs(y_pred - y_true))
mse  = np.mean((y_pred - y_true) ** 2)
rmse = np.sqrt(mse)

# 计算 RSQ = Pearson 相关系数的平方
corr = np.corrcoef(y_true, y_pred)[0, 1]
rsq  = corr ** 2

print(f"\n========== 测试结果 ==========")
print(f"样本数 : {len(y_true)}")
print(f"MAE    : {mae:.4f}")
print(f"RMSE   : {rmse:.4f}")
print(f"MSE    : {mse:.4f}")
print(f"RSQ    : {rsq:.4f}")   # 即 Excel 的 RSQ 函数结果
print(f"==============================\n")

# ==================== 保存结果 CSV ====================
result_df = pd.DataFrame({
    'hologram_name': x_test,
    'y_true':        y_true,
    'y_pred':        y_pred,
    'error':         y_pred - y_true,
    'abs_error':     np.abs(y_pred - y_true),
})

# 附上汇总指标作为最后一行
summary = pd.DataFrame([{
    'hologram_name': 'SUMMARY',
    'y_true':  np.nan,
    'y_pred':  np.nan,
    'error':   np.nan,
    'abs_error': np.nan,
    'MAE':  mae,
    'RMSE': rmse,
    'MSE':  mse,
    'R2':   rsq,
}])

result_df = pd.concat([result_df, summary], ignore_index=True)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
result_df.to_csv(output_csv, index=False)
print(f"测试结果已保存: {output_csv}")
