from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import os

tf.config.optimizer.set_experimental_options({"disable_loop_optimization": True})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== 1. GPU 配置 ====================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"已为 {len(physical_devices)} 个 GPU 启用显存增长")
else:
    print("未检测到GPU，将使用CPU")

strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
print(f"使用 MirroredStrategy，跨设备通信方式: HierarchicalCopyAllReduce")

# ==================== 2. 导入自定义模块 ====================
from comparison_method.FocusNet.helpers import Rotate90Randomly, Fourier2D, Scheduler, RegressionSequence_exp

# ==================== 3. 超参数设置 ====================
name = '2headed_extra_layer_log_abs'
epochs = 150
batch_size = 64
val_batch_size = 1
changing_epoch = 120
changing_period = 30
learning_rate = 0.0005
decay = 0.1
factor = 4
dropout = 0.05
title = f'{name}_e{epochs}_ce{changing_epoch}_lr{learning_rate}_d{decay}_bs{batch_size}_' \
        f'dr{dropout}_cp_{changing_period}_x{factor}_exp256'
data_dir = r'F:\dongjiayao\Data\AutoFocusDatabase\hologram_amplitude\hologram'

# ==================== 继续训练配置 ====================
RESUME = False                          # True=继续训练，False=从头训练
resume_checkpoint = f'../models/{title}_cp.keras'   # 要恢复的检查点路径
hist_csv_file = f'../hist/{title}.csv'

# ==================== 4. 加载数据 ====================
df = pd.read_csv(r'F:\dongjiayao\Data\AutoFocusDatabase/AutoFocusDatabase_amplitude.csv')
x_set = df['hologram_name'].tolist()
y_set = (df['z'].values * 1e5).tolist()

# 第一步：切出 test 15%
x_trainval, x_test, y_trainval, y_test = train_test_split(
    x_set, y_set, test_size=0.15, random_state=42)

# 第二步：从剩余中切出 val（15/85 ≈ 17.6%，使得最终val占总体15%）
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.15/0.85, random_state=42)

print(f"train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

# 把 test 集的文件名和标签保存下来，供测试脚本使用
test_df = pd.DataFrame({'hologram_name': x_test, 'y_true': y_test})
os.makedirs('../hist', exist_ok=True)
test_df.to_csv(f'../hist/{title}_test_split.csv', index=False)
print(f"test split 已保存: ../hist/{title}_test_split.csv")

# ==================== 5. 创建数据序列 ====================
sequence_train = RegressionSequence_exp(x_train, y_train, [2025, 2148, 466, 180], data_dir, batch_size)
sequence_val   = RegressionSequence_exp(x_val,   y_val,   [2025, 2148, 466, 180], data_dir, val_batch_size)

# ==================== 6. 构建或恢复模型 ====================
# 计算已完成的 epoch 数（用于 initial_epoch 和学习率恢复）
initial_epoch = 0

if RESUME and os.path.exists(resume_checkpoint):
    print(f"\n[Resume] 从检查点恢复: {resume_checkpoint}")

    # 读取已有历史，推断已训练的 epoch 数
    if os.path.exists(hist_csv_file):
        hist_df_old = pd.read_csv(hist_csv_file)
        initial_epoch = len(hist_df_old)
        print(f"[Resume] 检测到历史记录，已完成 {initial_epoch} 个 epoch，"
              f"将从第 {initial_epoch + 1} 个 epoch 继续")
    else:
        print("[Resume] 未找到历史 CSV，initial_epoch 设为 0，学习率将从头计算")

    # 在 strategy 作用域内加载，权重自动分发到各 GPU
    with strategy.scope():
        model = tf.keras.models.load_model(
            resume_checkpoint,
            custom_objects={
                'Rotate90Randomly': Rotate90Randomly,
                'Fourier2D': Fourier2D,
            }
        )
        # 恢复学习率：按已训练 epoch 重放调度器
        scheduler_tmp = Scheduler(changing_period, changing_epoch)
        restored_lr = learning_rate
        for ep in range(initial_epoch):
            restored_lr = scheduler_tmp.schedule(ep, restored_lr)
        tf.keras.backend.set_value(model.optimizer.lr, restored_lr)
        print(f"[Resume] 学习率恢复为: {restored_lr:.2e}")

else:
    if RESUME:
        print(f"[Resume] 未找到检查点 {resume_checkpoint}，将从头开始训练")

    with strategy.scope():
        model = Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(256, 256, 1), dtype=tf.complex64),
            Rotate90Randomly(),
            Fourier2D(),
            layers.Conv2D(8  * factor, 7, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(16 * factor, 5, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(32 * factor, 5, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64 * factor, 3, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128 * factor, 3, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(256 * factor, 3, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(dropout),
            layers.Flatten(),
            layers.Dense(32 * factor, activation='swish'),
            layers.Dense(32 * factor, activation='swish'),
            layers.Dense(32 * factor, activation='swish'),
            layers.Dense(1)
        ])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=['mae']
        )

model.summary()

# ==================== 7. 回调函数 ====================
scheduler  = Scheduler(changing_period, changing_epoch)
checkpoint_path = f'../models/{title}_cp.keras'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True
)


# 继续训练时，追加写历史 CSV；从头训练时，覆盖写
class HistoryAppendCallback(tf.keras.callbacks.Callback):
    """将每个 epoch 的指标实时追加到 CSV，支持断点续训后历史拼接。"""
    def __init__(self, csv_path, append=False):
        super().__init__()
        self.csv_path = csv_path
        self.append   = append   # True=追加，False=覆盖

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row  = pd.DataFrame([logs])
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not self.append or not os.path.exists(self.csv_path):
            row.to_csv(self.csv_path, index=False)
            self.append = True   # 第一次写完后改为追加模式
        else:
            row.to_csv(self.csv_path, mode='a', header=False, index=False)

hist_callback = HistoryAppendCallback(
    csv_path=hist_csv_file,
    append=(RESUME and initial_epoch > 0)
)

# ==================== 8. 训练 ====================
print(f"\n[Train] initial_epoch={initial_epoch}  目标 epochs={epochs}")
history = model.fit(
    sequence_train,
    epochs=epochs,
    initial_epoch=initial_epoch,      # 告诉 Keras 从哪个 epoch 开始计数
    validation_data=sequence_val,
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(scheduler.schedule),
        cp_callback,
        hist_callback,                # 实时追加 CSV，替代训练后统一保存
    ]
)

# ==================== 9. 保存模型 ====================
os.makedirs(f'../models/{title}', exist_ok=True)
model.save(f'../models/{title}')
print("训练完成，模型已保存。")
