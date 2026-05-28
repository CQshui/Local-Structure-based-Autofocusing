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
from comparison_method.FocusNet.helpers import RegressionSequence, Rotate90Randomly, Fourier2D, Scheduler

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
        f'dr{dropout}_cp_{changing_period}_x{factor}_pascal256'
data_dir = r'F:\dongjiayao\Data\PASCAL\phase_amp_train\0.0_0_256\amplitude'

# ==================== 新增：继续训练配置 ====================
RESUME = True
resume_checkpoint = f'../models/{title}_cp.keras'
hist_csv_file = f'../hist/{title}.csv'

# ==================== 4. 加载数据 ====================
df = pd.read_csv(f'{data_dir}/AutoFocusDatabase.csv')
x_set  = df['hologram_name'].tolist()
y_set  = (df['z'].values * 1e5).tolist()
fx_ref = df['fx_ref'].tolist()
fy_ref = df['fy_ref'].tolist()

x_train, x_test, y_train, y_test, fx_ref_train, fx_ref_test, fy_ref_train, fy_ref_test = (
    train_test_split(x_set, y_set, fx_ref, fy_ref, test_size=0.2, random_state=42))

# ==================== 5. 创建数据序列 ====================
sequence_train = RegressionSequence(x_train, y_train, fx_ref_train, fy_ref_train, data_dir, batch_size)
sequence_val   = RegressionSequence(x_test,  y_test,  fx_ref_test,  fy_ref_test,  data_dir, val_batch_size)

# ==================== 6. 构建或恢复模型 ====================
initial_epoch = 0

if RESUME and os.path.exists(resume_checkpoint):
    print(f"\n[Resume] 从检查点恢复: {resume_checkpoint}")

    if os.path.exists(hist_csv_file):
        hist_df_old   = pd.read_csv(hist_csv_file)
        initial_epoch = len(hist_df_old)
        print(f"[Resume] 已完成 {initial_epoch} 个 epoch，从第 {initial_epoch + 1} 个继续")
    else:
        print("[Resume] 未找到历史 CSV，initial_epoch 设为 0，学习率将从头计算")

    with strategy.scope():
        model = tf.keras.models.load_model(
            resume_checkpoint,
            custom_objects={
                'Rotate90Randomly': Rotate90Randomly,
                'Fourier2D': Fourier2D,
            }
        )
        # 重放调度器，恢复到中断时的学习率
        scheduler_tmp = Scheduler(changing_period, changing_epoch)
        restored_lr   = learning_rate
        for ep in range(initial_epoch):
            restored_lr = scheduler_tmp.schedule(ep, restored_lr)
        tf.keras.backend.set_value(model.optimizer.lr, restored_lr)
        print(f"[Resume] 学习率恢复为: {restored_lr:.2e}")

else:
    if RESUME:
        print(f"[Resume] 未找到检查点 {resume_checkpoint}，从头开始训练")

    with strategy.scope():
        model = Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(256, 256, 1), dtype=tf.complex64),
            Rotate90Randomly(),
            Fourier2D(),
            layers.Conv2D(8   * factor, 7, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(16  * factor, 5, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(32  * factor, 5, padding='same', activation='swish'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64  * factor, 3, padding='same', activation='swish'),
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
scheduler = Scheduler(changing_period, changing_epoch)
checkpoint_path = f'../models/{title}_cp.keras'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True
)


class HistoryAppendCallback(tf.keras.callbacks.Callback):
    """逐 epoch 追加写 CSV，续训后历史连续不覆盖。"""
    def __init__(self, csv_path, append=False):
        super().__init__()
        self.csv_path = csv_path
        self.append   = append

    def on_epoch_end(self, epoch, logs=None):
        row = pd.DataFrame([logs or {}])
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not self.append or not os.path.exists(self.csv_path):
            row.to_csv(self.csv_path, index=False)
            self.append = True
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
    initial_epoch=initial_epoch,
    validation_data=sequence_val,
    callbacks=[
        tf.keras.callbacks.LearningRateScheduler(scheduler.schedule),
        cp_callback,
        hist_callback,
    ]
)

# ==================== 9. 保存模型 ====================
os.makedirs(f'../models/{title}', exist_ok=True)
model.save(f'../models/{title}')
print("训练完成，模型已保存。")
