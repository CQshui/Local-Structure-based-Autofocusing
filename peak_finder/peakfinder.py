import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


class PeakFinder:
    def __init__(self, func, lb=-0.00060, ub=0.00060, max_evals=50, target='min',
                 precision_factor=1.0, initial_points_factor=1.0, scale=None):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.max_evals = max_evals
        self.target = target
        self.x_history = []
        self.y_history = []
        self.eval_count = 0
        self.phase = 'exploration'
        self.precision_factor = precision_factor
        self.initial_points_factor = initial_points_factor
        self.scale = scale
        self.success = False  # 寻峰成功标志

    def evaluate(self, x):
        """简化的评估函数 - 直接clip，不施加额外惩罚"""
        if self.eval_count >= self.max_evals:
            return float('inf') if self.target == 'min' else float('-inf')

        # 检查重复
        for i, prev_x in enumerate(self.x_history):
            if abs(prev_x - x) < 1e-12:
                return self.y_history[i]

        # clip到边界
        x = np.clip(x, self.lb, self.ub)
        y = self.func(x, scale=self.scale)

        self.x_history.append(x)
        self.y_history.append(y)
        self.eval_count += 1
        return y

    def get_current_best(self):
        """获取当前最佳点"""
        if not self.x_history:
            return None, None

        if self.target == 'min':
            best_idx = np.argmin(self.y_history)
        else:
            best_idx = np.argmax(self.y_history)

        return self.x_history[best_idx], self.y_history[best_idx]

    def build_surrogate_model(self):
        """构建代理模型 - 使用所有点，不过滤"""
        if len(self.x_history) < 4:
            return None

        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)

        sorted_indices = np.argsort(x_array)
        x_sorted = x_array[sorted_indices]
        y_sorted = y_array[sorted_indices]

        try:
            return PchipInterpolator(x_sorted, y_sorted, extrapolate=False)
        except Exception:
            return None

    def find_surrogate_optimum(self, surrogate_model):
        """在代理模型上寻找最优点"""
        if surrogate_model is None:
            return None

        try:
            if self.target == 'min':
                objective = lambda x: surrogate_model(x)
            else:
                objective = lambda x: -surrogate_model(x)

            result = minimize_scalar(
                objective,
                bounds=(self.lb, self.ub),
                method='bounded',
                options={'xatol': 1e-11}
            )

            if result.success:
                return result.x
            else:
                return None

        except Exception:
            return None

    def acquisition_function(self, x_candidates):
        """改进的采集函数 - 更平衡的权重"""
        current_best_x, current_best_y = self.get_current_best()
        if current_best_x is None:
            return np.zeros(len(x_candidates))

        scores = []
        range_width = self.ub - self.lb

        for x in x_candidates:
            distance_to_best = abs(x - current_best_x) / range_width
            score = 1 - distance_to_best
            scores.append(score)

        return scores

    def adaptive_sampling_strategy(self):
        """优化的自适应采样策略"""
        current_best_x, _ = self.get_current_best()
        candidates = []

        if self.phase == 'exploration':
            n_points = max(15, int(20 * self.initial_points_factor))
            n_uniform = int(n_points * 0.7)
            n_centered = n_points - n_uniform

            uniform_samples = np.random.uniform(self.lb, self.ub, n_uniform)
            center = (self.lb + self.ub) / 2
            std = (self.ub - self.lb) / 6
            centered_samples = np.random.normal(center, std, n_centered)
            centered_samples = np.clip(centered_samples, self.lb, self.ub)

            candidates = np.concatenate([uniform_samples, centered_samples])

        else:
            surrogate = self.build_surrogate_model()

            if surrogate and current_best_x is not None:
                opt = self.find_surrogate_optimum(surrogate)
                if opt is not None:
                    candidates.append(opt)

                radius = 0.15 * (self.ub - self.lb)
                n_points = max(10, int(12 * self.initial_points_factor))
                for _ in range(n_points):
                    candidate = current_best_x + radius * np.random.uniform(-1, 1)
                    candidates.append(candidate)

                if len(self.y_history) >= 5:
                    y_array = np.array(self.y_history)
                    if self.target == 'min':
                        top_indices = np.argsort(y_array)[:3]
                    else:
                        top_indices = np.argsort(y_array)[-3:]

                    for idx in top_indices[1:]:
                        x_secondary = self.x_history[idx]
                        if abs(x_secondary - current_best_x) > 0.1 * (self.ub - self.lb):
                            for _ in range(3):
                                candidate = x_secondary + 0.05 * (self.ub - self.lb) * np.random.uniform(-1, 1)
                                candidates.append(candidate)

            if not candidates:
                n_points = max(10, int(12 * self.initial_points_factor))
                candidates = np.random.uniform(self.lb, self.ub, n_points).tolist()

        candidates = [np.clip(c, self.lb, self.ub) for c in candidates]

        if len(candidates) > 0:
            scores = self.acquisition_function(candidates)
            best_idx = np.argmax(scores)
            return candidates[best_idx]
        else:
            return (self.lb + self.ub) / 2

    def update_phase(self):
        n_evals = self.eval_count
        if n_evals < self.max_evals * 0.35:
            self.phase = 'exploration'
        elif n_evals < self.max_evals * 0.75:
            self.phase = 'exploitation'
        else:
            self.phase = 'refinement'

    # ========================
    # ★ V1改进：代理模型验证
    # ========================
    def calculate_surrogate_fitting_error(self, surrogate_model, error_metric='rmse'):
        """
        计算代理模型与实际函数的拟合误差

        Args:
            surrogate_model: PCHIP代理模型
            error_metric: 误差度量方式
                - 'rmse': 均方根误差
                - 'mae': 平均绝对误差
                - 'max': 最大误差
                - 'relative': 相对误差

        Returns:
            error_value: 误差值
            error_ratio: 误差相对于函数值范围的比例
        """
        if surrogate_model is None or len(self.y_history) < 4:
            return float('inf'), 1.0

        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)

        # 预测值
        y_pred = surrogate_model(x_array)

        # 计算误差
        errors = np.abs(y_array - y_pred)

        # 选择误差度量方式
        if error_metric == 'rmse':
            error_value = np.sqrt(np.mean(errors ** 2))
        elif error_metric == 'mae':
            error_value = np.mean(errors)
        elif error_metric == 'max':
            error_value = np.max(errors)
        elif error_metric == 'relative':
            # 相对误差：相对于函数值的范围
            y_range = np.max(y_array) - np.min(y_array)
            if y_range < 1e-12:
                error_value = np.max(errors)
            else:
                error_value = np.max(errors) / y_range
        else:
            error_value = np.mean(errors)

        # 计算误差相对于函数值范围的比例
        y_range = np.max(y_array) - np.min(y_array)
        if y_range < 1e-12:
            error_ratio = error_value / (1e-12) if error_value > 0 else 0
        else:
            error_ratio = error_value / y_range

        return error_value, error_ratio

    def judge_success(self, surrogate_error_threshold=0.15):
        """
        综合判定寻峰是否成功：
        1. 代理模型拟合误差（主要指标）
        2. 数据充分性

        Args:
            surrogate_error_threshold: 代理模型误差阈值（相对值）
                如果误差超过此阈值，判定为失败
                典型值：0.1 ~ 0.2

        Returns:
            success: 布尔值
            reason: 失败原因或成功信息
            metrics: 诊断信息字典
        """
        f_values = np.array(self.y_history)
        x_values = np.array(self.x_history)
        surrogate = self.build_surrogate_model()

        metrics = {
            'eval_count': self.eval_count,
            'data_points': len(f_values),
            'surrogate_error': None,
            'error_ratio': None,
            'y_range': np.max(f_values) - np.min(f_values)
        }

        # 检查1：数据充分性
        if len(f_values) < 10:
            return False, "数据点不足（<10个）", metrics

        # 检查2：代理模型是否构建成功
        if surrogate is None:
            return False, "代理模型构建失败", metrics

        # 检查3：代理模型拟合误差（★核心检查）
        error_value, error_ratio = self.calculate_surrogate_fitting_error(
            surrogate,
            error_metric='max'
        )

        metrics['surrogate_error'] = error_value
        metrics['error_ratio'] = error_ratio

        if error_ratio > surrogate_error_threshold:
            return False, f"代理模型拟合误差过大 (误差比: {error_ratio:.4f} > {surrogate_error_threshold})", metrics

        # 检查4：函数值范围是否太小（可能是噪声）
        if metrics['y_range'] < 1e-12:
            return False, "函数值范围过小，可能是噪声", metrics

        return True, "寻峰成功", metrics

    # ========================

    def find_optimum(self):
        """主搜索算法"""
        initial_points_num = max(5, int(7 * self.initial_points_factor))

        # 初始分段采样
        for i in range(initial_points_num):
            segment_start = self.lb + i * (self.ub - self.lb) / initial_points_num
            segment_end = self.lb + (i + 1) * (self.ub - self.lb) / initial_points_num
            point = np.random.uniform(segment_start, segment_end)
            self.evaluate(point)

        # 添加中心点
        self.evaluate((self.lb + self.ub) / 2)

        stagnation_count = 0
        prev_best_y = float('inf') if self.target == 'min' else float('-inf')

        while self.eval_count < self.max_evals:
            self.update_phase()
            current_best_x, current_best_y = self.get_current_best()
            next_x = self.adaptive_sampling_strategy()
            self.evaluate(next_x)

            if current_best_y is not None:
                improvement = abs(current_best_y - prev_best_y)
                convergence_threshold = 1e-4 / self.precision_factor
                if improvement < convergence_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                prev_best_y = current_best_y
                if stagnation_count >= 8:
                    break

        final_x, final_y = self.get_current_best()

        # ★ V1改进：综合判定 success（使用代理模型验证）
        self.success, reason, metrics = self.judge_success(surrogate_error_threshold=0.15)

        # return final_x, self.success
        return final_x


# ========================
# 测试用例
# ========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 测试用例1：单峰函数（应该成功）
    print("=" * 70)
    print("测试1：单峰函数 - (x - 0.0002)^2")
    print("=" * 70)


    def single_peak(x, scale=None):
        return (x - 0.0002) ** 2


    finder1 = PeakFinder(
        func=single_peak,
        lb=-0.0006,
        ub=0.0006,
        max_evals=50,
        target='min'
    )

    optimal_x1, success1 = finder1.find_optimum()
    _, _, metrics1 = finder1.judge_success()

    print(f"最优点: x = {optimal_x1:.8f}")
    print(f"最优值: {finder1.get_current_best()[1]:.8f}")
    print(f"成功: {success1}")
    print(f"诊断信息: {metrics1}")

    # 测试用例2：多峰函数（应该失败）
    print("\n" + "=" * 70)
    print("测试2：多峰函数 - sin(20*x) + 0.1*x^2")
    print("=" * 70)


    def multi_peak(x, scale=None):
        return np.sin(20 * x) + 0.1 * x ** 2


    finder2 = PeakFinder(
        func=multi_peak,
        lb=-0.0006,
        ub=0.0006,
        max_evals=50,
        target='min'
    )

    optimal_x2, success2 = finder2.find_optimum()
    _, _, metrics2 = finder2.judge_success()

    print(f"最优点: x = {optimal_x2:.8f}")
    print(f"最优值: {finder2.get_current_best()[1]:.8f}")
    print(f"成功: {success2}")
    print(f"诊断信息: {metrics2}")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (finder, func, title) in enumerate([
        (finder1, single_peak, "单峰函数 (成功预期)"),
        (finder2, multi_peak, "多峰函数 (失败预期)")
    ]):
        ax = axes[idx]

        # 绘制真实函数
        x_plot = np.linspace(finder.lb, finder.ub, 200)
        y_plot = [func(x) for x in x_plot]
        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='真实函数', alpha=0.7)

        # 绘制采样点
        ax.scatter(finder.x_history, finder.y_history, color='red', s=50,
                   label=f'采样点({len(finder.x_history)}个)', zorder=5)

        # 绘制代理模型
        surrogate = finder.build_surrogate_model()
        if surrogate is not None:
            y_surrogate = surrogate(x_plot)
            ax.plot(x_plot, y_surrogate, 'g--', linewidth=2, label='代理模型', alpha=0.7)

        # 标记最优点
        best_x, best_y = finder.get_current_best()
        ax.plot(best_x, best_y, 'r*', markersize=15, label=f'最优点', zorder=10)

        ax.set_title(f"{title}\n成功: {finder.success}")
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('peak_finder_v1_test.png', dpi=100, bbox_inches='tight')
    print("\n图表已保存为 peak_finder_v1_test.png")
    plt.show()
