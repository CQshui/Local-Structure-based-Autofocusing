import subprocess
import threading
import time
import os
import sys

# ==================== 配置区域 ====================

# ★ 关键：指定项目根目录和 Python 解释器
PROJECT_ROOT = r"F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing-upload"          # 项目根目录
PYTHON_EXE   = sys.executable

SCRIPT = r"F:\dongjiayao\Pycharm\Local-Structure-based-Autofocusing-upload\test\COCO\
test_singlePlane_COCO_comparison_tasks.py"

task_queues = {
    0: [
        [PYTHON_EXE, SCRIPT,
         "--gpu", "0",
         "--method", "eigen",
         "--root_dir", r"F:\dongjiayao\Data\COCO\val\holograms",
         "--save_dir", r"F:\dongjiayao\Data\COCO\val\result\eigen",
         "--test_csv", r"F:\dongjiayao\Data\COCO\val\holograms\AutoFocusDatabase.csv"],
        ],
    1: [
        ],
    2: [
        ],
    3: [
        ],
}

# =================================================


def run_task_on_gpu(gpu_id, command):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONPATH'] = PROJECT_ROOT

    n = max(1, (os.cpu_count() or 16) // len(task_queues))
    env['OMP_NUM_THREADS']      = str(n)
    env['MKL_NUM_THREADS']      = str(n)
    env['OPENBLAS_NUM_THREADS'] = str(n)
    env['NUMEXPR_NUM_THREADS']  = str(n)

    print(f"[GPU {gpu_id}] 开始执行: {' '.join(command)}  (每进程 {n} 线程)")
    start_time = time.time()

    try:
        proc = subprocess.Popen(command, env=env, cwd=PROJECT_ROOT)
        proc.wait()
        elapsed = time.time() - start_time
        if proc.returncode == 0:
            print(f"[GPU {gpu_id}] ✅ 完成 (耗时 {elapsed:.1f}s)")
            return True
        else:
            print(f"[GPU {gpu_id}] ❌ 失败 (退出码 {proc.returncode}, 耗时 {elapsed:.1f}s)")
            return False
    except Exception as e:
        print(f"[GPU {gpu_id}] ⚠️ 异常: {e}")
        return False


def gpu_worker(gpu_id, task_list):
    print(f"[GPU {gpu_id}] 队列启动，共 {len(task_list)} 个任务")
    for idx, cmd in enumerate(task_list, 1):
        print(f"[GPU {gpu_id}] 任务 [{idx}/{len(task_list)}] 准备运行")
        success = run_task_on_gpu(gpu_id, cmd)
        if not success:
            print(f"[GPU {gpu_id}] 任务失败，继续执行下一个任务...")
    print(f"[GPU {gpu_id}] 队列全部任务执行完毕。")


def main():
    print(f"解释器: {PYTHON_EXE}")
    print(f"项目根: {PROJECT_ROOT}\n")

    threads = []
    for gpu_id, tasks in task_queues.items():
        if not tasks:
            print(f"[GPU {gpu_id}] 队列为空，跳过")
            continue
        t = threading.Thread(target=gpu_worker, args=(gpu_id, tasks))
        t.start()
        threads.append(t)
        time.sleep(0.5)

    for t in threads:
        t.join()

    print("\n🎉 所有 GPU 队列任务全部完成！")


if __name__ == "__main__":
    main()
