import time
import subprocess
import os

# ================= 配置 =================
# 我们要等待运行的命令
NEXT_COMMAND = [
    "/data/XL/Myidea/bin/python",
    "qwen2_5_vl_query.py",
    "--model-path", "/data/XL/MMAD_Base/models/Qwen2.5-VL-7B-Instruct",
    "--few_shot_model", "2",
    "--similar_template",
    "--reproduce"
]
# 目标脚本关键词
TARGET_SCRIPT = "qwen2_5_vl_query.py"
# 最大并发数（当检测到运行的任务少于这个数时，就启动新任务）
MAX_CONCURRENT = 2
# 检查间隔 (秒)
CHECK_INTERVAL = 60
# =======================================

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def count_running_tasks():
    """统计当前正在运行的目标任务数"""
    try:
        # 使用 ps -ef | grep ... 统计
        # 注意要排除 grep 自身和这个排队脚本自身（如果它也包含关键词的话，不过这里 TARGET_SCRIPT 比较独特）
        cmd = f"ps -ef | grep {TARGET_SCRIPT} | grep -v grep | wc -l"
        result = subprocess.check_output(cmd, shell=True, encoding='utf-8')
        return int(result.strip())
    except Exception as e:
        log(f"检查进程数出错: {e}")
        return 999 # 出错时假设满载，不乱动

def main():
    log("=== 自动排队助手已启动 ===")
    log(f"待执行任务: {' '.join(NEXT_COMMAND)}")
    log(f"等待空位 (当前最大并发: {MAX_CONCURRENT})...")

    while True:
        count = count_running_tasks()
        log(f"当前运行任务数: {count}")

        if count < MAX_CONCURRENT:
            log("发现空位！正在启动下一项任务...")
            try:
                # 启动新任务
                # 使用 subprocess.run 运行，这样排队脚本会等待它结束（或者我们可以用 Popen 让它后台跑，但这里等待也无所谓，因为只有这一个任务要排）
                # 既然是排队，我们直接替换当前进程，或者作为子进程运行
                subprocess.run(NEXT_COMMAND, cwd="/data/XL/MMAD_Base/MMAD-main/evaluation/examples/Transformers")
                log("任务已完成！排队助手退出。")
                break
            except Exception as e:
                log(f"启动任务失败: {e}")
                break
        else:
            log("队列已满，继续等待...")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
