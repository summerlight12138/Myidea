import os
import time
import subprocess
import signal
import sys
import getpass

# ================= 配置区域 =================
# 显存警戒线 (MiB)，你的显卡是 49140 MiB，我们设得保守一点，到了 47500 就开始杀
MEMORY_THRESHOLD = 47500 
# 检查间隔 (秒)
CHECK_INTERVAL = 2
# 我们要保护的脚本关键词（只杀包含这个词的进程）
TARGET_SCRIPT = "qwen2_5_vl_query.py"
# 指定 python 解释器路径特征，防止误杀别人的 python
MY_PYTHON_PATH = "/data/XL/Myidea/bin/python"
# ===========================================

def log(message):
    """打印带时间的日志"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_gpu_memory_usage():
    """
    获取当前显存使用情况
    返回: 一个列表，包含每个 GPU 的已用显存 (MiB)
    例如: [24000, 48500]
    """
    try:
        # 使用 nvidia-smi 获取显存信息
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # 将输出转换为整数列表
        usages = [int(x.strip()) for x in result.strip().split('\n')]
        return usages
    except Exception as e:
        log(f"获取显存信息失败: {e}")
        return []

def get_target_processes():
    """
    获取当前用户运行的、符合条件的目标进程
    返回: 进程列表，每个元素是 (pid, cmdline)
    """
    current_user = getpass.getuser()
    target_procs = []
    
    try:
        # 获取所有进程的详细信息: user, pid, cmd
        # -ww 避免输出被截断
        ps_output = subprocess.check_output(['ps', '-ef', '-ww'], encoding='utf-8')
        
        for line in ps_output.split('\n'):
            parts = line.split()
            if len(parts) < 8:
                continue
                
            user = parts[0]
            pid = int(parts[1])
            cmdline = " ".join(parts[7:])
            
            # 筛选条件：
            # 1. 必须是当前用户
            # 2. 必须包含我们的目标脚本名
            # 3. 必须是我们指定的 Python 环境（双重保险）
            if user == current_user and TARGET_SCRIPT in cmdline and MY_PYTHON_PATH in cmdline:
                # 排除掉监控脚本自己（如果监控脚本也包含相同关键词的话，这里 monitor_oom 不包含 qwen 所以没事）
                target_procs.append((pid, cmdline))
                
    except Exception as e:
        log(f"获取进程信息失败: {e}")
        
    return target_procs

def kill_process(pid):
    """优雅地结束进程"""
    try:
        log(f"正在终止进程 PID: {pid} ...")
        os.kill(pid, signal.SIGTERM) # 先尝试发 SIGTERM 让它自己退出
        time.sleep(1)
        # 检查是否还在
        try:
            os.kill(pid, 0)
            log(f"进程 {pid} 还在，强制查杀 (SIGKILL)...")
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass # 进程已经没了
        log(f"进程 {pid} 已清理。")
    except Exception as e:
        log(f"查杀进程 {pid} 失败: {e}")

def main():
    log("=== 显存熔断保护助手已启动 ===")
    log(f"监控阈值: {MEMORY_THRESHOLD} MiB")
    log(f"目标脚本: {TARGET_SCRIPT}")
    log(f"当前用户: {getpass.getuser()}")
    
    while True:
        try:
            # 1. 检查显存
            gpu_usages = get_gpu_memory_usage()
            max_usage = max(gpu_usages) if gpu_usages else 0
            
            # 2. 如果显存超标
            if max_usage > MEMORY_THRESHOLD:
                log(f"⚠️ 警告！发现显存占用 ({max_usage} MiB) 超过阈值 ({MEMORY_THRESHOLD} MiB)！")
                
                # 3. 获取我们的任务进程
                procs = get_target_processes()
                log(f"当前运行的任务进程数: {len(procs)}")
                
                # 4. 只有当进程数大于 1 时才执行查杀（保留最后一个火种）
                if len(procs) > 1:
                    # 策略：杀掉 PID 最大的那个（通常是最后启动的）
                    # 这样可以保护运行最久的那个任务
                    procs.sort(key=lambda x: x[0], reverse=True)
                    victim_pid, victim_cmd = procs[0]
                    
                    log(f"决定牺牲最近启动的任务: PID {victim_pid}")
                    log(f"任务详情: {victim_cmd[:100]}...")
                    
                    kill_process(victim_pid)
                    
                    # 杀完后休息一会儿，给显存释放的时间，避免连续误杀
                    log("等待 10 秒让显存释放...")
                    time.sleep(10)
                else:
                    log("当前只剩一个任务在运行，不再查杀，听天由命...")
                    # 这种情况下如果还是 OOM，那就没办法了，但我们已经尽力保住了一个
                    
            else:
                # 显存正常，什么都不做
                pass
                
        except Exception as e:
            log(f"监控循环出错: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
