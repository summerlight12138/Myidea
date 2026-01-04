import os
import subprocess
import sys


def main():
    """小白版说明：一键顺序跑完 stage3_01 到 stage3_05，方便你整体查看 PatchCore 专家的效果和可视化。"""
    root = os.path.dirname(os.path.dirname(__file__))
    python_exec = sys.executable
    scripts = [
        "stage3_01_make_subset.py",
        "stage3_02_build_bank.py",
        "stage3_03_infer_evidence.py",
        "stage3_04_eval_patchcore.py",
        "stage3_05_make_overlay.py",
    ]
    for name in scripts:
        script_path = os.path.join(root, "scripts", name)
        print(f"=== Running {name} ===")
        subprocess.run([python_exec, script_path], check=True)
    print("=== Stage3 全流程执行完成 ===")


if __name__ == "__main__":
    main()
