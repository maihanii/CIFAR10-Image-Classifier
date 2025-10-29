import os
import subprocess

def run_script(script_path):
    print(f"\n🚀 Running: {script_path}")
    result = subprocess.run(["python", script_path], shell=True)
    if result.returncode == 0:
        print(f"✅ Finished: {script_path}\n")
    else:
        print(f"❌ Error while running {script_path}\n")

if __name__ == "__main__":
    print("🧠 CIFAR-10 Deep Learning Project Launcher\n")

    steps = [
        "src/eda.py",
        "src/train.py",
        "src/evaluate.py",
        "src/gui.py"
    ]

    for step in steps:
        if os.path.exists(step):
            run_script(step)
        else:
            print(f"⚠️ File not found: {step}")

    print("🎉 All steps executed successfully (if no errors above).")
