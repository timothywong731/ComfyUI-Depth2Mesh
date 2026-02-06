import os
import subprocess
import sys


def install():
    """
    Install requirements for the DepthMap2STL ComfyUI node.
    This script is automatically executed by ComfyUI Manager during installation.
    """
    # Locate requirements.txt in the same directory as this script
    req_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "requirements.txt"
    )

    if not os.path.isfile(req_file):
        print(f"[DepthMap2STL] Could not find requirements.txt at {req_file}")
        return

    print(f"[DepthMap2STL] Installing requirements from {req_file}...")

    command = [sys.executable, "-m", "pip", "install", "-r", req_file]

    try:
        subprocess.check_call(command)
        print("[DepthMap2STL] Installation successful.")
    except subprocess.CalledProcessError as e:
        print(f"[DepthMap2STL] Installation failed! Error code: {e.returncode}")
        print(f"[DepthMap2STL] Command attempted: {' '.join(command)}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[DepthMap2STL] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install()
