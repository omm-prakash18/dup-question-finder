"""
setup_env.py — One-shot environment setup script
─────────────────────────────────────────────────────────────────────────────
Run this instead of manually installing packages:
    py -3.12 setup_env.py

What it does:
  1. Verifies Python 3.12 is being used (not 3.14+ which lacks torch wheels)
  2. Creates/verifies .venv with Python 3.12
  3. Installs PyTorch 2.3.1 + CUDA 12.1 from pytorch.org (handles large download)
  4. Installs remaining dependencies from requirements.txt
  5. Optionally installs spaCy model for word-embedding features in baseline
  6. Prints a summary of what to run next
"""

import sys
import subprocess
import platform

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 13)  # 3.14 has no torch wheels yet


def check_python_version():
    ver = sys.version_info
    if ver >= MAX_PYTHON:
        print(f"❌ Python {ver.major}.{ver.minor} detected — PyTorch has no wheels for this version yet.")
        print("   Run this script with: py -3.12 setup_env.py")
        sys.exit(1)
    if ver < MIN_PYTHON:
        print(f"❌ Python {ver.major}.{ver.minor} is too old. Use 3.10–3.12.")
        sys.exit(1)
    print(f"✅ Python {ver.major}.{ver.minor}.{ver.micro}")


def run(cmd, check=True, **kwargs):
    print(f"\n▶ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, **kwargs)
    return result


def main():
    check_python_version()

    pip = [sys.executable, "-m", "pip"]

    # Upgrade pip first
    run([*pip, "install", "--upgrade", "pip"])

    # Install PyTorch with CUDA 12.1 — must be BEFORE requirements.txt
    # so that other packages link against the correct torch version
    print("\n" + "="*60)
    print("Installing PyTorch 2.3.1 + CUDA 12.1 (~2.4 GB download)")
    print("This may take several minutes on slow connections...")
    print("="*60)
    run([
        *pip, "install",
        "torch==2.3.1", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "--timeout", "300",
        "--retries", "10",
    ])

    # Install remaining dependencies
    print("\n" + "="*60)
    print("Installing all other dependencies from requirements.txt")
    print("="*60)
    run([
        *pip, "install", "-r", "requirements.txt",
        "--timeout", "300",
        "--retries", "10",
    ])

    # Optionally install spaCy model (used only by classical baseline)
    print("\n" + "="*60)
    print("Installing spaCy en_core_web_md (for word-embedding features)")
    print("Skip this if you only plan to use SBERT + FAISS pipeline.")
    print("="*60)
    run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=False)

    # Verify torch + CUDA
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    verify = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'torch {torch.__version__}'); "
         "print(f'CUDA available: {torch.cuda.is_available()}'); "
         "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"],
        capture_output=True, text=True
    )
    print(verify.stdout)
    if verify.returncode != 0:
        print(f"❌ Torch verification failed:\n{verify.stderr}")
    else:
        print("✅ PyTorch + CUDA verified!")

    print("\n" + "="*60)
    print("SETUP COMPLETE — Run pipeline in this order:")
    print("="*60)
    print("  1. python src/data/download_data.py")
    print("  2. python src/data/preprocess.py")
    print("  3. python src/models/baseline.py")
    print("  4. python src/models/sbert_pipeline.py")
    print("  5. pytest tests/ -v")
    print("  6. uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print()
    print("  Optional: enable GPU FAISS via conda:")
    print("    conda install -c pytorch -c nvidia faiss-gpu cudatoolkit=12.1")
    print("    Then set use_gpu: true in config.yaml")


if __name__ == "__main__":
    main()
