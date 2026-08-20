# Setup Guide

## Google Colab (Recommended)

1. Upload the project to your Google Drive or clone from GitHub.
2. Open a new Colab notebook (GPU runtime recommended: Runtime > Change runtime type > T4 GPU).
3. Run the following cells:

```python
# Mount Drive (if cloned to Drive)
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/robotic-arm-rl

# Install dependencies
!pip install -r requirements.txt

# Verify setup
!python scripts/verify_setup.py

# Train TD3 on Reach (smoke test, ~5 min on CPU)
!python scripts/train.py --algo td3 --task reach --seed 0 --steps 50000
```

## Local Setup

### Prerequisites
- Python 3.9+ (3.10 or 3.11 recommended)
- pip >= 21.0

### Steps

```bash
git clone <repo-url>
cd robotic-arm-rl
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python scripts/verify_setup.py
```

### PyBullet on Windows
If you encounter build errors for `pybullet`, try:
```bash
pip install pybullet --find-links https://github.com/Technoculture/pybullet-wheels/releases
```
Or use Colab (prebuilt wheels) to avoid build-from-source on Windows.

## TensorBoard

```bash
tensorboard --logdir logs/
# Open: http://localhost:6006
```
