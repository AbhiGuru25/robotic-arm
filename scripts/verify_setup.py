"""
scripts/verify_setup.py
=======================
One-command environment verification script.

Run this after installation to confirm all dependencies are correctly
installed and both panda-gym environments can be instantiated.

Usage
-----
    python scripts/verify_setup.py

Expected output on success
--------------------------
    [✓] Python version: 3.x.x
    [✓] PyTorch  x.x.x  |  CUDA available: True/False
    [✓] Gymnasium x.x.x
    [✓] panda-gym x.x.x  (PandaReach-v3, PandaPickAndPlace-v3)
    [✓] Stable-Baselines3 x.x.x
    [✓] TensorBoard x.x.x
    [✓] PyYAML x.x.x
    [✓] Environment smoke test PASSED  (PandaReach-v3, 5 steps)
    [✓] Environment smoke test PASSED  (PandaPickAndPlace-v3, 5 steps)
    ============================================================
    All checks passed! You are ready to train.
    ============================================================
"""

import sys
import importlib

# ----------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}[OK]{RESET}  {msg}")
def fail(msg): print(f"{RED}[FAIL]{RESET} {msg}"); sys.exit(1)
def warn(msg): print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        fail(f"Python >=3.9 required. Got {sys.version}")
    ok(f"Python {sys.version.split()[0]}")


def check_import(package: str, display: str = None, version_attr: str = "__version__"):
    display = display or package
    try:
        mod = importlib.import_module(package)
        ver = getattr(mod, version_attr, "unknown")
        ok(f"{display}  {ver}")
        return mod
    except ImportError as e:
        fail(f"{display} not installed: {e}")


def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        ok(f"PyTorch {torch.__version__}  |  CUDA available: {cuda}")
        if cuda:
            ok(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        fail("PyTorch not installed. Run: pip install torch")


def check_gym_envs():
    import gymnasium as gym
    import panda_gym  # noqa: F401 — registers environments

    tasks = ["PandaReach-v3", "PandaPickAndPlace-v3"]
    for env_id in tasks:
        try:
            env = gym.make(env_id)
            obs, _ = env.reset(seed=0)
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            ok(f"Smoke test PASSED  ({env_id}, 5 steps)  "
               f"obs_keys={list(obs.keys()) if isinstance(obs, dict) else 'flat'}")
        except Exception as e:
            fail(f"Environment {env_id} failed: {e}")


def check_yaml_configs():
    """Check that YAML config files exist and are parseable."""
    import yaml
    import pathlib

    config_dir = pathlib.Path(__file__).parent.parent / "configs"
    if not config_dir.exists():
        warn("configs/ directory not found — configs not yet generated.")
        return

    yaml_files = list(config_dir.glob("*.yaml"))
    if not yaml_files:
        warn("No YAML config files found in configs/.")
        return

    for f in yaml_files:
        try:
            with open(f, encoding="utf-8") as fh:
                yaml.safe_load(fh)
            ok(f"Config parsed: configs/{f.name}")
        except Exception as e:
            warn(f"Config parse warning {f.name}: {e}")


def main():
    print("=" * 60)
    print("  Robotic Arm RL — Environment Verification")
    print("=" * 60)
    print()

    check_python()
    check_torch()
    check_import("gymnasium", "Gymnasium")
    check_import("panda_gym", "panda-gym")
    check_import("stable_baselines3", "Stable-Baselines3")
    check_import("tensorboard", "TensorBoard")
    check_import("yaml", "PyYAML")
    check_import("matplotlib", "Matplotlib")
    check_import("numpy", "NumPy")

    print()
    check_gym_envs()

    print()
    check_yaml_configs()

    print()
    print("=" * 60)
    print("  All checks passed! You are ready to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()
