#!/usr/bin/env python3
"""
Hardware detection and autotune for optimal training settings.

Profiles the local machine and recommends optimal settings for:
- Number of environments
- Batch size
- Device selection

Profiles are stored in ~/.pokemon_red_rl/<user_id>.json and are reused
by run_actors_only.py and other scripts.

Usage:
    # Profile this machine
    python -m Daddy.autotune --user-id $(hostname)

    # Profile with custom benchmark duration
    python -m Daddy.autotune --user-id mac_actor --duration 30

    # Show current profile
    python -m Daddy.autotune --user-id mac_actor --show
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def detect_hardware() -> Dict:
    """Detect available hardware resources."""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 1
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)
    except ImportError:
        # Fallback without psutil
        import os
        cpu_count = os.cpu_count() or 1
        cpu_count_logical = cpu_count
        ram_gb = 8.0  # Assume 8GB if can't detect
    
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "cpu_count_logical": cpu_count_logical,
        "ram_gb": round(ram_gb, 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": 0,
        "gpu_count": 0,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        if info["gpu_count"] > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    
    return info


def recommend_settings(hardware: Dict) -> Dict:
    """
    Recommend optimal settings based on hardware.
    
    Returns dict with recommended settings for training.
    """
    recommendations = {}
    
    # Device selection
    if hardware.get("cuda_available"):
        recommendations["device"] = "cuda"
    elif hardware.get("mps_available"):
        recommendations["device"] = "mps"
    else:
        recommendations["device"] = "cpu"
    
    # Number of environments
    cpu_cores = hardware.get("cpu_count", 2)
    ram_gb = hardware.get("ram_gb", 8)
    
    # Each env uses ~0.5-1GB RAM
    # Leave 4GB for system on laptops, 2GB on servers
    is_laptop = hardware.get("platform") == "Darwin" or "laptop" in hardware.get("hostname", "").lower()
    reserved_ram = 4.0 if is_laptop else 2.0
    
    max_by_cpu = max(1, cpu_cores - 1)  # Leave 1 core for system
    max_by_ram = max(1, int((ram_gb - reserved_ram) / 1.0))
    
    # Cap based on platform
    if is_laptop:
        max_envs = 8
    elif recommendations["device"] == "cuda":
        max_envs = 32  # GPU can handle more
    else:
        max_envs = 16
    
    recommended_envs = min(max_by_cpu, max_by_ram, max_envs)
    recommendations["num_envs"] = max(1, recommended_envs)
    
    # Batch size
    if recommendations["device"] == "cuda":
        gpu_mem = hardware.get("gpu_memory_gb", 4)
        if gpu_mem >= 16:
            recommendations["batch_size"] = 128
        elif gpu_mem >= 8:
            recommendations["batch_size"] = 64
        else:
            recommendations["batch_size"] = 32
    elif recommendations["device"] == "mps":
        recommendations["batch_size"] = 64
    else:
        recommendations["batch_size"] = 32
    
    # Buffer size (smaller on limited RAM)
    if ram_gb >= 32:
        recommendations["buffer_size"] = 200000
    elif ram_gb >= 16:
        recommendations["buffer_size"] = 100000
    else:
        recommendations["buffer_size"] = 50000
    
    # Frame skip (higher on slower machines)
    if recommendations["device"] in ["cuda", "mps"]:
        recommendations["frame_skip"] = 4
    else:
        recommendations["frame_skip"] = 6
    
    return recommendations


def run_benchmark(
    duration_seconds: float = 20.0,
    num_envs: int = 2,
) -> Dict:
    """
    Run a short benchmark to measure actual performance.
    
    Returns dict with benchmark results.
    """
    results = {
        "duration_seconds": duration_seconds,
        "num_envs": num_envs,
        "steps": 0,
        "episodes": 0,
        "sps": 0.0,  # Steps per second
        "error": None,
    }
    
    try:
        from omegaconf import OmegaConf
        from .env_utils import SafeRedEnv, SimpleVectorEnv
        
        # Load config
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / "pokemonred_puffer" / "config.yaml"
        
        if not cfg_path.exists():
            results["error"] = f"Config not found: {cfg_path}"
            return results
        
        config = OmegaConf.load(cfg_path)
        env_cfg = OmegaConf.to_container(config.env, resolve=True)
        env_cfg["headless"] = True
        env_cfg["save_video"] = False
        env_cfg["two_bit"] = False
        
        # Resolve paths
        assets_root = repo_root / "archive" / "pokemonred_puffer_assets"
        rom_path = assets_root / "red.gb"
        state_dir = assets_root / "pyboy_states"
        
        if not rom_path.exists():
            rom_path = repo_root / "pokemonred_puffer" / "red.gb"
        if not state_dir.exists():
            state_dir = repo_root / "pokemonred_puffer" / "pyboy_states"
        
        env_cfg["gb_path"] = str(rom_path)
        env_cfg["state_dir"] = str(state_dir)
        
        if not rom_path.exists():
            results["error"] = f"ROM not found: {rom_path}"
            return results
        
        # Create environments
        def make_env(idx):
            cfg = OmegaConf.create(env_cfg)
            env = SafeRedEnv(cfg)
            return env
        
        print(f"[benchmark] Creating {num_envs} environments...")
        envs = SimpleVectorEnv([lambda i=idx: make_env(i) for idx in range(num_envs)])
        
        obs, _ = envs.reset(seed=7)
        
        print(f"[benchmark] Running for {duration_seconds}s...")
        start_time = time.time()
        steps = 0
        episodes = 0
        
        while time.time() - start_time < duration_seconds:
            # Random actions
            actions = np.random.randint(0, envs.single_action_space.n, size=num_envs)
            obs, rewards, terminated, truncated, infos = envs.step(actions)
            steps += num_envs
            episodes += int(np.sum(np.logical_or(terminated, truncated)))
        
        elapsed = time.time() - start_time
        
        envs.close()
        
        results["steps"] = steps
        results["episodes"] = episodes
        results["sps"] = round(steps / elapsed, 1)
        
        print(f"[benchmark] Complete: {results['sps']} SPS, {episodes} episodes")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"[benchmark] Error: {e}")
    
    return results


def get_profile_path(user_id: str) -> Path:
    """Get the path to a user's profile file."""
    profile_dir = Path.home() / ".pokemon_red_rl"
    return profile_dir / f"{user_id}.json"


def load_profile(user_id: str) -> Optional[Dict]:
    """Load an existing profile."""
    path = get_profile_path(user_id)
    if not path.exists():
        return None
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_profile(user_id: str, profile: Dict) -> Path:
    """Save a profile."""
    path = get_profile_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    
    return path


def autotune(
    user_id: str,
    benchmark_duration: float = 20.0,
    run_benchmark_flag: bool = True,
) -> Dict:
    """
    Run full autotune: detect hardware, optionally benchmark, save profile.
    
    Returns the complete profile dict.
    """
    print(f"[autotune] Detecting hardware for user: {user_id}")
    
    hardware = detect_hardware()
    print(f"[autotune] Platform: {hardware['platform']} ({hardware['platform_machine']})")
    print(f"[autotune] CPU: {hardware['cpu_count']} cores ({hardware['cpu_count_logical']} logical)")
    print(f"[autotune] RAM: {hardware['ram_gb']} GB")
    
    if hardware["cuda_available"]:
        print(f"[autotune] GPU: {hardware['gpu_name']} ({hardware['gpu_memory_gb']} GB)")
    elif hardware["mps_available"]:
        print("[autotune] MPS (Apple Silicon) available")
    else:
        print("[autotune] No GPU detected")
    
    recommendations = recommend_settings(hardware)
    print(f"[autotune] Recommended device: {recommendations['device']}")
    print(f"[autotune] Recommended num_envs: {recommendations['num_envs']}")
    print(f"[autotune] Recommended batch_size: {recommendations['batch_size']}")
    
    benchmark_results = None
    if run_benchmark_flag:
        benchmark_results = run_benchmark(
            duration_seconds=benchmark_duration,
            num_envs=min(2, recommendations["num_envs"]),
        )
        
        if benchmark_results.get("error"):
            print(f"[autotune] Benchmark failed: {benchmark_results['error']}")
        else:
            # Adjust recommendations based on benchmark
            sps = benchmark_results.get("sps", 0)
            if sps > 0:
                # Scale num_envs based on performance
                if sps < 50:
                    recommendations["num_envs"] = max(1, recommendations["num_envs"] // 2)
                elif sps > 200:
                    recommendations["num_envs"] = min(recommendations["num_envs"] * 2, 16)
    
    profile = {
        "user_id": user_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware,
        "recommendations": recommendations,
        "benchmark": benchmark_results,
        # Flat keys for easy access
        "recommended_num_envs": recommendations["num_envs"],
        "recommended_batch_size": recommendations["batch_size"],
        "recommended_device": recommendations["device"],
        "recommended_buffer_size": recommendations["buffer_size"],
        "recommended_frame_skip": recommendations["frame_skip"],
    }
    
    path = save_profile(user_id, profile)
    print(f"[autotune] Profile saved: {path}")
    
    profile["profile_path"] = str(path)
    return profile


def parse_args():
    parser = argparse.ArgumentParser(description="Hardware detection and autotune")
    parser.add_argument(
        "--user-id",
        default=None,
        help="User identifier (default: hostname)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Benchmark duration in seconds"
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the benchmark (just detect hardware)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show existing profile instead of running autotune"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine user ID
    if args.user_id is None:
        args.user_id = socket.gethostname().replace(".", "_")
    
    if args.show:
        profile = load_profile(args.user_id)
        if profile is None:
            print(f"No profile found for user: {args.user_id}")
            print(f"Run: python -m Daddy.autotune --user-id {args.user_id}")
        else:
            print(json.dumps(profile, indent=2))
        return
    
    profile = autotune(
        user_id=args.user_id,
        benchmark_duration=args.duration,
        run_benchmark_flag=not args.no_benchmark,
    )
    
    print("\n=== Recommendations ===")
    print(f"  Device:      {profile['recommended_device']}")
    print(f"  Num envs:    {profile['recommended_num_envs']}")
    print(f"  Batch size:  {profile['recommended_batch_size']}")
    print(f"  Buffer size: {profile['recommended_buffer_size']}")
    print(f"  Frame skip:  {profile['recommended_frame_skip']}")


if __name__ == "__main__":
    main()

