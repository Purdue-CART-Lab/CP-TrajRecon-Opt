#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:35:34 2026

@author: Tianheng Zhu
"""

import argparse
import pickle
from pathlib import Path
import sys
import yaml

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config for optimization.")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)

    # --- Make sure repo root is importable ---
    # If this script is in scripts/, repo root is parent of scripts/
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Import optimization entry
    from src.MILP_us101 import optimization
    
    io = cfg["io"]
    pkl_path = Path(io["pkl_path"]).expanduser()
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")

    df_dict = load_pkl(pkl_path)
    print(f"[INFO] Loaded pkl: {pkl_path}")

    for i in range(1, len(df_dict) + 1):
        try:
            optimization(df_dict, case_index=i, cfg=cfg)
        except Exception as e:
            print(f"[ERROR] case_index={i}: {e}")
            # continue to next
            continue

    print("[DONE] All requested cases processed.")

if __name__ == "__main__":
    main()
