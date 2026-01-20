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

# --- Make sure repo root is importable ---
# If this script is in scripts/, repo root is parent of scripts/
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import src.performance_metrics_us101 as pm_us101
import src.performance_metrics_lankershim as pm_lksm

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
    
    io = cfg["io"]
    pkl_path = Path(io["pkl_path"]).expanduser()
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    
    df_dict = load_pkl(pkl_path)
    print(f"[INFO] Loaded pkl: {pkl_path}")
    
    dataset_name = io["dataset_name"]
    output_root = io["output_root"]
    exp_name = io["experiment_name"]
    exp_id = io["experiment_id"]
    
    if dataset_name == "us101":
        print("[INFO] Position error calculation")
        mae_x, mape_x, rmse_x, mae_k = pm_us101.position_err_calculation_all(df_dict, output_root, exp_name, exp_id)
        mae_x_undtct, mape_x_undtct, rmse_x_undtct, mae_k_undtct = pm_us101.position_err_calculation_undetected(df_dict, output_root, exp_name, exp_id)
        print("[INFO] Lane change error calculation")
        mae_lc, mae_lc_undtct = pm_us101.lc_timing_err_calculation(df_dict, output_root, exp_name, exp_id)
    elif dataset_name == "lankershim":
        print("[INFO] Position error calculation")
        mae_x, mape_x, rmse_x, mae_k = pm_lksm.position_err_calculation_all(df_dict, output_root, exp_name, exp_id)
        mae_x_undtct, mape_x_undtct, rmse_x_undtct, mae_k_undtct = pm_lksm.position_err_calculation_undetected(df_dict, output_root, exp_name, exp_id)
        print("[INFO] Lane change error calculation")
        mae_lc, mae_lc_undtct = pm_lksm.lc_timing_err_calculation(df_dict, output_root, exp_name, exp_id)

    print("\n" + "=" * 70)
    print(f"Experiment: {exp_name} | Run ID: {exp_id}")
    print("=" * 70)
    
    # ==================================================
    # All trajectory points
    # ==================================================
    print("\n[All Trajectory Points]")
    print("-" * 70)
    print(f"MAE_x   : {mae_x:.4f}")
    print(f"MAPE_x  : {mape_x:.2f}%")
    print(f"RMSE_x  : {rmse_x:.4f}")
    print(f"MAE_k   : {mae_k:.4f}")
    print(f"MAE_LC  : {mae_lc:.4f}")
    
    # ==================================================
    # Undetected trajectory points
    # ==================================================
    print("\n[Undetected Trajectory Points]")
    print("-" * 70)
    print(f"MAE_x   : {mae_x_undtct:.4f}")
    print(f"MAPE_x  : {mape_x_undtct:.2f}%")
    print(f"RMSE_x  : {rmse_x_undtct:.4f}")
    print(f"MAE_k   : {mae_k_undtct:.4f}")
    print(f"MAE_LC  : {mae_lc_undtct:.4f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
