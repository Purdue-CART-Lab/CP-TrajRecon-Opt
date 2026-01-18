#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 18:02:48 2026

@author: Tianheng Zhu
"""

from __future__ import annotations

import argparse
import sys
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except ImportError as e:
    raise ImportError(
        "PyYAML is required. Install with: pip install pyyaml"
    ) from e

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
import data_prep as dp  # repo_root/src/data_prep.py


# -----------------------------
# Helpers
# -----------------------------
def _require(cfg: Dict[str, Any], key: str):
    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")
    return cfg[key]


def _get(cfg: Dict[str, Any], key: str, default: Any):
    return cfg[key] if key in cfg else default


def _as_list(x) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _maybe_mkdir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _safe_rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=existing)


def _ensure_columns(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Ensure a canonical column exists by copying from the first available alias.

    Example:
      aliases = {"v_length": ["v_length", "v_Length", "v_len"]}
    """
    df = df.copy()
    for canon, names in aliases.items():
        if canon in df.columns:
            continue
        for nm in names:
            if nm in df.columns:
                df[canon] = df[nm]
                break
    return df

def _scale_columns(df: pd.DataFrame, cols: List[str], factor: float) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(float) * factor
    return df


def _apply_spatial_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    filters example:
      {"Local_Y": {"min": 50, "max": 550}, "Lane_ID": {"in": [1,2,3,4,5,6]}}
    """
    df = df.copy()
    for col, spec in (filters or {}).items():
        if col not in df.columns:
            continue
        if isinstance(spec, dict):
            if "min" in spec:
                df = df[df[col] >= spec["min"]]
            if "max" in spec:
                df = df[df[col] <= spec["max"]]
            if "in" in spec:
                df = df[df[col].isin(spec["in"])]
        else:
            # allow shorthand: {"Lane_ID": [1,2,3]}
            if isinstance(spec, list):
                df = df[df[col].isin(spec)]
    return df.reset_index(drop=True)


# -----------------------------
# Dataset-specific loaders
# -----------------------------
def load_csv(cfg: Dict[str, Any]) -> pd.DataFrame:
    input_cfg = _require(cfg, "input")
    path = Path(_require(input_cfg, "path")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    read_kwargs = _get(input_cfg, "read_csv_kwargs", {})
    df = pd.read_csv(path, **read_kwargs)
    return df


def preprocess_us101(cfg: Dict[str, Any], df: pd.DataFrame) -> None:

    ds = _require(cfg, "dataset")

    # time settings
    time_is_ms = bool(_get(ds, "time_is_ms", True))
    interval_sec = _get(ds, "interval_sec", 1)

    # 1) keep only desired sampling interval
    if time_is_ms:
        mask = (df["Global_Time"] % (1000 * interval_sec) == 0)
        df = df.loc[mask].copy()
        df.loc[:, "Global_Time"] = (df["Global_Time"].astype("int64") // 1000).astype("int64")
    else:
        df = df[df["Global_Time"] % interval_sec == 0]

    # 2) unit conversion ft -> m
    if _get(ds, "unit_ft_to_m", True):
        cols = ['Local_X', 'Local_Y','Global_X', 'Global_Y', 'v_Length', 'v_Width', 'v_Vel', 'v_Acc', 'Space_Hdwy']
        cols = [c for c in cols if c in df.columns]
        df.loc[:, cols] = df[cols].astype(float) * 0.3048

    # 3) optional spatial/lane filters
    df = _apply_spatial_filters(df, _get(ds, "filters", {}))

    df.reset_index(drop=True, inplace=True)

    # 4) trajectroy cleaning
    df = dp.traj_preprocessing_US101(df)
    
    # 5) CP data generation
    cp = _require(cfg, "cp")

    mode = str(_get(cp, "mode", "prob")).lower()  # "prob" | "occlusion"
    PR = float(_require(cp, "penetration_rate"))
    valid_range = float(_require(cp, "valid_range"))
    det_range = float(_get(cp, "range", 80))
    plr = float(_get(cp, "plr", 0.0))
    seed = int(_get(cp, "seed", 42))
    
    time_based = bool(_get(cp, "time_based", True))
    if mode == "prob":
        df_CP = dp.CP_data_generation(df, PR, valid_range, Range=det_range, PLR=plr, seed=seed, time_based = time_based)
    elif mode == "occlusion":
        df_CP = dp.CP_data_generation_occlusion_us101(df, PR, valid_range, Range=det_range, PLR=plr, seed=seed, time_based = time_based)
    df_dict = dp.build_complete_trajectories_dict_highway(df_CP)
    
    # 6) Hardcode data cleaning for us101
    fix = df_dict[56]
    fix = fix[fix.Vehicle_ID != 2613]
    fix.reset_index(drop=True, inplace=True)
    df_dict[56] = fix

    fix = df_dict[19]
    fix = fix[fix.Vehicle_ID != 986]
    fix.reset_index(drop=True, inplace=True)
    df_dict[19] = fix

    fix = df_dict[55]
    fix = fix[~((fix['Vehicle_ID'] == 2493) & (fix['Frame_ID'] == 7301))]
    fix.reset_index(drop=True, inplace=True)
    df_dict[55] = fix
    
    # 7) save .pkl file
    out_cfg = _require(cfg, "output")
    out_path = Path(_require(out_cfg, "path")).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(df_dict, f)

    return None

def preprocess_lankershim(cfg: Dict[str, Any], df: pd.DataFrame) -> None:

    ds = _require(cfg, "dataset")

    # time settings
    time_is_ms = bool(_get(ds, "time_is_ms", True))
    interval_sec = _get(ds, "interval_sec", 1)

    # 1) keep only desired sampling interval
    if time_is_ms:
        mask = (df["Global_Time"] % (1000 * interval_sec) == 0)
        df = df.loc[mask].copy()
        df.loc[:, "Global_Time"] = (df["Global_Time"].astype("int64") // 1000).astype("int64")
    else:
        df = df[df['Global_Time'] % interval_sec == 0]

    # 2) unit conversion ft -> m
    if _get(ds, "unit_ft_to_m", True):
        cols = ["Local_X","Local_Y","Global_X","Global_Y","v_length","v_Width","v_Vel","v_Acc","Space_Hdwy"]
        cols = [c for c in cols if c in df.columns]
        df.loc[:, cols] = df[cols].astype(float) * 0.3048
    
    df.reset_index(drop=True, inplace=True)
    
    # 3) Resolve duplicated vehicle id
    duplicate_id_fix = _require(ds, "duplicate_id_fix")
    
    for i in range(duplicate_id_fix["start_index"],len(df)):
        df.iloc[i,0] = df.iloc[i,0] + duplicate_id_fix["offset"]

    # 4) trajectroy cleaning
    section_id = _get(ds, "section_id", 2)
    direction = _get(ds, "direction", 2)
    df_2_2 = dp.traj_preprocessing_Lankershim(df, section_id, direction)
    
    smoothing = _require(cfg, "smoothing")
    max_acc = float(_get(smoothing, "max_acc", 3.5))
    max_jerk = float(_get(smoothing, "max_jerk", 3))
    window_length = int(_get(smoothing, "window_length", 7))
    polyorder = int(_get(smoothing, "polyorder", 2))
    dt = float(_get(smoothing, "dt", 1))
    
    df_2_2 = dp.smooth_trajectories(
        df_2_2,
        vehicle_id_col='Vehicle_ID',
        time_col='Global_Time',
        position_col='Local_Y',
        speed_col='v_Vel',
        acc_col='v_Acc',
        max_acc=max_acc,
        max_jerk=max_jerk,
        window_length=window_length,
        polyorder=polyorder,
        dt=dt
    )
    
    # 5) CP data generation
    cp = _require(cfg, "cp")

    mode = str(_get(cp, "mode", "prob")).lower()  # "prob" | "occlusion"
    PR = float(_require(cp, "penetration_rate"))
    valid_range = float(_require(cp, "valid_range"))
    det_range = float(_get(cp, "range", 80))
    plr = float(_get(cp, "plr", 0.0))
    seed = int(_get(cp, "seed", 42))
    
    time_based = bool(_get(cp, "time_based", True))
    if mode == "prob":
        df_2_2_CP = dp.CP_data_generation(df_2_2, PR, valid_range, Range=det_range, PLR=plr, seed=seed, time_based = time_based)
    elif mode == "occlusion":
        df_2_2_CP = dp.CP_data_generation_occlusion_lankershim(df_2_2, PR, valid_range, Range=det_range, PLR=plr, seed=seed, time_based = time_based)
    df_2_2_dict = dp.build_complete_trajectories_dict_intersection(df_2_2_CP)
    
    # 6) Hardcode data cleaning for lankershim
    fix = df_2_2_dict[1]
    fix.at[781, 'Lane_ID'] = 4
    df_2_2_dict[1] = fix

    fix = df_2_2_dict[4]
    fix.at[3635, 'Lane_ID'] = 4
    fix.at[3654, 'Lane_ID'] = 4
    fix.at[3779, 'Lane_ID'] = 4
    fix.at[3789, 'Lane_ID'] = 2
    fix.at[3586, 'Lane_ID'] = 3
    df_2_2_dict[4] = fix

    fix = df_2_2_dict[8]
    fix.at[8553, 'Lane_ID'] = 4
    df_2_2_dict[8] = fix

    fix = df_2_2_dict[9]
    fix.at[9649, 'Lane_ID'] = 1
    fix.at[8936, 'Lane_ID'] = 1
    fix.at[9452, 'Lane_ID'] = 3
    df_2_2_dict[9] = fix

    fix = df_2_2_dict[11]
    fix = fix[fix.Vehicle_ID != 1421]
    fix = fix[fix.Vehicle_ID != 1418]
    fix = fix[fix.Vehicle_ID != 2048]
    fix.reset_index(drop=True, inplace=True)
    df_2_2_dict[11] = fix

    fix = df_2_2_dict[13]
    fix = fix[~((fix['Vehicle_ID'] == 2376) & (fix['Frame_ID'] == 2261))]
    fix = fix[~((fix['Vehicle_ID'] == 2376) & (fix['Frame_ID'] == 2271))]
    fix = fix[~((fix['Vehicle_ID'] == 2376) & (fix['Frame_ID'] == 2281))]
    fix.reset_index(drop=True, inplace=True)
    df_2_2_dict[13] = fix
    
    # 7) save .pkl file
    out_cfg = _require(cfg, "output")
    out_path = Path(_require(out_cfg, "path")).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(df_2_2_dict, f)
    
    return None

'''
def preprocess_magic(cfg: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    ds = _require(cfg, "dataset")

    time_col = _get(ds, "time_col", "Global_Time")
    interval_sec = _get(ds, "interval_sec", 1)

    # MAGIC script used: df[df['Global_Time'] % 1 == 0]
    df = _filter_by_mod(df, time_col, float(interval_sec))
    df = df.reset_index(drop=True)

    # Optional rename / ensure columns
    df = _safe_rename_columns(df, _get(ds, "rename_columns", {}))
    df = _ensure_columns(
        df,
        aliases={
            "v_length": ["v_length", "v_Length"],
            "v_Width": ["v_Width", "v_width"],
        },
    )

    # Optional recompute v_Vel/v_Acc from Local_Y
    if bool(_get(ds, "recalc_speed_acc", False)) and hasattr(dp, "recalc_speed_acc"):
        df = dp.recalc_speed_acc(
            df,
            id_col=_get(ds, "id_col", "Vehicle_ID"),
            time_col=time_col,
            pos_col=_get(ds, "pos_col", "Local_Y"),
        )

    df = dp.traj_preprocessing_MAGIC(df)
    df_CP = dp.CP_data_generation(traj_101, 0.03, 400, Range=80, PLR=0.8, seed=100)
    
    return df.reset_index(drop=True)
'''

def save_pkl(cfg: Dict[str, Any], obj: Any) -> Path:
    out_cfg = _require(cfg, "output")
    out_path = Path(_require(out_cfg, "path")).expanduser()
    _maybe_mkdir(out_path)

    with open(out_path, "wb") as f:
        pickle.dump(obj, f)

    return out_path


# -----------------------------
# Main
# -----------------------------
def run_from_config(cfg: Dict[str, Any]) -> Path:
    ds = _require(cfg, "dataset")
    ds_name = str(_require(ds, "name")).lower()  # "us101" | "lankershim" | "magic"

    df = load_csv(cfg)

    if ds_name == "us101":
        df = preprocess_us101(cfg, df)
    elif ds_name == "lankershim":
        df = preprocess_lankershim(cfg, df)
    #elif ds_name == "magic":
    #    df = preprocess_magic(cfg, df)
    else:
        raise ValueError(f"dataset.name must be 'us101' or 'lankershim', got: {ds_name}") # 'magic'

    out_cfg = _require(cfg, "output")
    out_path = Path(_require(out_cfg, "path")).expanduser()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset and generate CP trajectory dict from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/ngsim_us101_simple.yaml)")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out = run_from_config(cfg)
    print(f"[OK] Saved trajectory dict to: {out}")


if __name__ == "__main__":
    main()
