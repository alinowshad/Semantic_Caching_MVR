import json
import os
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmarks._plotter_helper import (
    compute_avg_latency_score,
    compute_cache_hit_rate_cumulative_list,
    compute_cache_hit_rate_score,
    compute_error_rate_cumulative_list,
    compute_error_rate_score,
    compute_false_positive_rate_score,
    compute_precision_score,
    compute_recall_score,
    convert_to_dataframe_from_json_file,
)


def __normalize_float_key_str(float_val: float) -> str:
    key_str = f"{float_val:.3f}".rstrip("0").rstrip(".")
    if not key_str and float_val == 0:
        key_str = "0"
    return key_str


def __collect_splitter_run_dirs_by_delta_and_config(results_dir_path: str):
    mapping: Dict[float, Dict[str, List[str]]] = {}
    if not os.path.exists(results_dir_path):
        return mapping

    pat = re.compile(
        r"^vcache_splitter_(?P<delta>[0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+)_(?P<sel>top_k|all)_(?P<k>[^_]+)_run_(?P<run>\d+)$"
    )

    for dir_name in os.listdir(results_dir_path):
        full_dir_path = os.path.join(results_dir_path, dir_name)
        if not os.path.isdir(full_dir_path):
            continue
        m = pat.match(dir_name)
        if not m:
            continue

        try:
            delta_val = float(m.group("delta"))
        except Exception:
            continue

        sel = m.group("sel")
        k = m.group("k")
        config_key = f"{sel}|{k}"
        mapping.setdefault(delta_val, {}).setdefault(config_key, []).append(full_dir_path)

    return mapping


def __try_parse_numeric(val: str):
    try:
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def __plot_splitter_k_sweep_for_deltas(
    *,
    results_dir: str,
    splitter_dirs_by_delta_cfg: Dict[float, Dict[str, List[str]]],
    timestamp: str,
    font_size: int,
):
    # Only plot when a delta has multiple configs (typically multiple k values).
    if not splitter_dirs_by_delta_cfg:
        return

    for delta_val, cfg_map in splitter_dirs_by_delta_cfg.items():
        if not cfg_map or len(cfg_map) <= 1:
            continue

        # Build per-selection series (top_k vs all) over k.
        per_sel: Dict[str, List[Dict[str, float | int | str]]] = {}
        for cfg_key, run_dirs in cfg_map.items():
            try:
                sel, k_raw = str(cfg_key).split("|", 1)
            except Exception:
                sel, k_raw = "unknown", str(cfg_key)

            stats = __aggregate_stats_for_cache_hit_error_rate(run_dirs)
            k_num = __try_parse_numeric(str(k_raw))
            per_sel.setdefault(sel, []).append(
                {
                    "k_raw": str(k_raw),
                    "k_num": k_num,
                    "cache_hit_rate_mean": float(stats.get("cache_hit_rate_mean", 0.0)),
                    "cache_hit_rate_ci_low": float(stats.get("cache_hit_rate_ci_low", 0.0)),
                    "cache_hit_rate_ci_up": float(stats.get("cache_hit_rate_ci_up", 0.0)),
                    "error_rate_mean": float(stats.get("error_rate_mean", 0.0)),
                    "error_rate_ci_low": float(stats.get("error_rate_ci_low", 0.0)),
                    "error_rate_ci_up": float(stats.get("error_rate_ci_up", 0.0)),
                }
            )

        # If we still effectively only have one k (e.g., multiple runs same k), skip.
        total_unique_k = set()
        for sel, rows in per_sel.items():
            for r in rows:
                total_unique_k.add(r.get("k_raw"))
        if len(total_unique_k) <= 1:
            continue

        def _sort_key(row):
            k_num = row.get("k_num")
            if isinstance(k_num, (int, float)):
                return (0, float(k_num))
            return (1, str(row.get("k_raw")))

        for sel in list(per_sel.keys()):
            per_sel[sel] = sorted(per_sel[sel], key=_sort_key)

        # Collect numeric k values only (k-sweep). Non-numeric entries like "all" are skipped.
        k_numeric_all: List[float] = []
        for _sel, rows in per_sel.items():
            for r in rows:
                k_num = r.get("k_num")
                if isinstance(k_num, (int, float)) and not np.isnan(float(k_num)):
                    k_numeric_all.append(float(k_num))

        if not k_numeric_all:
            continue

        # For the sensitivity analysis we only want K=1..10 on the x-axis.
        k_sweep_min = 1
        k_sweep_max = 10
        xticks = list(range(k_sweep_min, k_sweep_max + 1))

        delta_str = __normalize_float_key_str(float(delta_val))

        # -------- Plot 1: Cache Hit Rate vs k --------
        plt.figure(figsize=(12, 11))
        plt.rcParams.update({"font.size": font_size})

        ymins: List[float] = []
        ymaxs: List[float] = []
        for sel, rows in per_sel.items():
            rows_num = [
                r
                for r in rows
                if isinstance(r.get("k_num"), (int, float))
                and not np.isnan(float(r["k_num"]))
                and k_sweep_min <= float(r["k_num"]) <= k_sweep_max
            ]
            if not rows_num:
                continue
            xs = [float(r["k_num"]) for r in rows_num]
            ys = [float(r["cache_hit_rate_mean"]) for r in rows_num]
            ylo = [float(r["cache_hit_rate_ci_low"]) for r in rows_num]
            yhi = [float(r["cache_hit_rate_ci_up"]) for r in rows_num]
            plt.plot(xs, ys, marker="o", label=str(sel))
            try:
                plt.fill_between(xs, ylo, yhi, alpha=0.15)
            except Exception:
                pass
            ymins.extend(ylo)
            ymaxs.extend(yhi)

        if ymins and ymaxs:
            ymin = float(min(ymins))
            ymax = float(max(ymaxs))
            span = max(1e-9, ymax - ymin)
            pad = max(0.005, 0.1 * span)
            plt.ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

        plt.xlim(k_sweep_min - 0.5, k_sweep_max + 0.5)
        x_tick_fs = min(12, max(6, int(font_size * 0.4)))
        plt.xticks(xticks, [str(k) for k in xticks], fontsize=x_tick_fs)
        plt.tick_params(axis="x", pad=2)
        plt.xlabel("candidate_k")
        plt.ylabel("Cache Hit Rate")
        plt.title(f"vCache Splitter: Cache Hit Rate vs k (delta={delta_str})")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        out1 = os.path.join(
            results_dir,
            f"vcache_splitter_k_sweep_cache_hit_delta_{delta_str}_{timestamp}.pdf",
        )
        plt.savefig(out1, format="pdf", bbox_inches="tight", transparent=True)
        plt.close()

        # -------- Plot 2: Error Rate vs k --------
        plt.figure(figsize=(12, 11))
        plt.rcParams.update({"font.size": font_size})

        ymins = []
        ymaxs = []
        for sel, rows in per_sel.items():
            rows_num = [
                r
                for r in rows
                if isinstance(r.get("k_num"), (int, float))
                and not np.isnan(float(r["k_num"]))
                and k_sweep_min <= float(r["k_num"]) <= k_sweep_max
            ]
            if not rows_num:
                continue
            xs = [float(r["k_num"]) for r in rows_num]
            ys = [float(r["error_rate_mean"]) for r in rows_num]
            ylo = [float(r["error_rate_ci_low"]) for r in rows_num]
            yhi = [float(r["error_rate_ci_up"]) for r in rows_num]
            plt.plot(xs, ys, marker="o", label=str(sel))
            try:
                plt.fill_between(xs, ylo, yhi, alpha=0.15)
            except Exception:
                pass
            ymins.extend(ylo)
            ymaxs.extend(yhi)

        if ymins and ymaxs:
            ymin = float(min(ymins))
            ymax = float(max(ymaxs))
            span = max(1e-9, ymax - ymin)
            pad = max(0.005, 0.1 * span)
            plt.ylim(max(0.0, ymin - pad), min(1.0, ymax + pad))

        plt.xlim(k_sweep_min - 0.5, k_sweep_max + 0.5)
        x_tick_fs = min(12, max(6, int(font_size * 0.4)))
        plt.xticks(xticks, [str(k) for k in xticks], fontsize=x_tick_fs)
        plt.tick_params(axis="x", pad=2)
        plt.xlabel("candidate_k")
        plt.ylabel("Error Rate")
        plt.title(f"vCache Splitter: Error Rate vs k (delta={delta_str})")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        out2 = os.path.join(
            results_dir,
            f"vcache_splitter_k_sweep_error_rate_delta_{delta_str}_{timestamp}.pdf",
        )
        plt.savefig(out2, format="pdf", bbox_inches="tight", transparent=True)
        plt.close()


def __calculate_mean_and_ci_from_runs(
    per_run_values: List[float],
    z: float = 1.96,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
):
    """Calculates mean and confidence interval from a list of per-run metric values."""
    if not per_run_values:
        return 0.0, 0.0, 0.0
    if len(per_run_values) == 1:
        mean_val = per_run_values[0]
        # Apply clamping even for single value if specified
        if clamp_min is not None:
            mean_val = max(clamp_min, mean_val)
        if clamp_max is not None:
            mean_val = min(clamp_max, mean_val)
        return mean_val, mean_val, mean_val

    mean_val = np.mean(per_run_values)
    std_dev = np.std(per_run_values, ddof=1)
    sem = std_dev / np.sqrt(len(per_run_values))

    ci_low = mean_val - z * sem
    ci_up = mean_val + z * sem

    if clamp_min is not None:
        ci_low = max(clamp_min, ci_low)
        mean_val = max(
            clamp_min, mean_val
        )  # Ensure mean is also clamped if interval is
    if clamp_max is not None:
        ci_up = min(clamp_max, ci_up)
        mean_val = min(
            clamp_max, mean_val
        )  # Ensure mean is also clamped if interval is

    # Ensure CI bounds don't cross after clamping if mean was outside clamp range initially
    if clamp_min is not None and clamp_max is not None:
        mean_val = max(clamp_min, min(clamp_max, mean_val))
        ci_low = max(clamp_min, min(clamp_max, ci_low))
        ci_up = max(clamp_min, min(clamp_max, ci_up))
        if (
            ci_low > ci_up
        ):  # if clamping pushed lower bound above upper bound (e.g. very wide CI outside narrow clamp)
            ci_low = ci_up = mean_val

    return mean_val, ci_low, ci_up


def __collect_run_dirs_by_prefix_and_key(
    results_dir_path: str, dir_prefix_to_match: str
):
    """
    Collects full paths to run directories based on a prefix and extracts a normalized key.
    The key is typically a threshold or delta value formatted as a string.
    Example: results_dir_path contains "gptcache_0.8_run_1", "gptcache_0.8_run_2".
    If dir_prefix_to_match is "gptcache_", it will map {"0.8": [path_to_run1, path_to_run2]}.
    """
    mapping: Dict[str, List[str]] = {}
    if not os.path.exists(results_dir_path):
        return mapping

    for dir_name in os.listdir(results_dir_path):
        full_dir_path = os.path.join(results_dir_path, dir_name)
        if not dir_name.startswith(dir_prefix_to_match) or not os.path.isdir(
            full_dir_path
        ):
            continue

        name_part_after_prefix = dir_name[
            len(dir_prefix_to_match) :
        ]  # e.g., "0.8", "0.01_run_1"

        # Attempt to match a floating point number at the beginning of the remaining part.
        # This should capture values like "0.8", "0.01", "0.955" etc.
        # It handles cases like "0.8_run_1" by only taking "0.8".
        match = re.match(r"([0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+)", name_part_after_prefix)

        if match:
            key_str_from_dir = match.group(1)
            try:
                float_val = float(key_str_from_dir)
                # Normalize the string representation to match keys from data_frames
                # e.g., 0.8 -> "0.8", 0.0 -> "0", 0.010 -> "0.01"
                normalized_key_str = f"{float_val:.3f}".rstrip("0").rstrip(".")
                if not normalized_key_str and float_val == 0:  # Handles 0.0 becoming ""
                    normalized_key_str = "0"
                elif (
                    not normalized_key_str and "." in key_str_from_dir
                ):  # e.g. if input was "0."
                    normalized_key_str = key_str_from_dir.rstrip("0").rstrip(".")
                    if not normalized_key_str:
                        normalized_key_str = "0"

                mapping.setdefault(normalized_key_str, []).append(full_dir_path)
            except ValueError:
                # print(f"Warning: Could not parse float from '{key_str_from_dir}' in dir '{dir_name}'")
                continue
        # else:
        # print(f"Warning: No numeric key found in '{name_part_after_prefix}' for dir '{dir_name}'")

    return mapping


def __draw_confidence_series(
    x_means: List[float],
    y_means: List[float],
    x_lower_errors: List[float],
    x_upper_errors: List[float],
    y_lower_errors: List[float],
    y_upper_errors: List[float],
    is_multi_run: List[bool],
    color: str,
    label: str,
    marker_size: int,
    line_style: str = "-",
    line_width: int = 3,
    error_bar_alpha: float = 0.6,
    error_bar_capsize: int = 4,
    error_bar_elinewidth: float = 1.5,
):
    """Plots a series with optional confidence interval error bars."""
    if not x_means:
        return

    plt.plot(
        x_means,
        y_means,
        line_style,
        color=color,
        linewidth=line_width,
        label=label,
        zorder=3,
    )

    for i in range(len(x_means)):
        xm, ym = x_means[i], y_means[i]
        is_multi = is_multi_run[i]

        current_marker_size = 5 if is_multi else marker_size
        plt.plot(xm, ym, "o", color=color, markersize=current_marker_size, zorder=10)

        if is_multi:
            # Prepare error lists for errorbar, ensuring they are 2xN arrays
            # xerr should be [[lower_errors], [upper_errors]]
            current_x_err = None
            if (
                x_lower_errors[i] > 1e-9 or x_upper_errors[i] > 1e-9
            ):  # Check for non-zero error
                current_x_err = [[x_lower_errors[i]], [x_upper_errors[i]]]

            current_y_err = None
            if y_lower_errors[i] > 1e-9 or y_upper_errors[i] > 1e-9:
                current_y_err = [[y_lower_errors[i]], [y_upper_errors[i]]]

            if current_x_err or current_y_err:  # Only plot if there is some error
                plt.errorbar(
                    xm,
                    ym,
                    xerr=current_x_err,
                    yerr=current_y_err,
                    fmt="none",
                    ecolor=color,
                    elinewidth=error_bar_elinewidth,
                    capsize=error_bar_capsize,
                    alpha=error_bar_alpha,
                    zorder=5,
                )


# Aggregation functions for different metrics
def __aggregate_stats_for_roc(run_dirs: List[str], keep_split: int, z: float = 1.96):
    per_run_tpr_values = []
    per_run_fpr_values = []

    if not run_dirs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for rd_path in run_dirs:
        run_total_tp = 0
        run_total_fn = 0
        run_total_fp = 0
        run_total_tn = 0
        has_data_for_run = False

        for file_name in os.listdir(rd_path):
            if file_name.startswith("results_") and file_name.endswith(".json"):
                file_path = os.path.join(rd_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df, _, _ = convert_to_dataframe_from_json_file(
                        json_data=data, keep_split=keep_split
                    )
                    run_total_tp += np.sum(df["tp_list"])
                    run_total_fn += np.sum(df["fn_list"])
                    run_total_fp += np.sum(df["fp_list"])
                    run_total_tn += np.sum(df["tn_list"])
                    has_data_for_run = True
                except Exception:
                    # print(f"Warning: Error reading/processing {file_path} in __aggregate_stats_for_roc: {e}")
                    continue

        if has_data_for_run:
            current_run_tpr = (
                run_total_tp / (run_total_tp + run_total_fn)
                if (run_total_tp + run_total_fn) > 0
                else 0.0
            )
            current_run_fpr = (
                run_total_fp / (run_total_fp + run_total_tn)
                if (run_total_fp + run_total_tn) > 0
                else 0.0
            )
            per_run_tpr_values.append(current_run_tpr)
            per_run_fpr_values.append(current_run_fpr)

    # Calculate mean and CI for TPR
    tpr_mean, tpr_ci_low, tpr_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_tpr_values, z, clamp_min=0.0, clamp_max=1.0
    )

    # Calculate mean and CI for FPR
    fpr_mean, fpr_ci_low, fpr_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_fpr_values, z, clamp_min=0.0, clamp_max=1.0
    )

    return tpr_mean, tpr_ci_low, tpr_ci_up, fpr_mean, fpr_ci_low, fpr_ci_up


def __aggregate_stats_for_latency_error(run_dirs: List[str], z: float = 1.96):
    per_run_error_rate_values = []
    per_run_avg_latency_values = []

    if not run_dirs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for rd_path in run_dirs:
        run_total_fp = 0
        run_num_samples_for_fp = 0
        run_total_latency = 0.0
        run_num_latency_samples = 0
        has_data_for_run = False

        for file_name in os.listdir(rd_path):
            if file_name.startswith("results_") and file_name.endswith(".json"):
                file_path = os.path.join(rd_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df, _, _ = convert_to_dataframe_from_json_file(data)

                    run_total_fp += np.sum(df["fp_list"])
                    run_num_samples_for_fp += len(df["fp_list"])

                    run_total_latency += np.sum(df["latency_vectorq_list"])
                    run_num_latency_samples += len(df["latency_vectorq_list"])
                    has_data_for_run = True
                except Exception:
                    # print(f"Warning: Error reading/processing {file_path} in __aggregate_stats_for_latency_error: {e}")
                    continue

        if has_data_for_run:
            if run_num_samples_for_fp > 0:
                current_run_error_rate = run_total_fp / run_num_samples_for_fp
                per_run_error_rate_values.append(current_run_error_rate)

            if run_num_latency_samples > 0:
                current_run_avg_latency = run_total_latency / run_num_latency_samples
                per_run_avg_latency_values.append(current_run_avg_latency)

    # Calculate mean and CI for Error Rate
    err_mean, err_ci_low, err_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_error_rate_values, z, clamp_min=0.0, clamp_max=1.0
    )

    # Calculate mean and CI for Latency
    lat_mean, lat_ci_low, lat_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_avg_latency_values, z, clamp_min=0.0
    )

    return err_mean, err_ci_low, err_ci_up, lat_mean, lat_ci_low, lat_ci_up


def __aggregate_stats_for_cache_hit_error_rate(run_dirs: List[str], z: float = 1.96):
    per_run_cache_hit_rate_values = []
    per_run_error_rate_values = []

    if not run_dirs:
        return {
            "cache_hit_rate_mean": 0.0,
            "cache_hit_rate_ci_low": 0.0,
            "cache_hit_rate_ci_up": 0.0,
            "error_rate_mean": 0.0,
            "error_rate_ci_low": 0.0,
            "error_rate_ci_up": 0.0,
        }

    for rd_path in run_dirs:
        run_total_samples_ch = 0
        run_total_cache_hits = 0
        run_total_samples_er = 0
        run_total_fp = 0
        has_data_for_run = False

        for file_name in os.listdir(rd_path):
            if file_name.startswith("results_") and file_name.endswith(".json"):
                file_path = os.path.join(rd_path, file_name)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    df, _, _ = convert_to_dataframe_from_json_file(data)

                    run_total_samples_ch += len(
                        df["cache_hit_list"]
                    )  # Or len(df) if more appropriate
                    run_total_cache_hits += int(np.sum(df["cache_hit_list"]))

                    run_total_samples_er += len(df["fp_list"])  # Or len(df)
                    run_total_fp += int(np.sum(df["fp_list"]))
                    has_data_for_run = True
                except Exception:
                    # print(f"Warning: Error reading/processing {file_path} in __aggregate_stats_for_cache_hit_error_rate: {e}")
                    continue

        if has_data_for_run:
            if run_total_samples_ch > 0:
                current_run_ch_rate = run_total_cache_hits / run_total_samples_ch
                per_run_cache_hit_rate_values.append(current_run_ch_rate)

            if run_total_samples_er > 0:
                current_run_er_rate = run_total_fp / run_total_samples_er
                per_run_error_rate_values.append(current_run_er_rate)

    # Calculate mean and CI for Cache Hit Rate
    ch_mean, ch_ci_low, ch_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_cache_hit_rate_values, z, clamp_min=0.0, clamp_max=1.0
    )

    # Calculate mean and CI for Error Rate
    er_mean, er_ci_low, er_ci_up = __calculate_mean_and_ci_from_runs(
        per_run_error_rate_values, z, clamp_min=0.0, clamp_max=1.0
    )

    return {
        "cache_hit_rate_mean": ch_mean,
        "cache_hit_rate_ci_low": ch_ci_low,
        "cache_hit_rate_ci_up": ch_ci_up,
        "error_rate_mean": er_mean,
        "error_rate_ci_low": er_ci_low,
        "error_rate_ci_up": er_ci_up,
    }


# End of new helper functions


def __get_result_files(results_dir: str):
    if not os.path.exists(results_dir):
        print(f"No results found in {results_dir}")
        return [], [], [], [], [], [], [], []

    gptcache_files: List[str] = []
    vcache_local_files: List[str] = []
    vcache_splitter_files: List[str] = []
    vcache_global_files: List[str] = []
    berkeley_embedding_files: List[str] = []
    vcache_berkeley_embedding_files: List[str] = []
    sigmoid_probability_files: List[str] = []
    sigmoid_only_files: List[str] = []

    for d in os.listdir(results_dir):
        # Process GPTCache (static threshold) directories
        if (d.startswith("gptcache_") or d.startswith("static_")) and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    gptcache_files.append(os.path.join(dir_path, file))

        # Process vCache local directories
        elif (
            d.startswith("vcache_local_") or d.startswith("vectorq_local_")
        ) and os.path.isdir(os.path.join(results_dir, d)):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_local_files.append(os.path.join(dir_path, file))

        elif d.startswith("vcache_splitter_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_splitter_files.append(os.path.join(dir_path, file))

        # Process vCache global directories
        elif (
            d.startswith("vcache_global_") or d.startswith("vectorq_global_")
        ) and os.path.isdir(os.path.join(results_dir, d)):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_global_files.append(os.path.join(dir_path, file))

        # Process Fine-tuned Embedding directories
        elif d.startswith("berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    berkeley_embedding_files.append(os.path.join(dir_path, file))

        # Process vCache Fine-tuned Embedding directories
        elif d.startswith("vcache_berkeley_embedding_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    vcache_berkeley_embedding_files.append(os.path.join(dir_path, file))

        # Process Sigmoid Probability directories
        elif d.startswith("sigmoid_probability_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    sigmoid_probability_files.append(os.path.join(dir_path, file))

        # Process Sigmoid Only directories
        elif d.startswith("sigmoid_only_") and os.path.isdir(
            os.path.join(results_dir, d)
        ):
            dir_path: str = os.path.join(results_dir, d)
            for file in os.listdir(dir_path):
                if file.startswith("results_") and file.endswith(".json"):
                    sigmoid_only_files.append(os.path.join(dir_path, file))

    return (
        gptcache_files,
        vcache_local_files,
        vcache_splitter_files,
        vcache_global_files,
        berkeley_embedding_files,
        vcache_berkeley_embedding_files,
        sigmoid_probability_files,
        sigmoid_only_files,
    )


def generate_combined_plots(
    dataset: str,
    embedding_model_name: str,
    llm_model_name: str,
    results_dir: str,
    timestamp: str,
    font_size: int,
    keep_split: int = 100,
):
    resolved_dataset = dataset
    results_dir: str = f"{results_dir}/{resolved_dataset}/{embedding_model_name}/{llm_model_name}/"

    if not os.path.exists(results_dir):
        alt_datasets: List[str] = []
        if resolved_dataset.endswith(".csv") or resolved_dataset.endswith(".parquet"):
            alt_datasets.append(os.path.splitext(resolved_dataset)[0])
        else:
            alt_datasets.append(resolved_dataset + ".csv")
            alt_datasets.append(resolved_dataset + ".parquet")

        for alt_ds in alt_datasets:
            alt_dir = f"{os.path.dirname(os.path.dirname(results_dir))}/{alt_ds}/{embedding_model_name}/{llm_model_name}/"
            if os.path.exists(alt_dir):
                resolved_dataset = alt_ds
                results_dir = alt_dir
                break

    (
        gptcache_files,
        vcache_local_files,
        vcache_splitter_files,
        vcache_global_files,
        berkeley_embedding_files,
        vcache_berkeley_embedding_files,
        sigmoid_probability_files,
        sigmoid_only_files,
    ) = __get_result_files(results_dir)

    if (
        not gptcache_files
        and not vcache_local_files
        and not vcache_splitter_files
        and not vcache_global_files
        and not berkeley_embedding_files
        and not vcache_berkeley_embedding_files
        and not sigmoid_probability_files
        and not sigmoid_only_files
    ):
        print(
            f"No folders found for {dataset}, {embedding_model_name}, {llm_model_name}\n"
            f"in {results_dir}"
        )
        return

    chopped_index = None

    ############################################################
    ### Baseline: GPTCache
    gptcache_data_frames: Dict[float, pd.DataFrame] = {}
    for gptcache_file_path in gptcache_files:
        try:
            with open(gptcache_file_path, "r") as f:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                threshold: float = data["config"]["threshold"]
                gptcache_data_frames[threshold] = dataframe
                chopped_index = chopped_index
        except Exception as e:
            print(f"Error loading {gptcache_file_path}: {e}")
            continue

    ############################################################
    ### Baseline: vCache Local
    vcache_local_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_local_file_path in vcache_local_files:
        try:
            with open(vcache_local_file_path, "r") as f:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                delta: float = data["config"]["delta"]
                vcache_local_data_frames[delta] = dataframe
                chopped_index = chopped_index
        except Exception as e:
            print(f"Error loading {vcache_local_file_path}: {e}")
            continue

    ############################################################
    ### Baseline: vCache Splitter
    # Support multiple splitter configs (top_k/all, candidate_k sweeps):
    # select best config per delta and only plot that.
    vcache_splitter_data_frames: Dict[float, pd.DataFrame] = {}
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None

    splitter_dirs_by_delta_cfg = __collect_splitter_run_dirs_by_delta_and_config(
        results_dir
    )
    if splitter_dirs_by_delta_cfg:
        __plot_splitter_k_sweep_for_deltas(
            results_dir=results_dir,
            splitter_dirs_by_delta_cfg=splitter_dirs_by_delta_cfg,
            timestamp=timestamp,
            font_size=font_size,
        )
        vcache_splitter_run_dirs_map_override = {}
        for delta_val, cfg_map in splitter_dirs_by_delta_cfg.items():
            best_cfg = None
            best_stats = None
            for cfg_key, run_dirs in cfg_map.items():
                stats = __aggregate_stats_for_cache_hit_error_rate(run_dirs)
                if best_stats is None:
                    best_stats = stats
                    best_cfg = cfg_key
                    continue

                # Prefer higher cache hit rate; tie-breaker lower error rate.
                if stats["cache_hit_rate_mean"] > best_stats["cache_hit_rate_mean"]:
                    best_stats = stats
                    best_cfg = cfg_key
                elif stats["cache_hit_rate_mean"] == best_stats["cache_hit_rate_mean"]:
                    if stats["error_rate_mean"] < best_stats["error_rate_mean"]:
                        best_stats = stats
                        best_cfg = cfg_key

            if best_cfg is None:
                continue

            selected_run_dirs = cfg_map.get(best_cfg, [])
            if not selected_run_dirs:
                continue

            key_str = __normalize_float_key_str(float(delta_val))
            vcache_splitter_run_dirs_map_override[key_str] = selected_run_dirs

            # Load a representative dataframe for this delta from the first selected run dir.
            loaded_df = None
            for file_name in os.listdir(selected_run_dirs[0]):
                if file_name.startswith("results_") and file_name.endswith(".json"):
                    try:
                        with open(
                            os.path.join(selected_run_dirs[0], file_name), "r"
                        ) as f:
                            data: Any = json.load(f)
                        loaded_df, _, chopped_index = convert_to_dataframe_from_json_file(
                            json_data=data, keep_split=keep_split
                        )
                        break
                    except Exception:
                        loaded_df = None
                        continue
            if loaded_df is not None:
                vcache_splitter_data_frames[float(delta_val)] = loaded_df
    else:
        # Backward-compatible path (single splitter config): keep existing behavior.
        for vcache_splitter_file_path in vcache_splitter_files:
            try:
                with open(vcache_splitter_file_path, "r") as f:
                    data: Any = json.load(f)
                    dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                        json_data=data, keep_split=keep_split
                    )
                    delta: float = data["config"]["delta"]
                    vcache_splitter_data_frames[delta] = dataframe
                    chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {vcache_splitter_file_path}: {e}")
                continue

    ############################################################
    ### Baseline: vCache Global
    vcache_global_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_global_file_path in vcache_global_files:
        with open(vcache_global_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                delta: float = data["config"]["delta"]
                vcache_global_data_frames[delta] = dataframe
                chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {vcache_global_file_path}: {e}")
                continue

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame] = {}
    for berkeley_embedding_file_path in berkeley_embedding_files:
        with open(berkeley_embedding_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                threshold: float = data["config"]["threshold"]
                berkeley_embedding_data_frames[threshold] = dataframe
                chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {berkeley_embedding_file_path}: {e}")
                continue

    ############################################################
    ### vCache + Fine-tuned Embedding
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame] = {}
    for vcache_berkeley_embedding_file_path in vcache_berkeley_embedding_files:
        with open(vcache_berkeley_embedding_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                delta: float = data["config"]["delta"]
                vcache_berkeley_embedding_data_frames[delta] = dataframe
                chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {vcache_berkeley_embedding_file_path}: {e}")
                continue

    ############################################################
    ### Sigmoid Probability
    sigmoid_probability_data_frames: Dict[float, pd.DataFrame] = {}
    for sigmoid_probability_file_path in sigmoid_probability_files:
        with open(sigmoid_probability_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                delta: float = data["config"]["delta"]
                sigmoid_probability_data_frames[delta] = dataframe
                chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {sigmoid_probability_file_path}: {e}")
                continue

    ############################################################
    ### Sigmoid Only
    sigmoid_only_data_frames: Dict[float, pd.DataFrame] = {}
    for sigmoid_only_file_path in sigmoid_only_files:
        with open(sigmoid_only_file_path, "r") as f:
            try:
                data: Any = json.load(f)
                dataframe, _, chopped_index = convert_to_dataframe_from_json_file(
                    json_data=data, keep_split=keep_split
                )
                delta: float = data["config"]["delta"]
                sigmoid_only_data_frames[delta] = dataframe
                chopped_index = chopped_index
            except Exception as e:
                print(f"Error loading {sigmoid_only_file_path}: {e}")
                continue

    if chopped_index is None:
        print(
            f"No data found for {dataset}, {embedding_model_name}, {llm_model_name} in {results_dir}"
        )
        return

    __plot_legend(
        gptcache_data_frames=gptcache_data_frames,
        vcache_local_data_frames=vcache_local_data_frames,
        vcache_splitter_data_frames=vcache_splitter_data_frames,
        vcache_global_data_frames=vcache_global_data_frames,
        berkeley_embedding_data_frames=berkeley_embedding_data_frames,
        vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
        sigmoid_probability_data_frames=sigmoid_probability_data_frames,
        sigmoid_only_data_frames=sigmoid_only_data_frames,
        results_dir=results_dir,
        timestamp=timestamp,
        font_size=font_size,
    )

    try:
        __plot_roc(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            keep_split=keep_split,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        print(f"Error plotting ROC: {e}")

    try:
        __plot_precision_vs_recall(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        print(f"Error plotting precision vs recall: {e}")

    try:
        __plot_avg_latency_vs_error_rate(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            sigmoid_probability_data_frames=sigmoid_probability_data_frames,
            sigmoid_only_data_frames=sigmoid_only_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        print(f"Error plotting avg latency vs error rate: {e}")

    try:
        __plot_cache_hit_vs_error_rate(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            sigmoid_probability_data_frames=sigmoid_probability_data_frames,
            sigmoid_only_data_frames=sigmoid_only_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        print(f"Error plotting cache hit vs error rate: {e}")

    try:
        __plot_cache_hit_vs_error_rate_vs_sample_size(
            gptcache_data_frames=gptcache_data_frames,
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            berkeley_embedding_data_frames=berkeley_embedding_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            keep_split=keep_split,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        # This plot is a nice-to-have; do not crash the whole plotting pipeline.
        # Some result folders may not contain all deltas/configs, which can raise
        # KeyError/ValueError in internal plotting logic.
        try:
            print(
                f"Error plotting cache hit vs error rate vs sample size: {type(e).__name__}: {e}"
            )
        except Exception:
            pass

    try:
        __plot_delta_accuracy(
            vcache_local_data_frames=vcache_local_data_frames,
            vcache_splitter_data_frames=vcache_splitter_data_frames,
            vcache_global_data_frames=vcache_global_data_frames,
            vcache_berkeley_embedding_data_frames=vcache_berkeley_embedding_data_frames,
            results_dir=results_dir,
            timestamp=timestamp,
            font_size=font_size,
            chopped_index=chopped_index,
            vcache_splitter_run_dirs_map_override=vcache_splitter_run_dirs_map_override,
        )
    except Exception as e:
        print(f"Error plotting delta accuracy: {e}")


def __plot_legend(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    sigmoid_probability_data_frames: Dict[float, pd.DataFrame],
    sigmoid_only_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
):
    figlegend = plt.figure(figsize=(12, 1.5))
    ax = figlegend.add_subplot(111)
    ax.axis("off")

    lines = []
    labels = []

    if gptcache_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#C23B48",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("GPTCache")

    if vcache_local_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#37A9EC",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache")

    if vcache_splitter_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#9B59B6",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache (Splitter)")

    if vcache_global_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#8CBE94",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache (Ablation)")

    if vcache_berkeley_embedding_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#EDBE24",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("vCache + Fine-tuned Embedding")

    if berkeley_embedding_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#3B686A",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("Fine-tuned Embedding")

    if sigmoid_probability_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#89D572",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("Sigmoid Probability")

    if sigmoid_only_data_frames:
        lines.append(
            Line2D(
                [0],
                [0],
                color="#E2A043",
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=8,
            )
        )
        labels.append("Sigmoid Only")

    ax.legend(lines, labels, loc="center", ncol=2, fontsize=font_size, frameon=False)

    legend_filename = results_dir + "/legend.pdf"
    figlegend.savefig(
        legend_filename, format="pdf", bbox_inches="tight", transparent=True
    )
    plt.close(figlegend)

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="--", alpha=0.7))
    labels.append("Random Classifier")

    lines.append(Line2D([0], [0], color="grey", linewidth=3, linestyle="-"))
    labels.append("No Cache")

    ax.legend(lines, labels, loc="center", ncol=3, fontsize=font_size, frameon=False)

    legend_filename = results_dir + "/legend_w_rnd_class.pdf"
    figlegend.savefig(
        legend_filename, format="pdf", bbox_inches="tight", transparent=True
    )
    plt.close(figlegend)


def __plot_roc(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    keep_split: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    plt.figure(figsize=(12, 10))

    # Collect all run directories once
    # base_results_dir = os.path.dirname(results_dir.rstrip('/')) # Get parent of timestamped dir if any
    # Or, if results_dir is the one containing gptcache_X.Y folders:
    # base_results_dir = results_dir # This should be correct based on how results_dir is constructed

    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_splitter_run_dirs_map = (
        vcache_splitter_run_dirs_map_override
        if vcache_splitter_run_dirs_map_override is not None
        else __collect_run_dirs_by_prefix_and_key(results_dir, "vcache_splitter_")
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        alpha=0.7,
        linewidth=3,
        label="Random Classifier",
    )

    # Helper to prepare data for a series
    def prepare_roc_series_data(data_frames, run_dirs_map_for_series, keep_split: int):
        thresholds_or_deltas = sorted(data_frames.keys())
        tpr_means, fpr_means = [], []
        tpr_low_err, tpr_up_err = [], []
        fpr_low_err, fpr_up_err = [], []
        is_multi = []

        for key_val in thresholds_or_deltas:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            if len(current_run_dirs) > 1:  # Multi-run
                tpr_m, tpr_l, tpr_u, fpr_m, fpr_l, fpr_u = __aggregate_stats_for_roc(
                    run_dirs=current_run_dirs, keep_split=keep_split
                )
                tpr_means.append(tpr_m)
                fpr_means.append(fpr_m)
                tpr_low_err.append(tpr_m - tpr_l)
                tpr_up_err.append(tpr_u - tpr_m)
                fpr_low_err.append(fpr_m - fpr_l)
                fpr_up_err.append(fpr_u - fpr_m)
                is_multi.append(True)
            elif (
                data_frames
            ):  # Single run (or fallback to single file if no dirs found but df exists)
                df = data_frames[key_val]
                tpr_m = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])
                fpr_m = compute_false_positive_rate_score(
                    fp=df["fp_list"], tn=df["tn_list"]
                )
                tpr_means.append(tpr_m)
                fpr_means.append(fpr_m)
                tpr_low_err.append(0)
                tpr_up_err.append(0)
                fpr_low_err.append(0)
                fpr_up_err.append(0)
                is_multi.append(False)

        # Ensure errors are non-negative
        tpr_low_err = [max(0, err) for err in tpr_low_err]
        tpr_up_err = [max(0, err) for err in tpr_up_err]
        fpr_low_err = [max(0, err) for err in fpr_low_err]
        fpr_up_err = [max(0, err) for err in fpr_up_err]

        return (
            fpr_means,
            tpr_means,
            fpr_low_err,
            fpr_up_err,
            tpr_low_err,
            tpr_up_err,
            is_multi,
        )

    ############################################################
    ### Baseline: GPTCache
    if gptcache_data_frames:
        gpt_fpr, gpt_tpr, gpt_fpr_le, gpt_fpr_ue, gpt_tpr_le, gpt_tpr_ue, gpt_multi = (
            prepare_roc_series_data(
                data_frames=gptcache_data_frames,
                run_dirs_map_for_series=gptcache_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            gpt_fpr,
            gpt_tpr,
            gpt_fpr_le,
            gpt_fpr_ue,
            gpt_tpr_le,
            gpt_tpr_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            10,
        )

    ############################################################
    ### Baseline: vCache Local
    if vcache_local_data_frames:
        vl_fpr, vl_tpr, vl_fpr_le, vl_fpr_ue, vl_tpr_le, vl_tpr_ue, vl_multi = (
            prepare_roc_series_data(
                data_frames=vcache_local_data_frames,
                run_dirs_map_for_series=vcache_local_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            vl_fpr,
            vl_tpr,
            vl_fpr_le,
            vl_fpr_ue,
            vl_tpr_le,
            vl_tpr_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            10,
        )

    ############################################################
    ### Baseline: vCache Splitter
    if vcache_splitter_data_frames:
        vs_fpr, vs_tpr, vs_fpr_le, vs_fpr_ue, vs_tpr_le, vs_tpr_ue, vs_multi = (
            prepare_roc_series_data(
                data_frames=vcache_splitter_data_frames,
                run_dirs_map_for_series=vcache_splitter_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            vs_fpr,
            vs_tpr,
            vs_fpr_le,
            vs_fpr_ue,
            vs_tpr_le,
            vs_tpr_ue,
            vs_multi,
            "#9B59B6",
            "vCache (Splitter)",
            10,
        )

    ############################################################
    ### Baseline: vCache Global
    if vcache_global_data_frames:
        vg_fpr, vg_tpr, vg_fpr_le, vg_fpr_ue, vg_tpr_le, vg_tpr_ue, vg_multi = (
            prepare_roc_series_data(
                data_frames=vcache_global_data_frames,
                run_dirs_map_for_series=vcache_global_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            vg_fpr,
            vg_tpr,
            vg_fpr_le,
            vg_fpr_ue,
            vg_tpr_le,
            vg_tpr_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            10,
        )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_fpr, be_tpr, be_fpr_le, be_fpr_ue, be_tpr_le, be_tpr_ue, be_multi = (
            prepare_roc_series_data(
                data_frames=berkeley_embedding_data_frames,
                run_dirs_map_for_series=berkeley_embedding_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            be_fpr,
            be_tpr,
            be_fpr_le,
            be_fpr_ue,
            be_tpr_le,
            be_tpr_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            10,
        )

    ############################################################
    ### vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_fpr, vb_tpr, vb_fpr_le, vb_fpr_ue, vb_tpr_le, vb_tpr_ue, vb_multi = (
            prepare_roc_series_data(
                data_frames=vcache_berkeley_embedding_data_frames,
                run_dirs_map_for_series=vcache_berkeley_embedding_run_dirs_map,
                keep_split=keep_split,
            )
        )
        __draw_confidence_series(
            vb_fpr,
            vb_tpr,
            vb_fpr_le,
            vb_fpr_ue,
            vb_tpr_le,
            vb_tpr_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            10,
        )

    plt.xlabel("FPR", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/roc_{chopped_index}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_precision_vs_recall(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    plt.figure(figsize=(12, 8))

    ############################################################
    ### Baseline: GPTCache
    gptcache_thresholds = sorted(gptcache_data_frames.keys())
    gptcache_precision_values = []
    gptcache_recall_values = []

    for threshold in gptcache_thresholds:
        df = gptcache_data_frames[threshold]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        gptcache_precision_values.append(precision)
        gptcache_recall_values.append(recall)

    if gptcache_thresholds:
        plt.plot(
            gptcache_recall_values,
            gptcache_precision_values,
            "o-",
            color="#C23B48",
            linewidth=3,
            label="GPTCache",
            markersize=8,
        )

        for i, threshold in enumerate(gptcache_thresholds):
            plt.annotate(
                text="",
                xy=(gptcache_recall_values[i], gptcache_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Local
    vcache_local_deltas = sorted(vcache_local_data_frames.keys())
    vcache_local_precision_values = []
    vcache_local_recall_values = []

    for delta in vcache_local_deltas:
        df = vcache_local_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_local_precision_values.append(precision)
        vcache_local_recall_values.append(recall)

    if vcache_local_deltas:
        plt.plot(
            vcache_local_recall_values,
            vcache_local_precision_values,
            "o-",
            color="#37A9EC",
            linewidth=3,
            label="vCache",
            markersize=8,
        )

        for i, _ in enumerate(vcache_local_precision_values):
            plt.annotate(
                text="",
                xy=(vcache_local_recall_values[i], vcache_local_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Splitter
    if vcache_splitter_data_frames:
        vcache_splitter_deltas = sorted(vcache_splitter_data_frames.keys())
        vcache_splitter_precision_values = []
        vcache_splitter_recall_values = []

        for delta in vcache_splitter_deltas:
            df = vcache_splitter_data_frames[delta]
            precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
            recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

            vcache_splitter_precision_values.append(precision)
            vcache_splitter_recall_values.append(recall)

        plt.plot(
            vcache_splitter_recall_values,
            vcache_splitter_precision_values,
            "o-",
            color="#9B59B6",
            linewidth=3,
            label="vCache (Splitter)",
            markersize=8,
        )

        for i, _ in enumerate(vcache_splitter_precision_values):
            plt.annotate(
                text="",
                xy=(
                    vcache_splitter_recall_values[i],
                    vcache_splitter_precision_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: vCache Global
    vcache_global_deltas = sorted(vcache_global_data_frames.keys())
    vcache_global_precision_values = []
    vcache_global_recall_values = []

    for delta in vcache_global_deltas:
        df = vcache_global_data_frames[delta]
        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_global_precision_values.append(precision)
        vcache_global_recall_values.append(recall)

    if vcache_global_deltas:
        plt.plot(
            vcache_global_recall_values,
            vcache_global_precision_values,
            "o-",
            color="#8CBE94",
            linewidth=3,
            label="vCache (Ablation)",
            markersize=8,
        )

        for i, delta in enumerate(vcache_global_deltas):
            plt.annotate(
                text="",
                xy=(vcache_global_recall_values[i], vcache_global_precision_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    berkeley_embedding_thresholds = sorted(berkeley_embedding_data_frames.keys())
    berkeley_embedding_precision_values = []
    berkeley_embedding_recall_values = []

    for threshold in berkeley_embedding_thresholds:
        df = berkeley_embedding_data_frames[threshold]

        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        berkeley_embedding_precision_values.append(precision)
        berkeley_embedding_recall_values.append(recall)

    if berkeley_embedding_thresholds:
        plt.plot(
            berkeley_embedding_recall_values,
            berkeley_embedding_precision_values,
            "o-",
            color="#3B686A",
            linewidth=3,
            label="Fine-tuned Embedding",
            markersize=8,
        )

        for i, threshold in enumerate(berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(
                    berkeley_embedding_recall_values[i],
                    berkeley_embedding_precision_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    ############################################################
    ### vCache + Fine-tuned Embedding
    vcache_berkeley_embedding_thresholds = sorted(
        vcache_berkeley_embedding_data_frames.keys()
    )
    vcache_berkeley_embedding_precision_values = []
    vcache_berkeley_embedding_recall_values = []

    for delta in vcache_berkeley_embedding_thresholds:
        df = vcache_berkeley_embedding_data_frames[delta]

        precision = compute_precision_score(tp=df["tp_list"], fp=df["fp_list"])
        recall = compute_recall_score(tp=df["tp_list"], fn=df["fn_list"])

        vcache_berkeley_embedding_precision_values.append(precision)
        vcache_berkeley_embedding_recall_values.append(recall)

    if vcache_berkeley_embedding_thresholds:
        plt.plot(
            vcache_berkeley_embedding_recall_values,
            vcache_berkeley_embedding_precision_values,
            "o-",
            color="#EDBE24",
            linewidth=3,
            label="vCache + Fine-tuned Embedding",
            markersize=8,
        )

        for i, delta in enumerate(vcache_berkeley_embedding_thresholds):
            plt.annotate(
                text="",
                xy=(
                    vcache_berkeley_embedding_recall_values[i],
                    vcache_berkeley_embedding_precision_values[i],
                ),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=font_size - 4,
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall", fontsize=font_size)
    plt.ylabel("Precision", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/precision_vs_recall_{chopped_index}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_avg_latency_vs_error_rate(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    sigmoid_probability_data_frames: Dict[float, pd.DataFrame],
    sigmoid_only_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    plt.figure(figsize=(12, 10))
    ERROR_RATE_UPPER_BOUND = 6  # 6%

    # Collect all run directories once
    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_splitter_run_dirs_map = (
        vcache_splitter_run_dirs_map_override
        if vcache_splitter_run_dirs_map_override is not None
        else __collect_run_dirs_by_prefix_and_key(results_dir, "vcache_splitter_")
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )
    sigmoid_probability_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "sigmoid_probability_"
    )
    sigmoid_only_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "sigmoid_only_"
    )

    avg_latency_no_cache = -1  # Initialize

    # Helper to prepare data for latency vs error rate
    def prepare_latency_error_series_data(
        data_frames, run_dirs_map_for_series, error_rate_bound
    ):
        keys = sorted(data_frames.keys())
        err_means, lat_means = [], []
        err_low_err, err_up_err = [], []
        lat_low_err, lat_up_err = [], []
        is_multi = []
        nonlocal avg_latency_no_cache  # To update from the first available df

        for key_val in keys:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            current_err_m, current_lat_m = -1, -1  # Init for filtering

            if len(current_run_dirs) > 1:  # Multi-run
                err_m, err_l, err_u, lat_m, lat_l, lat_u = (
                    __aggregate_stats_for_latency_error(current_run_dirs)
                )
                current_err_m = err_m * 100  # Convert to percentage
                current_lat_m = lat_m

                if current_err_m <= error_rate_bound:
                    err_means.append(current_err_m)
                    lat_means.append(current_lat_m)
                    err_low_err.append((err_m - err_l) * 100)
                    err_up_err.append((err_u - err_m) * 100)
                    lat_low_err.append(lat_m - lat_l)
                    lat_up_err.append(lat_u - lat_m)
                    is_multi.append(True)
            elif data_frames:  # Single run
                df = data_frames[key_val]
                err_m_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                lat_m_single = compute_avg_latency_score(
                    latency_list=df["latency_vectorq_list"]
                )
                current_err_m = err_m_single
                current_lat_m = lat_m_single

                if current_err_m <= error_rate_bound:
                    err_means.append(current_err_m)
                    lat_means.append(current_lat_m)
                    err_low_err.append(0)
                    err_up_err.append(0)
                    lat_low_err.append(0)
                    lat_up_err.append(0)
                    is_multi.append(False)

            # Update avg_latency_no_cache from the first valid df processed
            if avg_latency_no_cache == -1 and data_frames and key_val in data_frames:
                df_for_no_cache = data_frames[key_val]
                if (
                    "latency_direct_list" in df_for_no_cache.columns
                    and not df_for_no_cache["latency_direct_list"].empty
                ):
                    avg_latency_no_cache = compute_avg_latency_score(
                        latency_list=df_for_no_cache["latency_direct_list"]
                    )

        # Ensure errors are non-negative
        err_low_err = [max(0, err) for err in err_low_err]
        err_up_err = [max(0, err) for err in err_up_err]
        lat_low_err = [max(0, err) for err in lat_low_err]
        lat_up_err = [max(0, err) for err in lat_up_err]

        return (
            lat_means,
            err_means,
            lat_low_err,
            lat_up_err,
            err_low_err,
            err_up_err,
            is_multi,
        )

    ############################################################
    ### Baseline: GPTCache
    if gptcache_data_frames:
        gpt_lat, gpt_err, gpt_lat_le, gpt_lat_ue, gpt_err_le, gpt_err_ue, gpt_multi = (
            prepare_latency_error_series_data(
                gptcache_data_frames, gptcache_run_dirs_map, ERROR_RATE_UPPER_BOUND
            )
        )
        # Plotting x=latency, y=error_rate
        __draw_confidence_series(
            gpt_lat,
            gpt_err,
            gpt_lat_le,
            gpt_lat_ue,
            gpt_err_le,
            gpt_err_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            8,
        )

    ############################################################
    ### Baseline: No Cache (plotted after at least one series to get avg_latency_no_cache)
    # This logic is kept similar to original, plotted as a vertical line.

    ############################################################
    ### Baseline: vCache Local
    if vcache_local_data_frames:
        vl_lat, vl_err, vl_lat_le, vl_lat_ue, vl_err_le, vl_err_ue, vl_multi = (
            prepare_latency_error_series_data(
                vcache_local_data_frames,
                vcache_local_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vl_lat,
            vl_err,
            vl_lat_le,
            vl_lat_ue,
            vl_err_le,
            vl_err_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            8,
        )

    ############################################################
    ### Baseline: vCache Splitter
    if vcache_splitter_data_frames:
        vs_lat, vs_err, vs_lat_le, vs_lat_ue, vs_err_le, vs_err_ue, vs_multi = (
            prepare_latency_error_series_data(
                vcache_splitter_data_frames,
                vcache_splitter_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vs_lat,
            vs_err,
            vs_lat_le,
            vs_lat_ue,
            vs_err_le,
            vs_err_ue,
            vs_multi,
            "#9B59B6",
            "vCache (Splitter)",
            8,
        )

    ############################################################
    ### Baseline: vCache Global
    if vcache_global_data_frames:
        vg_lat, vg_err, vg_lat_le, vg_lat_ue, vg_err_le, vg_err_ue, vg_multi = (
            prepare_latency_error_series_data(
                vcache_global_data_frames,
                vcache_global_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vg_lat,
            vg_err,
            vg_lat_le,
            vg_lat_ue,
            vg_err_le,
            vg_err_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            8,
        )

    ############################################################
    ### Baseline: Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_lat, be_err, be_lat_le, be_lat_ue, be_err_le, be_err_ue, be_multi = (
            prepare_latency_error_series_data(
                berkeley_embedding_data_frames,
                berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            be_lat,
            be_err,
            be_lat_le,
            be_lat_ue,
            be_err_le,
            be_err_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            8,
        )

    ############################################################
    ### vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_lat, vb_err, vb_lat_le, vb_lat_ue, vb_err_le, vb_err_ue, vb_multi = (
            prepare_latency_error_series_data(
                vcache_berkeley_embedding_data_frames,
                vcache_berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vb_lat,
            vb_err,
            vb_lat_le,
            vb_lat_ue,
            vb_err_le,
            vb_err_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            8,
        )

    ############################################################
    ### Sigmoid Probability
    if sigmoid_probability_data_frames:
        sp_lat, sp_err, sp_lat_le, sp_lat_ue, sp_err_le, sp_err_ue, sp_multi = (
            prepare_latency_error_series_data(
                sigmoid_probability_data_frames,
                sigmoid_probability_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            sp_lat,
            sp_err,
            sp_lat_le,
            sp_lat_ue,
            sp_err_le,
            sp_err_ue,
            sp_multi,
            "#89D572",
            "Sigmoid Probability",
            8,
        )

    ############################################################
    ### Sigmoid Only
    if sigmoid_only_data_frames:
        so_lat, so_err, so_lat_le, so_lat_ue, so_err_le, so_err_ue, so_multi = (
            prepare_latency_error_series_data(
                sigmoid_only_data_frames,
                sigmoid_only_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            so_lat,
            so_err,
            so_lat_le,
            so_lat_ue,
            so_err_le,
            so_err_ue,
            so_multi,
            "#E2A043",
            "Sigmoid Only",
            8,
        )

    ############################################################
    ### Baseline: No Cache Plotting
    # Ensure avg_latency_no_cache has been set by one of the prepare_latency_error_series_data calls
    if avg_latency_no_cache != -1:  # Check if it was updated
        plt.axvline(
            x=avg_latency_no_cache,
            color="grey",
            linewidth=3,
            label="No Cache",
            zorder=1,  # Ensure it's behind most other elements
        )
        # Arrow annotation removed as it was empty and complex to replicate without specific text

    plt.xlabel("Average Latency (sec)", fontsize=font_size)
    plt.ylabel("Error Rate (%)", fontsize=font_size)
    plt.ylim(bottom=0.0)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    plt.gca().spines["top"].set_linewidth(1)
    plt.gca().spines["right"].set_linewidth(1)
    plt.gca().spines["bottom"].set_linewidth(1)
    plt.gca().spines["left"].set_linewidth(1)

    filename = results_dir + f"/avg_latency_vs_error_rate_{chopped_index}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_cache_hit_vs_error_rate(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    sigmoid_probability_data_frames: Dict[float, pd.DataFrame],
    sigmoid_only_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    plt.figure(figsize=(12, 10))
    ERROR_RATE_UPPER_BOUND = 8  # 6%

    # Collect run directories for each type
    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_splitter_run_dirs_map = (
        vcache_splitter_run_dirs_map_override
        if vcache_splitter_run_dirs_map_override is not None
        else __collect_run_dirs_by_prefix_and_key(results_dir, "vcache_splitter_")
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )
    sigmoid_probability_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "sigmoid_probability_"
    )
    sigmoid_only_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "sigmoid_only_"
    )

    # Helper to prepare data for a series
    def prepare_cache_hit_error_series_data(
        data_frames, run_dirs_map_for_series, error_rate_bound
    ):
        keys = sorted(data_frames.keys())
        err_means, ch_means = [], []
        err_low_err, err_up_err = [], []
        ch_low_err, ch_up_err = [], []
        is_multi = []

        for key_val in keys:
            key_str = f"{key_val:.3f}".rstrip("0").rstrip(".")
            if not key_str and key_val == 0:
                key_str = "0"

            current_run_dirs = run_dirs_map_for_series.get(key_str, [])

            if len(current_run_dirs) > 1:  # Multi-run
                stats = __aggregate_stats_for_cache_hit_error_rate(current_run_dirs)
                if stats is None:
                    continue  # Should not happen with new return type

                er_m = stats["error_rate_mean"] * 100
                ch_m = stats["cache_hit_rate_mean"] * 100

                if er_m <= error_rate_bound:
                    err_means.append(er_m)
                    ch_means.append(ch_m)
                    # errors are relative to mean
                    err_low_err.append(
                        (stats["error_rate_mean"] - stats["error_rate_ci_low"]) * 100
                    )
                    err_up_err.append(
                        (stats["error_rate_ci_up"] - stats["error_rate_mean"]) * 100
                    )
                    ch_low_err.append(
                        (stats["cache_hit_rate_mean"] - stats["cache_hit_rate_ci_low"])
                        * 100
                    )
                    ch_up_err.append(
                        (stats["cache_hit_rate_ci_up"] - stats["cache_hit_rate_mean"])
                        * 100
                    )
                    is_multi.append(True)
            elif data_frames:  # Single run
                df = data_frames[key_val]
                er_m_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                ch_m_single = (
                    compute_cache_hit_rate_score(cache_hit_list=df["cache_hit_list"])
                    * 100
                )

                if er_m_single <= error_rate_bound:
                    err_means.append(er_m_single)
                    ch_means.append(ch_m_single)
                    err_low_err.append(0)
                    err_up_err.append(0)
                    ch_low_err.append(0)
                    ch_up_err.append(0)
                    is_multi.append(False)

        # Ensure errors are non-negative
        err_low_err = [max(0, err) for err in err_low_err]
        err_up_err = [max(0, err) for err in err_up_err]
        ch_low_err = [max(0, err) for err in ch_low_err]
        ch_up_err = [max(0, err) for err in ch_up_err]

        return (
            err_means,
            ch_means,
            err_low_err,
            err_up_err,
            ch_low_err,
            ch_up_err,
            is_multi,
        )

    # GPTCache
    if gptcache_data_frames:
        gpt_err, gpt_ch, gpt_err_le, gpt_err_ue, gpt_ch_le, gpt_ch_ue, gpt_multi = (
            prepare_cache_hit_error_series_data(
                gptcache_data_frames, gptcache_run_dirs_map, ERROR_RATE_UPPER_BOUND
            )
        )
        __draw_confidence_series(
            gpt_err,
            gpt_ch,
            gpt_err_le,
            gpt_err_ue,
            gpt_ch_le,
            gpt_ch_ue,
            gpt_multi,
            "#C23B48",
            "GPTCache",
            10,
        )  # x=error, y=cache_hit

    # vCache Local
    if vcache_local_data_frames:
        vl_err, vl_ch, vl_err_le, vl_err_ue, vl_ch_le, vl_ch_ue, vl_multi = (
            prepare_cache_hit_error_series_data(
                vcache_local_data_frames,
                vcache_local_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vl_err,
            vl_ch,
            vl_err_le,
            vl_err_ue,
            vl_ch_le,
            vl_ch_ue,
            vl_multi,
            "#37A9EC",
            "vCache",
            8,
        )

    # vCache Splitter
    if vcache_splitter_data_frames:
        vs_err, vs_ch, vs_err_le, vs_err_ue, vs_ch_le, vs_ch_ue, vs_multi = (
            prepare_cache_hit_error_series_data(
                vcache_splitter_data_frames,
                vcache_splitter_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vs_err,
            vs_ch,
            vs_err_le,
            vs_err_ue,
            vs_ch_le,
            vs_ch_ue,
            vs_multi,
            "#9B59B6",
            "vCache (Splitter)",
            8,
        )

    # vCache Global
    if vcache_global_data_frames:
        vg_err, vg_ch, vg_err_le, vg_err_ue, vg_ch_le, vg_ch_ue, vg_multi = (
            prepare_cache_hit_error_series_data(
                vcache_global_data_frames,
                vcache_global_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vg_err,
            vg_ch,
            vg_err_le,
            vg_err_ue,
            vg_ch_le,
            vg_ch_ue,
            vg_multi,
            "#8CBE94",
            "vCache (Ablation)",
            8,
        )

    # Fine-tuned Embedding
    if berkeley_embedding_data_frames:
        be_err, be_ch, be_err_le, be_err_ue, be_ch_le, be_ch_ue, be_multi = (
            prepare_cache_hit_error_series_data(
                berkeley_embedding_data_frames,
                berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            be_err,
            be_ch,
            be_err_le,
            be_err_ue,
            be_ch_le,
            be_ch_ue,
            be_multi,
            "#3B686A",
            "Fine-tuned Embedding",
            8,
        )

    # vCache + Fine-tuned Embedding
    if vcache_berkeley_embedding_data_frames:
        vb_err, vb_ch, vb_err_le, vb_err_ue, vb_ch_le, vb_ch_ue, vb_multi = (
            prepare_cache_hit_error_series_data(
                vcache_berkeley_embedding_data_frames,
                vcache_berkeley_embedding_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            vb_err,
            vb_ch,
            vb_err_le,
            vb_err_ue,
            vb_ch_le,
            vb_ch_ue,
            vb_multi,
            "#EDBE24",
            "vCache + Fine-tuned Embedding",
            8,
        )

    # Sigmoid Probability
    if sigmoid_probability_data_frames:
        sp_err, sp_ch, sp_err_le, sp_err_ue, sp_ch_le, sp_ch_ue, sp_multi = (
            prepare_cache_hit_error_series_data(
                sigmoid_probability_data_frames,
                sigmoid_probability_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            sp_err,
            sp_ch,
            sp_err_le,
            sp_err_ue,
            sp_ch_le,
            sp_ch_ue,
            sp_multi,
            "#89D572",
            "Sigmoid Probability",
            8,
        )

    # Sigmoid Only
    if sigmoid_only_data_frames:
        so_err, so_ch, so_err_le, so_err_ue, so_ch_le, so_ch_ue, so_multi = (
            prepare_cache_hit_error_series_data(
                sigmoid_only_data_frames,
                sigmoid_only_run_dirs_map,
                ERROR_RATE_UPPER_BOUND,
            )
        )
        __draw_confidence_series(
            so_err,
            so_ch,
            so_err_le,
            so_err_ue,
            so_ch_le,
            so_ch_ue,
            so_multi,
            "#E2A043",
            "Sigmoid Only",
            8,
        )

    plt.xlabel("Error Rate (%)", fontsize=font_size)
    plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
    plt.tick_params(axis="both", labelsize=font_size - 2)

    yticks = plt.yticks()[0]
    if yticks[0] == 0.0:
        plt.yticks(yticks[1:])

    for spine in ["top", "right", "bottom", "left"]:
        plt.gca().spines[spine].set_linewidth(1)

    filename = results_dir + f"/cache_hit_vs_error_rate_{chopped_index}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()


def __plot_cache_hit_vs_error_rate_vs_sample_size(
    gptcache_data_frames: Dict[float, pd.DataFrame],
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    keep_split: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    # Collect all run directories once
    gptcache_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "gptcache_"
    )
    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )
    vcache_splitter_run_dirs_map = (
        vcache_splitter_run_dirs_map_override
        if vcache_splitter_run_dirs_map_override is not None
        else __collect_run_dirs_by_prefix_and_key(results_dir, "vcache_splitter_")
    )
    vcache_global_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_global_"
    )
    berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "berkeley_embedding_"
    )
    vcache_berkeley_embedding_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_berkeley_embedding_"
    )

    # Use all deltas that actually exist in the results.
    # Target error rate for each delta is the *actual* vCache Local error rate.
    target_deltas = []
    target_error_rates = []
    for delta_val in sorted(vcache_local_data_frames.keys()):
        delta_key_str = __normalize_float_key_str(delta_val)
        local_run_dirs = vcache_local_run_dirs_map.get(delta_key_str, [])
        if len(local_run_dirs) > 1:
            stats = __aggregate_stats_for_cache_hit_error_rate(local_run_dirs)
            if stats is None:
                continue
            target_deltas.append(delta_val)
            target_error_rates.append(stats["error_rate_mean"] * 100)
        else:
            df = vcache_local_data_frames[delta_val]
            target_deltas.append(delta_val)
            target_error_rates.append(compute_error_rate_score(fp=df["fp_list"]) * 100)

    # Helper function to compute cumulative metrics with confidence intervals
    def compute_cumulative_metrics_with_ci(run_dirs, delta_key_str, z=1.96):
        per_run_fps = []
        per_run_cache_hits = []

        # First collect fps and cache_hits for each run
        for rd_path in run_dirs:
            run_fps = []
            run_cache_hits = []

            for file_name in os.listdir(rd_path):
                if file_name.startswith("results_") and file_name.endswith(".json"):
                    file_path = os.path.join(rd_path, file_name)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        df, _, _ = convert_to_dataframe_from_json_file(
                            json_data=data, keep_split=keep_split
                        )
                        run_fps.extend(df["fp_list"])
                        run_cache_hits.extend(df["cache_hit_list"])
                    except Exception:
                        continue

            if run_fps:
                per_run_fps.append(run_fps)

            if run_cache_hits:
                per_run_cache_hits.append(run_cache_hits)

        # Ensure all runs have the same length by truncating to the minimum length
        if per_run_fps:
            min_len = min(len(fps) for fps in per_run_fps)
            per_run_fps = [fps[:min_len] for fps in per_run_fps]

            # Calculate cumulative sums and rates for each run
            per_run_cumulative_error_rates = []
            for fps in per_run_fps:
                cum_sum = np.cumsum(fps)
                indices = np.arange(1, len(fps) + 1)
                cum_rates = cum_sum / indices
                per_run_cumulative_error_rates.append(cum_rates)

            # Calculate mean and CI at each position
            error_means = []
            error_ci_lows = []
            error_ci_ups = []

            for i in range(min_len):
                values = [rates[i] for rates in per_run_cumulative_error_rates]
                mean, ci_low, ci_up = __calculate_mean_and_ci_from_runs(
                    values, z, clamp_min=0.0, clamp_max=1.0
                )
                error_means.append(mean * 100)  # Convert to percentage
                error_ci_lows.append((mean - ci_low) * 100)
                error_ci_ups.append((ci_up - mean) * 100)
        else:
            error_means, error_ci_lows, error_ci_ups = [], [], []
            min_len = 0

        if per_run_cache_hits:
            min_len_ch = min(len(hits) for hits in per_run_cache_hits)
            min_len = min_len_ch if not min_len else min(min_len, min_len_ch)
            per_run_cache_hits = [hits[:min_len] for hits in per_run_cache_hits]

            # Calculate cumulative sums and rates for each run
            per_run_cumulative_cache_hit_rates = []
            for hits in per_run_cache_hits:
                cum_sum = np.cumsum(hits)
                indices = np.arange(1, len(hits) + 1)
                cum_rates = cum_sum / indices
                per_run_cumulative_cache_hit_rates.append(cum_rates)

            # Calculate mean and CI at each position
            cache_hit_means = []
            cache_hit_ci_lows = []
            cache_hit_ci_ups = []

            for i in range(min_len):
                values = [rates[i] for rates in per_run_cumulative_cache_hit_rates]
                mean, ci_low, ci_up = __calculate_mean_and_ci_from_runs(
                    values, z, clamp_min=0.0, clamp_max=1.0
                )
                cache_hit_means.append(mean * 100)  # Convert to percentage
                cache_hit_ci_lows.append((mean - ci_low) * 100)
                cache_hit_ci_ups.append((ci_up - mean) * 100)
        else:
            cache_hit_means, cache_hit_ci_lows, cache_hit_ci_ups = [], [], []

        return (
            error_means,
            error_ci_lows,
            error_ci_ups,
            cache_hit_means,
            cache_hit_ci_lows,
            cache_hit_ci_ups,
            min_len,
        )

    for target_delta, target_error_rate in zip(target_deltas, target_error_rates):
        if target_delta not in vcache_local_data_frames:
            continue
        ############################################################
        ### Baseline: vCache Local
        vcache_local_df = vcache_local_data_frames[target_delta]
        delta_key_str = f"{target_delta:.3f}".rstrip("0").rstrip(".")
        if not delta_key_str and target_delta == 0:
            delta_key_str = "0"

        local_run_dirs = vcache_local_run_dirs_map.get(delta_key_str, [])

        if len(local_run_dirs) > 1:
            (
                vcache_local_error_rates,
                vcache_local_error_ci_lows,
                vcache_local_error_ci_ups,
                vcache_local_cache_hit_rates,
                vcache_local_cache_hit_ci_lows,
                vcache_local_cache_hit_ci_ups,
                vcache_local_samples,
            ) = compute_cumulative_metrics_with_ci(local_run_dirs, delta_key_str)
        else:
            vcache_local_error_rates = [
                rate * 100
                for rate in compute_error_rate_cumulative_list(
                    fp=vcache_local_df["fp_list"]
                )
            ]
            vcache_local_cache_hit_rates = [
                rate * 100
                for rate in compute_cache_hit_rate_cumulative_list(
                    cache_hit_list=vcache_local_df["cache_hit_list"]
                )
            ]
            vcache_local_samples = len(vcache_local_error_rates)
            vcache_local_error_ci_lows = [0] * vcache_local_samples
            vcache_local_error_ci_ups = [0] * vcache_local_samples
            vcache_local_cache_hit_ci_lows = [0] * vcache_local_samples
            vcache_local_cache_hit_ci_ups = [0] * vcache_local_samples

        ############################################################
        ### Baseline: vCache Splitter
        delta_key_str = f"{target_delta:.3f}".rstrip("0").rstrip(".")
        if not delta_key_str and target_delta == 0:
            delta_key_str = "0"

        splitter_run_dirs = vcache_splitter_run_dirs_map.get(delta_key_str, [])
        if vcache_splitter_data_frames and target_delta in vcache_splitter_data_frames:
            df = vcache_splitter_data_frames[target_delta]
            if len(splitter_run_dirs) > 1:
                (
                    vcache_splitter_error_rates,
                    vcache_splitter_error_ci_lows,
                    vcache_splitter_error_ci_ups,
                    vcache_splitter_cache_hit_rates,
                    vcache_splitter_cache_hit_ci_lows,
                    vcache_splitter_cache_hit_ci_ups,
                    vcache_splitter_samples,
                ) = compute_cumulative_metrics_with_ci(splitter_run_dirs, delta_key_str)
            else:
                vcache_splitter_error_rates = [
                    rate * 100
                    for rate in compute_error_rate_cumulative_list(fp=df["fp_list"])
                ]
                vcache_splitter_cache_hit_rates = [
                    rate * 100
                    for rate in compute_cache_hit_rate_cumulative_list(
                        cache_hit_list=df["cache_hit_list"]
                    )
                ]
                vcache_splitter_samples = len(vcache_splitter_error_rates)
                vcache_splitter_error_ci_lows = [0] * vcache_splitter_samples
                vcache_splitter_error_ci_ups = [0] * vcache_splitter_samples
                vcache_splitter_cache_hit_ci_lows = [0] * vcache_splitter_samples
                vcache_splitter_cache_hit_ci_ups = [0] * vcache_splitter_samples
        else:
            vcache_splitter_error_rates = []
            vcache_splitter_cache_hit_rates = []
            vcache_splitter_error_ci_lows = []
            vcache_splitter_error_ci_ups = []
            vcache_splitter_cache_hit_ci_lows = []
            vcache_splitter_cache_hit_ci_ups = []

        ############################################################
        ### Baseline: vCache Global
        if vcache_global_data_frames and target_delta in vcache_global_data_frames:
            vcache_global_df = vcache_global_data_frames[target_delta]
            global_run_dirs = vcache_global_run_dirs_map.get(delta_key_str, [])

            if len(global_run_dirs) > 1:
                (
                    vcache_global_error_rates,
                    vcache_global_error_ci_lows,
                    vcache_global_error_ci_ups,
                    vcache_global_cache_hit_rates,
                    vcache_global_cache_hit_ci_lows,
                    vcache_global_cache_hit_ci_ups,
                    vcache_global_samples,
                ) = compute_cumulative_metrics_with_ci(global_run_dirs, delta_key_str)
            else:
                vcache_global_error_rates = [
                    rate * 100
                    for rate in compute_error_rate_cumulative_list(
                        fp=vcache_global_df["fp_list"]
                    )
                ]
                vcache_global_cache_hit_rates = [
                    rate * 100
                    for rate in compute_cache_hit_rate_cumulative_list(
                        cache_hit_list=vcache_global_df["cache_hit_list"]
                    )
                ]
                vcache_global_samples = len(vcache_global_error_rates)
                vcache_global_error_ci_lows = [0] * vcache_global_samples
                vcache_global_error_ci_ups = [0] * vcache_global_samples
                vcache_global_cache_hit_ci_lows = [0] * vcache_global_samples
                vcache_global_cache_hit_ci_ups = [0] * vcache_global_samples

        ############################################################
        ### Baseline: GPTCache
        if gptcache_data_frames:
            gptcache_thresholds = sorted(gptcache_data_frames.keys())
            gptcache_closest_error_rate_diff = float("inf")
            gptcache_closest_threshold = None
            gptcache_df = None

            for threshold in gptcache_thresholds:
                df = gptcache_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)

                if error_rate_diff < gptcache_closest_error_rate_diff:
                    gptcache_closest_error_rate_diff = error_rate_diff
                    gptcache_closest_threshold = threshold
                    gptcache_df = df

            if gptcache_closest_threshold is not None:
                threshold_key_str = f"{gptcache_closest_threshold:.3f}".rstrip(
                    "0"
                ).rstrip(".")
                if not threshold_key_str and gptcache_closest_threshold == 0:
                    threshold_key_str = "0"

                gptcache_run_dirs = gptcache_run_dirs_map.get(threshold_key_str, [])

                if len(gptcache_run_dirs) > 1:
                    (
                        gptcache_error_rates,
                        gptcache_error_ci_lows,
                        gptcache_error_ci_ups,
                        gptcache_cache_hit_rates,
                        gptcache_cache_hit_ci_lows,
                        gptcache_cache_hit_ci_ups,
                        gptcache_samples,
                    ) = compute_cumulative_metrics_with_ci(
                        gptcache_run_dirs, threshold_key_str
                    )
                else:
                    gptcache_error_rates = [
                        rate * 100
                        for rate in compute_error_rate_cumulative_list(
                            fp=gptcache_df["fp_list"]
                        )
                    ]
                    gptcache_cache_hit_rates = [
                        rate * 100
                        for rate in compute_cache_hit_rate_cumulative_list(
                            cache_hit_list=gptcache_df["cache_hit_list"]
                        )
                    ]
                    gptcache_samples = len(gptcache_error_rates)
                    gptcache_error_ci_lows = [0] * gptcache_samples
                    gptcache_error_ci_ups = [0] * gptcache_samples
                    gptcache_cache_hit_ci_lows = [0] * gptcache_samples
                    gptcache_cache_hit_ci_ups = [0] * gptcache_samples

        ############################################################
        ### Baseline: Berkeley Embedding
        if berkeley_embedding_data_frames:
            berkeley_embedding_thresholds = sorted(
                berkeley_embedding_data_frames.keys()
            )
            berkeley_embedding_closest_error_rate_diff = float("inf")
            berkeley_embedding_closest_threshold = None
            berkeley_embedding_df = None

            for threshold in berkeley_embedding_thresholds:
                df = berkeley_embedding_data_frames[threshold]
                error_rate = compute_error_rate_score(fp=df["fp_list"]) * 100
                error_rate_diff = abs(error_rate - target_error_rate)

                if error_rate_diff < berkeley_embedding_closest_error_rate_diff:
                    berkeley_embedding_closest_error_rate_diff = error_rate_diff
                    berkeley_embedding_closest_threshold = threshold
                    berkeley_embedding_df = df

            if berkeley_embedding_closest_threshold is not None:
                threshold_key_str = (
                    f"{berkeley_embedding_closest_threshold:.3f}".rstrip("0").rstrip(
                        "."
                    )
                )
                if not threshold_key_str and berkeley_embedding_closest_threshold == 0:
                    threshold_key_str = "0"

                berkeley_embedding_run_dirs = berkeley_embedding_run_dirs_map.get(
                    threshold_key_str, []
                )

                if len(berkeley_embedding_run_dirs) > 1:
                    (
                        berkeley_embedding_error_rates,
                        berkeley_embedding_error_ci_lows,
                        berkeley_embedding_error_ci_ups,
                        berkeley_embedding_cache_hit_rates,
                        berkeley_embedding_cache_hit_ci_lows,
                        berkeley_embedding_cache_hit_ci_ups,
                        berkeley_embedding_samples,
                    ) = compute_cumulative_metrics_with_ci(
                        berkeley_embedding_run_dirs, threshold_key_str
                    )
                else:
                    berkeley_embedding_error_rates = [
                        rate * 100
                        for rate in compute_error_rate_cumulative_list(
                            fp=berkeley_embedding_df["fp_list"]
                        )
                    ]
                    berkeley_embedding_cache_hit_rates = [
                        rate * 100
                        for rate in compute_cache_hit_rate_cumulative_list(
                            cache_hit_list=berkeley_embedding_df["cache_hit_list"]
                        )
                    ]
                    berkeley_embedding_samples = len(berkeley_embedding_error_rates)
                    berkeley_embedding_error_ci_lows = [0] * berkeley_embedding_samples
                    berkeley_embedding_error_ci_ups = [0] * berkeley_embedding_samples
                    berkeley_embedding_cache_hit_ci_lows = [
                        0
                    ] * berkeley_embedding_samples
                    berkeley_embedding_cache_hit_ci_ups = [
                        0
                    ] * berkeley_embedding_samples

        ############################################################
        ### vCache + Berkeley Embedding
        if (
            vcache_berkeley_embedding_data_frames
            and target_delta in vcache_berkeley_embedding_data_frames
        ):
            vcache_berkeley_embedding_df = vcache_berkeley_embedding_data_frames[
                target_delta
            ]
            vcache_berkeley_run_dirs = vcache_berkeley_embedding_run_dirs_map.get(
                delta_key_str, []
            )

            if len(vcache_berkeley_run_dirs) > 1:
                (
                    vcache_berkeley_embedding_error_rates,
                    vcache_berkeley_embedding_error_ci_lows,
                    vcache_berkeley_embedding_error_ci_ups,
                    vcache_berkeley_embedding_cache_hit_rates,
                    vcache_berkeley_embedding_cache_hit_ci_lows,
                    vcache_berkeley_embedding_cache_hit_ci_ups,
                    vcache_berkeley_embedding_samples,
                ) = compute_cumulative_metrics_with_ci(
                    vcache_berkeley_run_dirs, delta_key_str
                )
            else:
                vcache_berkeley_embedding_error_rates = [
                    rate * 100
                    for rate in compute_error_rate_cumulative_list(
                        fp=vcache_berkeley_embedding_df["fp_list"]
                    )
                ]
                vcache_berkeley_embedding_cache_hit_rates = [
                    rate * 100
                    for rate in compute_cache_hit_rate_cumulative_list(
                        cache_hit_list=vcache_berkeley_embedding_df["cache_hit_list"]
                    )
                ]
                vcache_berkeley_embedding_samples = len(
                    vcache_berkeley_embedding_error_rates
                )
                vcache_berkeley_embedding_error_ci_lows = [
                    0
                ] * vcache_berkeley_embedding_samples
                vcache_berkeley_embedding_error_ci_ups = [
                    0
                ] * vcache_berkeley_embedding_samples
                vcache_berkeley_embedding_cache_hit_ci_lows = [
                    0
                ] * vcache_berkeley_embedding_samples
                vcache_berkeley_embedding_cache_hit_ci_ups = [
                    0
                ] * vcache_berkeley_embedding_samples

        # Adjust sample sizes to account for chopped_index
        # This shifts the x-axis to start from the actual sample index after chopping
        sample_sizes = np.arange(
            chopped_index + 1, chopped_index + len(vcache_local_error_rates) + 1
        )

        # Plot 1: Error rates vs sample size
        plt.figure(figsize=(12, 11))

        # Helper to add confidence band
        def add_confidence_band(x, y, y_low, y_up, color, alpha=0.2):
            if any(y_low) or any(y_up):  # Only add if there are actual CI values
                plt.fill_between(
                    x,
                    [max(0, y[i] - y_low[i]) for i in range(len(y))],
                    [min(100, y[i] + y_up[i]) for i in range(len(y))],
                    color=color,
                    alpha=alpha,
                )

        # vCache Local
        plt.plot(
            sample_sizes[::45],
            vcache_local_error_rates[::45],
            "-",
            color="#37A9EC",
            linewidth=2,
            label=f"vCache (δ={target_delta})",
            markersize=1,
        )
        add_confidence_band(
            sample_sizes[::45],
            vcache_local_error_rates[::45],
            vcache_local_error_ci_lows[::45],
            vcache_local_error_ci_ups[::45],
            "#37A9EC",
        )

        # vCache Splitter
        if vcache_splitter_data_frames and target_delta in vcache_splitter_data_frames:
            plt.plot(
                sample_sizes[::45],
                vcache_splitter_error_rates[::45],
                "-",
                color="#9B59B6",
                linewidth=2,
                label=f"vCache Splitter (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::45],
                vcache_splitter_error_rates[::45],
                vcache_splitter_error_ci_lows[::45],
                vcache_splitter_error_ci_ups[::45],
                "#9B59B6",
            )

        # vCache Global
        if vcache_global_data_frames and target_delta in vcache_global_data_frames:
            plt.plot(
                sample_sizes[::45],
                vcache_global_error_rates[::45],
                "-",
                color="#8CBE94",
                linewidth=2,
                label=f"vCache (Ablation) (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::45],
                vcache_global_error_rates[::45],
                vcache_global_error_ci_lows[::45],
                vcache_global_error_ci_ups[::45],
                "#8CBE94",
            )

        # GPTCache
        if gptcache_data_frames and gptcache_df is not None:
            plt.plot(
                sample_sizes[::45],
                gptcache_error_rates[::45],
                "-",
                color="#C23B48",
                linewidth=2,
                label=f"GPTCache (t={gptcache_closest_threshold})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::45],
                gptcache_error_rates[::45],
                gptcache_error_ci_lows[::45],
                gptcache_error_ci_ups[::45],
                "#C23B48",
            )

        # Berkeley Embedding
        if berkeley_embedding_data_frames and berkeley_embedding_df is not None:
            plt.plot(
                sample_sizes[::45],
                berkeley_embedding_error_rates[::45],
                "-",
                color="#3B686A",
                linewidth=2,
                label=f"FT Emb (t={berkeley_embedding_closest_threshold})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::45],
                berkeley_embedding_error_rates[::45],
                berkeley_embedding_error_ci_lows[::45],
                berkeley_embedding_error_ci_ups[::45],
                "#3B686A",
            )

        # vCache + Berkeley Embedding
        if (
            vcache_berkeley_embedding_data_frames
            and target_delta in vcache_berkeley_embedding_data_frames
        ):
            plt.plot(
                sample_sizes[::45],
                vcache_berkeley_embedding_error_rates[::45],
                "-",
                color="#EDBE24",
                linewidth=2,
                label=f"vCache+FT (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::45],
                vcache_berkeley_embedding_error_rates[::45],
                vcache_berkeley_embedding_error_ci_lows[::45],
                vcache_berkeley_embedding_error_ci_ups[::45],
                "#EDBE24",
            )

        plt.xlabel("Sample Size", fontsize=font_size)
        plt.ylabel("Error Rate (%)", fontsize=font_size)
        plt.tick_params(axis="both", labelsize=font_size - 2)
        plt.legend(fontsize=font_size - 10, handlelength=0.5)

        error_rate_filename = (
            results_dir
            + f"/error_rate_vs_sample_size_delta_{target_delta:.3f}_{chopped_index}.pdf"
        )
        plt.savefig(
            error_rate_filename, format="pdf", bbox_inches="tight", transparent=True
        )
        plt.close()

        # Plot 2: Cache hit rates vs sample size
        plt.figure(figsize=(12, 11))

        # vCache Local
        plt.plot(
            sample_sizes[::5],
            vcache_local_cache_hit_rates[::5],
            "-",
            color="#37A9EC",
            linewidth=2,
            label=f"vCache (δ={target_delta})",
            markersize=1,
        )
        add_confidence_band(
            sample_sizes[::5],
            vcache_local_cache_hit_rates[::5],
            vcache_local_cache_hit_ci_lows[::5],
            vcache_local_cache_hit_ci_ups[::5],
            "#37A9EC",
        )

        # vCache Splitter
        if vcache_splitter_data_frames and target_delta in vcache_splitter_data_frames:
            plt.plot(
                sample_sizes[::5],
                vcache_splitter_cache_hit_rates[::5],
                "-",
                color="#9B59B6",
                linewidth=2,
                label=f"vCache Splitter (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::5],
                vcache_splitter_cache_hit_rates[::5],
                vcache_splitter_cache_hit_ci_lows[::5],
                vcache_splitter_cache_hit_ci_ups[::5],
                "#9B59B6",
            )

        # vCache Global
        if vcache_global_data_frames and target_delta in vcache_global_data_frames:
            plt.plot(
                sample_sizes[::5],
                vcache_global_cache_hit_rates[::5],
                "-",
                color="#8CBE94",
                linewidth=2,
                label=f"vCache (Ablation) (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::5],
                vcache_global_cache_hit_rates[::5],
                vcache_global_cache_hit_ci_lows[::5],
                vcache_global_cache_hit_ci_ups[::5],
                "#8CBE94",
            )

        # GPTCache
        if gptcache_data_frames and gptcache_df is not None:
            plt.plot(
                sample_sizes[::5],
                gptcache_cache_hit_rates[::5],
                "-",
                color="#C23B48",
                linewidth=2,
                label=f"GPTCache (t={gptcache_closest_threshold})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::5],
                gptcache_cache_hit_rates[::5],
                gptcache_cache_hit_ci_lows[::5],
                gptcache_cache_hit_ci_ups[::5],
                "#C23B48",
            )

        # Berkeley Embedding
        if berkeley_embedding_data_frames and berkeley_embedding_df is not None:
            plt.plot(
                sample_sizes[::5],
                berkeley_embedding_cache_hit_rates[::5],
                "-",
                color="#3B686A",
                linewidth=2,
                label=f"FT Embedding (t={berkeley_embedding_closest_threshold})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::5],
                berkeley_embedding_cache_hit_rates[::5],
                berkeley_embedding_cache_hit_ci_lows[::5],
                berkeley_embedding_cache_hit_ci_ups[::5],
                "#3B686A",
            )

        # vCache + Berkeley Embedding
        if (
            vcache_berkeley_embedding_data_frames
            and target_delta in vcache_berkeley_embedding_data_frames
        ):
            plt.plot(
                sample_sizes[::5],
                vcache_berkeley_embedding_cache_hit_rates[::5],
                "-",
                color="#EDBE24",
                linewidth=2,
                label=f"vCache + FT Embedding (δ={target_delta})",
                markersize=1,
            )
            add_confidence_band(
                sample_sizes[::5],
                vcache_berkeley_embedding_cache_hit_rates[::5],
                vcache_berkeley_embedding_cache_hit_ci_lows[::5],
                vcache_berkeley_embedding_cache_hit_ci_ups[::5],
                "#EDBE24",
            )

        plt.xlabel("Sample Size", fontsize=font_size)
        plt.ylabel("Cache Hit Rate (%)", fontsize=font_size)
        plt.tick_params(axis="both", labelsize=font_size - 2)
        plt.legend(fontsize=font_size - 10, handlelength=0.5)

        cache_hit_filename = (
            results_dir
            + f"/cache_hit_rate_vs_sample_size_delta_{target_delta:.3f}_{chopped_index}.pdf"
        )
        plt.savefig(
            cache_hit_filename, format="pdf", bbox_inches="tight", transparent=True
        )
        plt.close()


def __plot_delta_accuracy(
    vcache_local_data_frames: Dict[float, pd.DataFrame],
    vcache_splitter_data_frames: Dict[float, pd.DataFrame],
    vcache_global_data_frames: Dict[float, pd.DataFrame],
    vcache_berkeley_embedding_data_frames: Dict[float, pd.DataFrame],
    results_dir: str,
    timestamp: str,
    font_size: int,
    chopped_index: int,
    vcache_splitter_run_dirs_map_override: Dict[str, List[str]] | None = None,
):
    plt.figure(figsize=(12, 8))

    vcache_local_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
        results_dir, "vcache_local_"
    )

    if vcache_splitter_run_dirs_map_override is not None:
        vcache_splitter_run_dirs_map = vcache_splitter_run_dirs_map_override
    else:
        vcache_splitter_run_dirs_map = __collect_run_dirs_by_prefix_and_key(
            results_dir, "vcache_splitter_"
        )

    all_deltas_sorted = sorted(
        set(vcache_local_data_frames.keys()) | set(vcache_splitter_data_frames.keys())
    )

    if all_deltas_sorted:
        local_error_rate_means = []
        local_error_rate_cis_lower_err = []  # error from mean to lower bound
        local_error_rate_cis_upper_err = []  # error from mean to upper bound
        local_is_multi_run_list = []

        splitter_error_rate_means = []
        splitter_error_rate_cis_lower_err = []  # error from mean to lower bound
        splitter_error_rate_cis_upper_err = []  # error from mean to upper bound
        splitter_is_multi_run_list = []

        delta_labels = []

        for delta_val in all_deltas_sorted:
            delta_key_str = f"{delta_val:.3f}".rstrip("0").rstrip(".")
            if not delta_key_str and delta_val == 0:
                delta_key_str = "0"

            local_run_dirs = vcache_local_run_dirs_map.get(delta_key_str, [])
            if len(local_run_dirs) > 1:
                er_mean, er_ci_low, er_ci_up, _, _, _ = __aggregate_stats_for_latency_error(
                    local_run_dirs
                )
                local_error_rate_means.append(er_mean * 100)
                local_error_rate_cis_lower_err.append(max(0, (er_mean - er_ci_low) * 100))
                local_error_rate_cis_upper_err.append(max(0, (er_ci_up - er_mean) * 100))
                local_is_multi_run_list.append(True)
            else:
                if delta_val in vcache_local_data_frames:
                    df = vcache_local_data_frames[delta_val]
                    er_mean_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                    local_error_rate_means.append(er_mean_single)
                else:
                    local_error_rate_means.append(0)
                local_error_rate_cis_lower_err.append(0)
                local_error_rate_cis_upper_err.append(0)
                local_is_multi_run_list.append(False)

            splitter_run_dirs = vcache_splitter_run_dirs_map.get(delta_key_str, [])
            if len(splitter_run_dirs) > 1:
                er_mean, er_ci_low, er_ci_up, _, _, _ = __aggregate_stats_for_latency_error(
                    splitter_run_dirs
                )
                splitter_error_rate_means.append(er_mean * 100)
                splitter_error_rate_cis_lower_err.append(max(0, (er_mean - er_ci_low) * 100))
                splitter_error_rate_cis_upper_err.append(max(0, (er_ci_up - er_mean) * 100))
                splitter_is_multi_run_list.append(True)
            else:
                if delta_val in vcache_splitter_data_frames:
                    df = vcache_splitter_data_frames[delta_val]
                    er_mean_single = compute_error_rate_score(fp=df["fp_list"]) * 100
                    splitter_error_rate_means.append(er_mean_single)
                else:
                    splitter_error_rate_means.append(0)
                splitter_error_rate_cis_lower_err.append(0)
                splitter_error_rate_cis_upper_err.append(0)
                splitter_is_multi_run_list.append(False)

            delta_labels.append(
                f"{(delta_val * 100):.2f}"
            )  # Format to 2 decimal places

        x_pos = np.arange(len(all_deltas_sorted))
        bar_width = 0.35

        # Prepare yerr for plt.bar. It needs to be a 2xN array for asymmetric error bars.
        local_actual_errors_for_bar = [
            list(z)
            for z in zip(local_error_rate_cis_lower_err, local_error_rate_cis_upper_err)
        ]
        local_y_errors_for_plot = []
        for i in range(len(local_is_multi_run_list)):
            if local_is_multi_run_list[i]:
                local_y_errors_for_plot.append(local_actual_errors_for_bar[i])
            else:
                local_y_errors_for_plot.append([0, 0])
        local_y_errors_for_plot_transposed = np.array(local_y_errors_for_plot).T

        splitter_actual_errors_for_bar = [
            list(z)
            for z in zip(
                splitter_error_rate_cis_lower_err, splitter_error_rate_cis_upper_err
            )
        ]
        splitter_y_errors_for_plot = []
        for i in range(len(splitter_is_multi_run_list)):
            if splitter_is_multi_run_list[i]:
                splitter_y_errors_for_plot.append(splitter_actual_errors_for_bar[i])
            else:
                splitter_y_errors_for_plot.append([0, 0])
        splitter_y_errors_for_plot_transposed = np.array(splitter_y_errors_for_plot).T

        plt.bar(
            x_pos - bar_width / 2,
            local_error_rate_means,
            bar_width,
            color="#37A9EC",
            label="vCache Local Actual Error",
            yerr=local_y_errors_for_plot_transposed,
            capsize=10,
        )
        plt.bar(
            x_pos + bar_width / 2,
            splitter_error_rate_means,
            bar_width,
            color="#3B686A",
            label="vCache Splitter Actual Error",
            yerr=splitter_y_errors_for_plot_transposed,
            capsize=10,
        )

        for i, delta_target_val in enumerate(all_deltas_sorted):
            plt.hlines(
                y=delta_target_val * 100,  # Convert to percentage scale
                xmin=i - 0.45,
                xmax=i + 0.45,
                colors="#EDBE24",
                linestyles="dashed",
                linewidth=3,
            )

        custom_lines = [
            Line2D([0], [0], color="#EDBE24", linestyle="dashed", lw=4),
            Line2D([0], [0], color="#37A9EC", lw=4),
            Line2D([0], [0], color="#3B686A", lw=4),
        ]
        plt.legend(
            custom_lines,
            ["$\delta$ Target", "vCache Local Actual Error", "vCache Splitter Actual Error"],
            fontsize=font_size - 2,  # Adjusted for potentially more space needed
            handlelength=1.5,  # Adjusted handle length
            loc="upper left",  # Ensure legend doesn't overlap much
        )

        plt.xlabel("$\delta$ Values", fontsize=font_size)
        plt.xticks(rotation=45, ha="right")  # Rotate for better readability
        plt.ylabel("Error Rate (%)", fontsize=font_size)
        plt.xticks(x_pos, delta_labels, fontsize=font_size)
        plt.yticks(fontsize=font_size - 2)

        # Format y-ticks to show 2 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        plt.gca().spines["top"].set_linewidth(1)
        plt.gca().spines["right"].set_linewidth(1)
        plt.gca().spines["bottom"].set_linewidth(1)
        plt.gca().spines["left"].set_linewidth(1)

        all_plot_values = (
            local_error_rate_means
            + [
                er + ci_u
                for er, ci_u in zip(local_error_rate_means, local_error_rate_cis_upper_err)
            ]
            + splitter_error_rate_means
            + [
                er + ci_u
                for er, ci_u in zip(
                    splitter_error_rate_means, splitter_error_rate_cis_upper_err
                )
            ]
            + [delta * 100 for delta in all_deltas_sorted]
        )
        if all_plot_values:
            y_min = 0
            y_max = max(all_plot_values) * 1.15
            if y_max < 8.0:  # Adjusted for percentage scale
                y_max = 8.0  # Ensure a minimum sensible y_max
            plt.ylim(y_min, y_max)
        else:
            plt.ylim(0, 8.0)  # Default if no values, adjusted for percentage scale

    filename = results_dir + f"/delta_accuracy_{chopped_index}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.close()
