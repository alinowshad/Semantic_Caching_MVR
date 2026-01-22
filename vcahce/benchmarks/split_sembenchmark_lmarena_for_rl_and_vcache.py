from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import Dataset, load_dataset


def _ensure_hf_cache_env(hf_cache_base: str | None = None) -> Dict[str, str]:
    hf_cache_base = hf_cache_base or os.environ.get("HF_CACHE_BASE", "/tmp/hf")
    hf_home = os.path.join(hf_cache_base, "home")
    hf_hub_cache = os.path.join(hf_cache_base, "hub")
    hf_datasets_cache = os.path.join(hf_cache_base, "datasets")
    hf_transformers_cache = os.path.join(hf_cache_base, "transformers")

    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(hf_hub_cache, exist_ok=True)
    os.makedirs(hf_datasets_cache, exist_ok=True)
    os.makedirs(hf_transformers_cache, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_hub_cache)
    os.environ.setdefault("DATASETS_CACHE", hf_datasets_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_transformers_cache)

    return {
        "HF_CACHE_BASE": hf_cache_base,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub_cache,
        "DATASETS_CACHE": hf_datasets_cache,
        "TRANSFORMERS_CACHE": hf_transformers_cache,
    }


def _auto_group_key(columns: Sequence[str]) -> str | None:

    if "id_set" in columns:
        return "id_set"

    if "ID_Set" in columns:
        return "ID_Set"

    if "label_id_set" in columns:
        return "label_id_set"

    if "group_id" in columns:
        return "group_id"

    return None


def _safe_str(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return None


def _canonical_pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _write_json_array(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def _export_csv(ds: Dataset, out_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ds.to_pandas().to_csv(out_path, index=False)


def _build_group_index(
    ds: Dataset, *, prompt_key: str, group_key: str
) -> Tuple[List[str], List[Any], Dict[Any, List[int]]]:
    prompts: List[str] = []
    groups: List[Any] = []
    by_group: Dict[Any, List[int]] = {}

    for idx, ex in enumerate(ds):
        p = _safe_str(ex.get(prompt_key))
        if p is None:
            prompts.append("")
            groups.append(None)
            continue
        g = ex.get(group_key)
        prompts.append(p)
        groups.append(g)
        by_group.setdefault(g, []).append(idx)

    return prompts, groups, by_group


def _sample_rl_indices(rng: random.Random, n: int, rl_size: int) -> List[int]:
    if rl_size > n:
        raise ValueError(f"rl_size={rl_size} > dataset size n={n}")
    return rng.sample(list(range(n)), rl_size)


def _pick_sentence_2(
    *,
    rng: random.Random,
    target_correct: int,
    s1_idx: int,
    prompts: List[str],
    groups: List[Any] | None,
    by_group: Dict[Any, List[int]] | None,
    candidate_indices: Sequence[int],
    used_pair_keys: set[Tuple[str, str]],
    max_tries: int = 200,
) -> Tuple[str, int]:
    s1 = prompts[s1_idx]
    if not s1:
        raise ValueError("Empty sentence_1")

    for _ in range(max_tries):
        if target_correct == 1 and groups is not None and by_group is not None:
            g = groups[s1_idx]
            cand = by_group.get(g, [])
            if len(cand) <= 1:
                continue
            s2_idx = rng.choice(cand)
            if s2_idx == s1_idx:
                continue
        elif target_correct == 0 and groups is not None and by_group is not None:
            g = groups[s1_idx]
            s2_idx = rng.choice(candidate_indices)
            if s2_idx == s1_idx:
                continue
            if groups[s2_idx] == g:
                continue
        else:
            s2_idx = rng.choice(candidate_indices)
            if s2_idx == s1_idx:
                continue

        s2 = prompts[s2_idx]
        if not s2:
            continue
        key = _canonical_pair_key(s1, s2)
        if key in used_pair_keys:
            continue
        used_pair_keys.add(key)
        return s2, target_correct

    raise RuntimeError("Failed to sample a unique pair after many tries")


def _targets_for_size(rng: random.Random, *, size: int, pos_fraction: float) -> List[int]:
    if abs(float(pos_fraction) - 0.5) > 1e-12:
        raise ValueError("pos_fraction must be exactly 0.5 to enforce 1:1 labels per split")
    if int(size) % 2 != 0:
        raise ValueError(f"split size must be even to enforce 1:1 labels per split. got size={int(size)}")
    half = int(size) // 2
    targets: List[int] = [1] * half + [0] * half
    rng.shuffle(targets)
    return targets


def _count_labels(pairs: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    pos = 0
    neg = 0
    for p in pairs:
        if int(p.get("correct", 0)) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def _build_s1_candidates(
    *,
    prompts: List[str],
    groups: List[Any] | None,
    by_group: Dict[Any, List[int]] | None,
    candidate_indices: Sequence[int],
) -> Tuple[List[int], List[int]]:
    non_empty = [i for i in candidate_indices if bool(prompts[i])]
    if groups is None or by_group is None:
        return non_empty, non_empty

    pos_s1: List[int] = []
    neg_s1: List[int] = []
    total = len(candidate_indices)
    for i in non_empty:
        g = groups[i]
        g_members = by_group.get(g, [])
        if len(g_members) >= 2:
            pos_s1.append(i)
        if total - len(g_members) >= 1:
            neg_s1.append(i)

    return pos_s1, neg_s1


def _build_pairs(
    *,
    rng: random.Random,
    targets: Sequence[int],
    prompts: List[str],
    groups: List[Any] | None,
    by_group: Dict[Any, List[int]] | None,
    candidate_indices: Sequence[int],
    s1_pos_indices: Sequence[int],
    s1_neg_indices: Sequence[int],
    used_pair_keys: set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for target_correct in targets:
        target_correct_int = int(target_correct)
        s1_pool = s1_pos_indices if target_correct_int == 1 else s1_neg_indices
        if not s1_pool:
            raise ValueError(
                "No eligible sentence_1 candidates available for correct="
                f"{target_correct_int}. Check group sizes / group_key or increase rl-size."
            )

        # If sampling fails due to uniqueness constraints or group constraints, retry
        # with a different sentence_1.
        last_err: Exception | None = None
        for _ in range(200):
            s1_idx = rng.choice(list(s1_pool))
            s1 = prompts[s1_idx]
            if not s1:
                continue
            try:
                s2, c = _pick_sentence_2(
                    rng=rng,
                    target_correct=target_correct_int,
                    s1_idx=s1_idx,
                    prompts=prompts,
                    groups=groups,
                    by_group=by_group,
                    candidate_indices=candidate_indices,
                    used_pair_keys=used_pair_keys,
                )
                pairs.append({"sentence_1": s1, "sentence_2": s2, "correct": int(c)})
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                continue

        if last_err is not None:
            raise last_err
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="vCache/SemBenchmarkLmArena",
        help="HF dataset id (default: vCache/SemBenchmarkLmArena)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to load (default: train)",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Which column to use as sentence text (default: prompt)",
    )
    parser.add_argument(
        "--group-key",
        default=None,
        help="Optional grouping column for positive/negative pair construction. If not set, auto-detect id_set/ID_Set/group_id.",
    )
    parser.add_argument(
        "--pos-fraction",
        type=float,
        default=0.5,
        help="Fraction of correct=1 pairs in RL dataset (default: 0.5)",
    )
    parser.add_argument(
        "--allow-random-labels",
        action="store_true",
        help="If no group key is available, allow random pairing (labels still follow pos-fraction but are not semantically grounded).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rl-size", type=int, default=4000)
    parser.add_argument("--rl-train-size", type=int, default=3600)
    parser.add_argument("--rl-val-size", type=int, default=200)
    parser.add_argument("--rl-test-size", type=int, default=200)
    parser.add_argument(
        "--out-rl-train-json",
        required=True,
        help="Output path for RL train JSON array (3800 by default).",
    )
    parser.add_argument(
        "--out-rl-val-json",
        required=True,
        help="Output path for RL val JSON array (200 by default).",
    )
    parser.add_argument(
        "--out-rl-test-json",
        required=True,
        help="Output path for RL test JSON array (200 by default).",
    )
    parser.add_argument(
        "--out-vcache-csv",
        required=True,
        help="Output CSV path for the remaining vCache test dataset (all other rows).",
    )
    parser.add_argument(
        "--hf-cache-base",
        default=os.environ.get("HF_CACHE_BASE", None),
        help="Base dir for HF caches (datasets/hub/transformers). Overrides HF_CACHE_BASE.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (will fail if data is not already cached locally).",
    )
    parser.add_argument(
        "--debug-print-all-group-counts",
        action="store_true",
        help="Print every group value and its row count (can be very large).",
    )
    parser.add_argument(
        "--debug-group-topk",
        type=int,
        default=20,
        help="How many largest groups to print in debug summary (default: 20).",
    )
    parser.add_argument(
        "--out-group-counts-csv",
        default=None,
        help="Optional CSV path to write per-group counts (columns: group_key, group_value, count).",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    total_rl = int(args.rl_train_size) + int(args.rl_val_size) + int(args.rl_test_size)
    if total_rl != int(args.rl_size):
        raise ValueError(
            f"rl-size mismatch: rl_size={int(args.rl_size)} but train+val+test={total_rl}"
        )

    if not (0.0 <= float(args.pos_fraction) <= 1.0):
        raise ValueError("--pos-fraction must be in [0, 1]")

    cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    ds = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=cache_paths["DATASETS_CACHE"],
        token=hf_token,
    )

    if not isinstance(ds, Dataset):
        raise ValueError("Expected a single Dataset split")

    cols = list(ds.column_names)
    if args.prompt_key not in cols:
        raise ValueError(f"prompt key not found: {args.prompt_key}. columns={cols}")

    group_key = args.group_key or _auto_group_key(cols)

    rng = random.Random(int(args.seed))
    n = int(ds.num_rows)

    rl_indices = _sample_rl_indices(rng, n, int(args.rl_size))
    rl_indices_set = set(rl_indices)
    vcache_indices = [i for i in range(n) if i not in rl_indices_set]

    ds_vcache = ds.select(vcache_indices)
    _export_csv(ds_vcache, args.out_vcache_csv)

    ds_rl_pool = ds.select(rl_indices)

    prompts_rl: List[str]
    groups_rl: List[Any] | None
    by_group_rl: Dict[Any, List[int]] | None

    if group_key is None:
        if not bool(args.allow_random_labels):
            raise ValueError(
                "No group key found (id_set/ID_Set/group_id). Pass --group-key explicitly or use --allow-random-labels."
            )
        prompts_rl = [_safe_str(ex.get(args.prompt_key)) or "" for ex in ds_rl_pool]
        groups_rl = None
        by_group_rl = None
    else:
        prompts_rl, groups_rl, by_group_rl = _build_group_index(
            ds_rl_pool, prompt_key=args.prompt_key, group_key=group_key
        )

        group_sizes = [len(v) for v in by_group_rl.values()]
        groups_total = len(group_sizes)
        groups_ge2 = sum(1 for s in group_sizes if s >= 2)
        pos_capacity = sum((s * (s - 1)) // 2 for s in group_sizes if s >= 2)
        required_pos = (int(args.rl_train_size) + int(args.rl_val_size) + int(args.rl_test_size)) // 2
        print(
            f"[INFO] group_key={group_key} groups={groups_total} groups>=2={groups_ge2} "
            f"pos_pair_capacity={pos_capacity} required_pos_pairs={required_pos}"
        )

        topk = max(int(args.debug_group_topk), 0)
        if topk > 0 or bool(args.debug_print_all_group_counts) or args.out_group_counts_csv:
            group_counts = {k: len(v) for k, v in by_group_rl.items()}

            if topk > 0:
                top_items = sorted(group_counts.items(), key=lambda kv: kv[1], reverse=True)[:topk]
                print(f"[INFO] Top-{topk} largest groups by {group_key}:")
                for gv, cnt in top_items:
                    print(f"  {group_key}={gv} count={cnt}")

                hist: Dict[int, int] = {}
                for cnt in group_counts.values():
                    hist[int(cnt)] = hist.get(int(cnt), 0) + 1
                for size in sorted(hist.keys())[:20]:
                    print(f"[INFO] group_size={size} -> num_groups={hist[size]}")

            if bool(args.debug_print_all_group_counts):
                print(f"[INFO] All groups by {group_key} (total={len(group_counts)}):")
                for gv, cnt in sorted(group_counts.items(), key=lambda kv: (kv[1], kv[0])):
                    print(f"  {group_key}={gv} count={cnt}")

            if args.out_group_counts_csv:
                out_dir = os.path.dirname(os.path.abspath(args.out_group_counts_csv))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                rows = [
                    {"group_key": str(group_key), "group_value": gv, "count": int(cnt)}
                    for gv, cnt in sorted(group_counts.items(), key=lambda kv: kv[1], reverse=True)
                ]
                pd.DataFrame(rows).to_csv(args.out_group_counts_csv, index=False)
                print(f"[OK] Wrote group counts CSV -> {os.path.abspath(args.out_group_counts_csv)}")

        if pos_capacity < required_pos:
            raise ValueError(
                "Not enough unique positive pairs available under the chosen group_key="
                f"{group_key}. Need at least {required_pos} unique positive pairs, "
                f"but capacity is {pos_capacity}. Try a different --group-key or reduce --rl-size."
            )

    candidate_indices = list(range(int(ds_rl_pool.num_rows)))
    s1_pos_indices, s1_neg_indices = _build_s1_candidates(
        prompts=prompts_rl,
        groups=groups_rl,
        by_group=by_group_rl,
        candidate_indices=candidate_indices,
    )

    used_pair_keys: set[Tuple[str, str]] = set()
    train_targets = _targets_for_size(
        rng, size=int(args.rl_train_size), pos_fraction=float(args.pos_fraction)
    )
    val_targets = _targets_for_size(
        rng, size=int(args.rl_val_size), pos_fraction=float(args.pos_fraction)
    )
    test_targets = _targets_for_size(
        rng, size=int(args.rl_test_size), pos_fraction=float(args.pos_fraction)
    )

    train_pairs = _build_pairs(
        rng=rng,
        targets=train_targets,
        prompts=prompts_rl,
        groups=groups_rl,
        by_group=by_group_rl,
        candidate_indices=candidate_indices,
        s1_pos_indices=s1_pos_indices,
        s1_neg_indices=s1_neg_indices,
        used_pair_keys=used_pair_keys,
    )
    val_pairs = _build_pairs(
        rng=rng,
        targets=val_targets,
        prompts=prompts_rl,
        groups=groups_rl,
        by_group=by_group_rl,
        candidate_indices=candidate_indices,
        s1_pos_indices=s1_pos_indices,
        s1_neg_indices=s1_neg_indices,
        used_pair_keys=used_pair_keys,
    )
    test_pairs = _build_pairs(
        rng=rng,
        targets=test_targets,
        prompts=prompts_rl,
        groups=groups_rl,
        by_group=by_group_rl,
        candidate_indices=candidate_indices,
        s1_pos_indices=s1_pos_indices,
        s1_neg_indices=s1_neg_indices,
        used_pair_keys=used_pair_keys,
    )

    _write_json_array(args.out_rl_train_json, train_pairs)
    _write_json_array(args.out_rl_val_json, val_pairs)
    _write_json_array(args.out_rl_test_json, test_pairs)

    train_pos, train_neg = _count_labels(train_pairs)
    val_pos, val_neg = _count_labels(val_pairs)
    test_pos, test_neg = _count_labels(test_pairs)

    print(f"[OK] Loaded dataset={args.dataset} split={args.split} rows={n}")
    print(f"[OK] RL pool rows={int(ds_rl_pool.num_rows)}")
    print(
        f"[OK] RL train: n={len(train_pairs)} pos={train_pos} neg={train_neg} -> {os.path.abspath(args.out_rl_train_json)}"
    )
    print(
        f"[OK] RL val:   n={len(val_pairs)} pos={val_pos} neg={val_neg} -> {os.path.abspath(args.out_rl_val_json)}"
    )
    print(
        f"[OK] RL test:  n={len(test_pairs)} pos={test_pos} neg={test_neg} -> {os.path.abspath(args.out_rl_test_json)}"
    )
    print("[OK] label_ratio per split: pos=neg (1:1)")
    if group_key is None:
        print("[WARN] No group key used; pairs are randomly labeled (see --allow-random-labels).")
    else:
        print(f"[OK] Using group_key={group_key} for correct labels")
    print(
        f"[OK] vCache test CSV: rows={int(ds_vcache.num_rows)} -> {os.path.abspath(args.out_vcache_csv)}"
    )


if __name__ == "__main__":
    main()
