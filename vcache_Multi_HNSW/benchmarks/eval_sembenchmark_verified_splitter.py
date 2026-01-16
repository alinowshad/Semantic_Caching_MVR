"""
Offline evaluation on SemBenchmark datasets using VerifiedSplitterDecisionPolicy (no live model calls).

This mirrors `benchmarks/eval_sembenchmark_verified.py`, but the cache decision is made via:
  (query_prompt, cached_prompt) -> MaxSimSplitter -> MaxSim similarity score
and then the verified Bayesian explore/exploit logic.

Notes:
- The vector DB candidate selection uses the same BGE embedding model as the MaxSim splitter (via BGEEmbeddingEngine).
- The MaxSim splitter requires a checkpoint; pass it via `--splitter-checkpoint`.

Usage (example):
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_TOKEN=hf_...                  # optional but recommended to avoid rate limits
  python benchmarks/eval_sembenchmark_verified_splitter.py \
    --dataset vCache/SemBenchmarkClassification \
    --embedding-col emb_gte \
    --llm-col response_llama_3_8b \
    --splitter-checkpoint /path/to/ckpt_or_dir \
    --delta 0.02
poetry run python benchmarks/eval_sembenchmark_verified_splitter.py \
  --dataset /home/zhengzishan/Semantic_Caching_MVR/vcahce/datasets/filtered_sembenchmark_train.csv \
  --llm-col response_llama_3_8b \
  --deltas 0.01 0.015 0.02 0.03 0.05 0.07 0.08 \
  --candidate-selection multivector_top_k \
  --candidate-k 10 \
  --splitter-checkpoint ~/checkpoints_words/epoch=29-step=1620.ckpt \
  --splitter-device cuda:3 \
  --similarity-evaluator string \
  --sleep 0.1 \
  --output-json results/local_verified_splitter.json \
  --benchmark-output-dir results/benchmark_compat \
  --benchmark-run-index 1
  env:

zhengzishan@user-Super-Server:~/Semantic_Caching_MVR/vcache_Multi_HNSW$ poetry run python -m pip uninstall -y chroma-hnswlib
Found existing installation: chroma-hnswlib 0.7.6
Uninstalling chroma-hnswlib-0.7.6:
  Successfully uninstalled chroma-hnswlib-0.7.6
zhengzishan@user-Super-Server:~/Semantic_Caching_MVR/vcache_Multi_HNSW$ poetry run python -m pip install -e vcache/vcache_core/cache/embedding_store/hnswlib
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.config import VCacheConfig
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.bge import (
    BGEEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import NoEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.splitter.MaxSimSplitter import MaxSimSplitter
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.vcache_policy.strategies.verified_splitter import VerifiedSplitterDecisionPolicy


# Ignore scikit-learn 1.8+ warning (LogisticRegression(penalty=...) deprecation) coming from verified policy internals.
warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)

def _ensure_hf_cache_env(hf_cache_base: str | None = None) -> Dict[str, str]:
    """
    Set HF cache env vars if HF_CACHE_BASE is provided (or use /tmp).
    Returns resolved paths.
    """
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


def _get_id_set(row: Dict[str, Any]) -> int:
    v = row.get("id_set", -1)
    if v == -1:
        v = row.get("ID_Set", -1)
    try:
        return int(v)
    except Exception:
        return -1


def _score_step(
    *,
    is_cache_hit: bool,
    label_id_set: int,
    label_response: str,
    cache_response: str,
    response_metadata,
    nn_metadata,
    use_llm_judge: bool = False,
) -> Tuple[int, int, int, int]:
    if is_cache_hit:
        if label_id_set != -1:
            cache_response_correct = label_id_set == getattr(
                response_metadata, "id_set", -999999
            )
        elif use_llm_judge:
            cache_response_correct = label_response == cache_response
        else:
            cache_response_correct = answers_have_same_meaning_static(
                label_response, cache_response
            )

        if cache_response_correct:
            return 1, 0, 0, 0
        return 0, 1, 0, 0

    # cache miss
    if label_id_set != -1:
        nn_response_correct = label_id_set == getattr(nn_metadata, "id_set", -999999)
    elif use_llm_judge:
        nn_response_correct = label_response == getattr(nn_metadata, "response", "")
    else:
        nn_response_correct = answers_have_same_meaning_static(
            label_response, getattr(nn_metadata, "response", "")
        )

    if nn_response_correct:
        return 0, 0, 0, 1  # FN
    return 0, 0, 1, 0  # TN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="HF dataset id, e.g. vCache/SemBenchmarkClassification",
    )
    parser.add_argument(
        "--embedding-col",
        help="Embedding column (optional, no longer used for embeddings as BGE is live).",
    )
    parser.add_argument(
        "--llm-col", required=True, help="LLM response column, e.g. response_llama_3_8b"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate. If None, use all.",
    )
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of deltas to run (overrides --delta). Example: --deltas 0.015 0.02 0.03",
    )
    parser.add_argument(
        "--similarity-evaluator",
        choices=["benchmark_id_set", "string"],
        default="benchmark_id_set",
        help="How to evaluate correctness (match benchmark.py run-combination).",
    )
    parser.add_argument("--max-capacity", type=int, default=200_000)
    parser.add_argument(
        "--splitter-checkpoint",
        required=True,
        help="MaxSimSplitter checkpoint file or directory (auto-picks latest ckpt).",
    )
    parser.add_argument(
        "--splitter-device",
        default="cpu",
        help="Device for MaxSimSplitter (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--candidate-selection",
        choices=["top_k", "all", "multivector_top_k"],
        default="top_k",
        help="How to pick cached prompts to score with MaxSim (fast: top_k; slow: all).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="When candidate-selection=top_k, how many vector-DB neighbors to rerank. "
        "When candidate-selection=multivector_top_k, how many neighbors to retrieve PER query vector "
        "(then union parent IDs).",
    )
    parser.add_argument(
        "--candidate-ks",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of candidate-k values to run (overrides --candidate-k). Example: --candidate-ks 5 10 15",
    )
    parser.add_argument(
        "--use-cached-candidate-segments",
        action="store_true",
        help="Cache candidate MaxSim segment tensors in metadata and only segment the query at request time.",
    )
    parser.add_argument(
        "--multivector-max-elements",
        type=int,
        default=2_000_000,
        help="Capacity of the multivector HNSW index in *vectors* (not docs). Only used when candidate-selection=multivector_top_k.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.002,
        help="Optional per-step sleep to allow background updates (match benchmark.py).",
    )
    parser.add_argument(
        "--hf-cache-base",
        default=os.environ.get("HF_CACHE_BASE", None),
        help="Base dir for HuggingFace caches (datasets/hub/transformers). Overrides HF_CACHE_BASE.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save per-sample results for curve plotting.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=str,
        default=None,
        help="If set, write benchmark.py-compatible outputs under this directory (creates vcache_splitter_*_run_* folders with results_<timestamp>.json).",
    )
    parser.add_argument(
        "--benchmark-run-index",
        type=int,
        default=1,
        help="run index used in vcache_splitter_*_run_<idx> folder naming when --benchmark-output-dir is set.",
    )
    parser.add_argument(
        "--benchmark-timestamp",
        type=str,
        default=None,
        help="Timestamp string used in results_<timestamp>.json when --benchmark-output-dir is set. Default matches benchmark.py format: YYYY-MM-DD_HH-MM",
    )
    args = parser.parse_args()

    # Mirror: must be set before importing HF libs in a fresh process.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    dataset_is_local_file = False
    try:
        dataset_is_local_file = os.path.exists(str(args.dataset))
    except Exception:
        dataset_is_local_file = False
    if str(args.dataset).endswith(".csv") or str(args.dataset).endswith(".parquet"):
        dataset_is_local_file = True

    if dataset_is_local_file:
        dataset_path = os.path.abspath(str(args.dataset))
        if dataset_path.endswith(".csv"):
            try:
                df = pd.read_csv(dataset_path)
            except Exception:
                df = pd.read_csv(
                    dataset_path,
                    engine="python",
                    on_bad_lines="skip",
                    low_memory=False,
                )
        elif dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError(
                f"Unsupported local dataset file format: {dataset_path} (expected .csv or .parquet)"
            )

        if "prompt" not in df.columns:
            raise ValueError(
                f"Local dataset is missing required column 'prompt'. Available columns: {list(df.columns)}"
            )
        if args.llm_col not in df.columns:
            raise ValueError(
                f"Local dataset is missing required LLM response column '{args.llm_col}'. Available columns: {list(df.columns)}"
            )

        id_col = None
        if "ID_Set" in df.columns:
            id_col = "ID_Set"
        elif "id_set" in df.columns:
            id_col = "id_set"

        if args.similarity_evaluator == "benchmark_id_set":
            has_usable_ids = False
            if id_col is not None:
                try:
                    s = pd.to_numeric(df[id_col], errors="coerce").fillna(-1)
                    has_usable_ids = bool((s.astype(int) != -1).any())
                except Exception:
                    has_usable_ids = False
            if not has_usable_ids:
                args.similarity_evaluator = "string"

        rows = df.to_dict("records")
        if args.max_samples is not None:
            rows = rows[: int(args.max_samples)]
    else:
        cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        # Load dataset
        split = "train"
        if args.max_samples is not None:
            split = f"train[:{args.max_samples}]"

        rows = load_dataset(
            args.dataset,
            split=split,
            cache_dir=cache_paths["DATASETS_CACHE"],
            token=hf_token,
        )

    deltas = (
        list(args.deltas)
        if args.deltas is not None and len(args.deltas) > 0
        else [float(args.delta)]
    )
    if str(args.candidate_selection) == "all":
        candidate_ks = [-1]
    else:
        candidate_ks = (
            list(args.candidate_ks)
            if args.candidate_ks is not None and len(args.candidate_ks) > 0
            else [int(args.candidate_k)]
        )
    run_grid = [(float(d), int(k)) for d in deltas for k in candidate_ks]

    shared_embedder = EmbeddingModel(device=args.splitter_device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)
    splitter = MaxSimSplitter(
        checkpoint_path=args.splitter_checkpoint,
        device=args.splitter_device,
        embedding_model=shared_embedder,
    )

    if args.similarity_evaluator == "string":
        similarity_evaluator = StringComparisonSimilarityEvaluator()
    else:
        similarity_evaluator = BenchmarkComparisonSimilarityEvaluator()

    def _resolve_output_path(*, delta: float, candidate_k: int) -> str | None:
        if not args.output_json:
            return None
        base = str(args.output_json)
        if len(run_grid) <= 1:
            return base

        base_abs = os.path.abspath(base)
        if base_abs.endswith(os.sep) or os.path.isdir(base_abs):
            os.makedirs(base_abs, exist_ok=True)
            name = f"verified_splitter_delta{delta}_k{candidate_k}.json"
            return os.path.join(base_abs, name)

        root, ext = os.path.splitext(base)
        ext = ext if ext else ".json"
        return f"{root}_delta{delta}_k{candidate_k}{ext}"

    benchmark_timestamp = args.benchmark_timestamp or datetime.now().strftime(
        "%Y-%m-%d_%H-%M"
    )

    def _resolve_benchmark_output_dir(*, delta: float, candidate_k: int) -> str | None:
        if not args.benchmark_output_dir:
            return None

        base_dir = os.path.abspath(str(args.benchmark_output_dir))
        os.makedirs(base_dir, exist_ok=True)

        candidate_k_label = "all" if int(candidate_k) == -1 else str(int(candidate_k))
        sel_for_dir = "all" if str(args.candidate_selection) == "all" else "top_k"
        dir_name = f"vcache_splitter_{delta}_{sel_for_dir}_{candidate_k_label}_run_{int(args.benchmark_run_index)}"
        out_dir = os.path.join(base_dir, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _compute_statistics_json(
        *,
        cache_hit_list: list[int],
        cache_miss_list: list[int],
        tp_list: list[int],
        fp_list: list[int],
        tn_list: list[int],
        fn_list: list[int],
        latency_direct_list: list[float],
        latency_vectorq_list: list[float],
    ) -> Dict[str, Any]:
        n = int(len(latency_vectorq_list))
        avg_latency_vcache_overall = float(sum(latency_vectorq_list) / n) if n > 0 else 0.0
        avg_latency_direct_overall = float(sum(latency_direct_list) / n) if n > 0 else 0.0

        hit_latencies_v = [latency_vectorq_list[i] for i in range(n) if int(cache_hit_list[i]) > 0]
        miss_latencies_v = [latency_vectorq_list[i] for i in range(n) if int(cache_miss_list[i]) > 0]
        hit_latencies_d = [latency_direct_list[i] for i in range(n) if int(cache_hit_list[i]) > 0]
        miss_latencies_d = [latency_direct_list[i] for i in range(n) if int(cache_miss_list[i]) > 0]

        avg_latency_vcache_cache_hit = (
            float(sum(hit_latencies_v) / len(hit_latencies_v)) if hit_latencies_v else 0.0
        )
        avg_latency_vcache_cache_miss = (
            float(sum(miss_latencies_v) / len(miss_latencies_v)) if miss_latencies_v else 0.0
        )
        avg_latency_direct_cache_hit = (
            float(sum(hit_latencies_d) / len(hit_latencies_d)) if hit_latencies_d else 0.0
        )
        avg_latency_direct_cache_miss = (
            float(sum(miss_latencies_d) / len(miss_latencies_d)) if miss_latencies_d else 0.0
        )

        cache_hit_rate_vcache = float(sum(cache_hit_list) / n) if n > 0 else 0.0
        cache_miss_rate_vcache = 1.0 - cache_hit_rate_vcache
        error_rate_vcache = float(sum(fp_list) / n) if n > 0 else 0.0

        duration_vcache = float(sum(latency_vectorq_list))
        duration_direct = float(sum(latency_direct_list))

        tp_sum = int(sum(tp_list))
        fp_sum = int(sum(fp_list))
        tn_sum = int(sum(tn_list))
        fn_sum = int(sum(fn_list))

        accuracy_vcache = float((tp_sum + tn_sum) / n) if n > 0 else 0.0
        precision_vcache = (
            float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        )
        recall_vcache = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
        f1_score_vcache = (
            float(2 * precision_vcache * recall_vcache / (precision_vcache + recall_vcache))
            if (precision_vcache + recall_vcache) > 0
            else 0.0
        )

        hits_last = int(cache_hit_list[-1]) if cache_hit_list else 0
        misses_last = int(cache_miss_list[-1]) if cache_miss_list else 0

        return {
            "avg_latency": {
                "cache": {
                    "overall": float(avg_latency_vcache_overall),
                    "cache_hit": float(avg_latency_vcache_cache_hit),
                    "cache_miss": float(avg_latency_vcache_cache_miss),
                },
                "direct": {"overall": float(avg_latency_direct_overall)},
                "difference": {
                    "overall": float(avg_latency_direct_overall - avg_latency_vcache_overall),
                    "cache_hit": float(avg_latency_direct_cache_hit - avg_latency_vcache_cache_hit),
                    "cache_miss": float(avg_latency_direct_cache_miss - avg_latency_vcache_cache_miss),
                },
                "ratio": {
                    "overall": float(avg_latency_direct_overall / avg_latency_vcache_overall)
                    if avg_latency_vcache_overall > 0
                    else "N/A",
                    "cache_hit": float(avg_latency_direct_cache_hit / avg_latency_vcache_cache_hit)
                    if avg_latency_vcache_cache_hit > 0
                    else "N/A",
                    "cache_miss": float(avg_latency_direct_cache_miss / avg_latency_vcache_cache_miss)
                    if avg_latency_vcache_cache_miss > 0
                    else "N/A",
                },
            },
            "cache": {
                "hit_rate": float(cache_hit_rate_vcache),
                "miss_rate": float(cache_miss_rate_vcache),
                "total_samples": int(n),
                "hits": int(hits_last),
                "misses": int(misses_last),
                "error_rate": float(error_rate_vcache),
            },
            "duration": {
                "vectorq": float(duration_vcache),
                "direct": float(duration_direct),
            },
            "statistics": {
                "accuracy": float(accuracy_vcache),
                "precision": float(precision_vcache),
                "recall": float(recall_vcache),
                "f1_score": float(f1_score_vcache),
            },
        }

    all_summaries: list[dict] = []

    for run_i, (delta, candidate_k) in enumerate(run_grid, start=1):
        inference_engine = BenchmarkInferenceEngine()
        config = VCacheConfig(
            embedding_engine=embedding_engine,
            inference_engine=inference_engine,
            vector_db=HNSWLibVectorDB(
                similarity_metric_type=SimilarityMetricType.COSINE,
                max_capacity=int(args.max_capacity),
            ),
            embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
            eviction_policy=NoEvictionPolicy(),
            similarity_evaluator=similarity_evaluator,
        )

        policy = VerifiedSplitterDecisionPolicy(
            delta=float(delta),
            splitter=splitter,
            device=args.splitter_device,
            candidate_selection=args.candidate_selection,
            candidate_k=int(candidate_k),
            use_cached_candidate_segments=bool(args.use_cached_candidate_segments),
            multivector_max_elements=int(args.multivector_max_elements),
        )
        vcache = VCache(config=config, policy=policy)

        hits = 0
        tp = fp = tn = fn = 0
        n = 0
        per_sample_results: list[dict] = []

        cache_hit_list: list[int] = []
        cache_miss_list: list[int] = []
        tp_list: list[int] = []
        fp_list: list[int] = []
        tn_list: list[int] = []
        fn_list: list[int] = []
        latency_direct_list: list[float] = []
        latency_vectorq_list: list[float] = []

        t0 = time.time()
        pbar = tqdm(
            rows,
            desc=f"Evaluating (Splitter) run={run_i}/{len(run_grid)} delta={delta} k={candidate_k}",
            unit="samples",
        )
        for r in pbar:
            prompt = r["prompt"]
            system_prompt = r.get("output_format", "") or ""
            id_set = _get_id_set(r)

            label_response = r[args.llm_col]
            inference_engine.set_next_response(label_response)

            step_t0 = time.time()
            is_hit, resp, resp_meta, nn_meta = vcache.infer_with_cache_info(
                prompt=prompt,
                system_prompt=system_prompt,
                id_set=id_set,
            )
            step_latency = time.time() - step_t0

            n += 1
            hits += int(is_hit)
            d_tp, d_fp, d_tn, d_fn = _score_step(
                is_cache_hit=is_hit,
                label_id_set=id_set,
                label_response=label_response,
                cache_response=resp,
                response_metadata=resp_meta,
                nn_metadata=nn_meta,
            )
            tp += d_tp
            fp += d_fp
            tn += d_tn
            fn += d_fn

            cache_hit_list.append(int(bool(is_hit)))
            cache_miss_list.append(int(not bool(is_hit)))
            tp_list.append(int(d_tp))
            fp_list.append(int(d_fp))
            tn_list.append(int(d_tn))
            fn_list.append(int(d_fn))
            latency_direct_list.append(float(step_latency))
            latency_vectorq_list.append(float(step_latency))

            per_sample_results.append(
                {
                    "sample_index": n,
                    "is_hit": is_hit,
                    "running_hit_rate": hits / n,
                    "tp": d_tp,
                    "fp": d_fp,
                    "tn": d_tn,
                    "fn": d_fn,
                }
            )

            pbar.set_postfix({"hits": f"{hits}/{n}", "hit_rate": f"{(hits/n):.1%}"})

            if args.sleep and args.sleep > 0:
                time.sleep(float(args.sleep))

        elapsed = time.time() - t0

        print(f"dataset={args.dataset}")
        print(f"columns: embedding={args.embedding_col} llm={args.llm_col}")
        print(f"delta={delta} n={n} time={elapsed:.2f}s")
        print(f"candidate_selection={args.candidate_selection} candidate_k={candidate_k}")
        print(f"hit_rate={hits}/{n} ({(hits/max(1,n)):.1%})")
        print(f"tp={tp} fp={fp} tn={tn} fn={fn}")

        summary = {
            "delta": float(delta),
            "candidate_k": int(candidate_k),
            "candidate_selection": str(args.candidate_selection),
            "n": n,
            "hits": hits,
            "hit_rate": hits / max(1, n),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "total_time": elapsed,
        }
        all_summaries.append(summary)

        out_path = _resolve_output_path(delta=float(delta), candidate_k=int(candidate_k))
        if out_path:
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            output_data = {"args": vars(args), "summary": summary, "per_sample": per_sample_results}
            with open(out_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {out_path}")

        bench_dir = _resolve_benchmark_output_dir(delta=float(delta), candidate_k=int(candidate_k))
        if bench_dir:
            observations_dict: Dict[str, Dict[str, float]] = {}
            gammas_dict: Dict[str, float] = {}
            t_hats_dict: Dict[str, float] = {}
            t_primes_dict: Dict[str, float] = {}
            var_ts_dict: Dict[str, float] = {}

            try:
                metadata_objects: List[EmbeddingMetadataObj] = (
                    vcache.vcache_config.embedding_metadata_storage.get_all_embedding_metadata_objects()
                )
            except Exception:
                metadata_objects = []

            for metadata_object in metadata_objects:
                try:
                    embedding_id = str(metadata_object.embedding_id)
                except Exception:
                    continue
                observations_dict[embedding_id] = getattr(metadata_object, "observations", {})
                gammas_dict[embedding_id] = getattr(metadata_object, "gamma", None)
                t_hats_dict[embedding_id] = getattr(metadata_object, "t_hat", None)
                t_primes_dict[embedding_id] = getattr(metadata_object, "t_prime", None)
                var_ts_dict[embedding_id] = getattr(metadata_object, "var_t", None)

            try:
                global_observations_dict = vcache.vcache_policy.global_observations
                global_gamma = vcache.vcache_policy.bayesian.global_gamma
                global_t_hat = vcache.vcache_policy.bayesian.global_t_hat
                global_t_prime = vcache.vcache_policy.bayesian.global_t_prime
                global_var_t = vcache.vcache_policy.bayesian.global_var_t
            except Exception:
                global_observations_dict = {}
                global_gamma = None
                global_t_hat = None
                global_t_prime = None
                global_var_t = None

            candidate_k_label = "all" if int(candidate_k) == -1 else str(int(candidate_k))
            bench_data: Dict[str, Any] = {
                "config": {
                    "filepath": str(args.dataset),
                    "embedding_model": str(args.embedding_col or ""),
                    "llm_model": str(args.llm_col),
                    "eviction_policy": str(NoEvictionPolicy()),
                    "is_static_threshold": False,
                    "threshold": None,
                    "delta": float(delta),
                    "splitter_candidate_selection": str(args.candidate_selection),
                    "splitter_candidate_k": (None if int(candidate_k) == -1 else int(candidate_k)),
                    "splitter_use_cached_candidate_segments": bool(args.use_cached_candidate_segments),
                    "splitter_device": str(args.splitter_device),
                    "splitter_checkpoint": str(args.splitter_checkpoint),
                },
                "cache_hit_list": cache_hit_list,
                "cache_miss_list": cache_miss_list,
                "tp_list": tp_list,
                "fp_list": fp_list,
                "tn_list": tn_list,
                "fn_list": fn_list,
                "latency_direct_list": latency_direct_list,
                "latency_vectorq_list": latency_vectorq_list,
                "observations_dict": observations_dict,
                "gammas_dict": gammas_dict,
                "t_hats_dict": t_hats_dict,
                "t_primes_dict": t_primes_dict,
                "var_ts_dict": var_ts_dict,
                "global_observations_dict": global_observations_dict,
                "global_gamma": global_gamma,
                "global_t_hat": global_t_hat,
                "global_t_prime": global_t_prime,
                "global_var_t": global_var_t,
            }

            bench_path = os.path.join(bench_dir, f"results_{benchmark_timestamp}.json")
            with open(bench_path, "w") as f:
                json.dump(bench_data, f, indent=4)
            print(f"Benchmark-format results saved to {bench_path}")

            statistics_path = os.path.join(
                bench_dir, f"statistics_{benchmark_timestamp}.json"
            )
            statistics_data = _compute_statistics_json(
                cache_hit_list=cache_hit_list,
                cache_miss_list=cache_miss_list,
                tp_list=tp_list,
                fp_list=fp_list,
                tn_list=tn_list,
                fn_list=fn_list,
                latency_direct_list=latency_direct_list,
                latency_vectorq_list=latency_vectorq_list,
            )
            with open(statistics_path, "w") as f:
                json.dump(statistics_data, f, indent=4)
            print(f"Benchmark-format statistics saved to {statistics_path}")

        time.sleep(0.1)
        vcache.vcache_policy.shutdown()

    if len(all_summaries) > 1:
        print("\nAll runs summary:")
        for s in all_summaries:
            print(
                f"delta={s['delta']} k={s['candidate_k']} hit_rate={s['hits']}/{s['n']} ({s['hit_rate']:.1%}) "
                f"tp={s['tp']} fp={s['fp']} tn={s['tn']} fn={s['fn']} time={s['total_time']:.2f}s"
            )


if __name__ == "__main__":
    main()


