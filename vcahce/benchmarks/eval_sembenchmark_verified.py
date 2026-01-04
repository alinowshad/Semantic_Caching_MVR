"""
Offline evaluation on SemBenchmark datasets using VerifiedDecisionPolicy.

This script uses live BGE embeddings (computed via EmbeddingModel) to match the
model used by the Splitter approach, ensuring a fair comparison between the two.
"""

from __future__ import annotations
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)
import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmarks.common.comparison import answers_have_same_meaning_static
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.bge import (
    BGEEmbeddingEngine,
)
from vcache.vcache_core.splitter.embedding_model import EmbeddingModel
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
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
from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy


def _ensure_hf_cache_env() -> Dict[str, str]:
    """
    Set HF cache env vars if HF_CACHE_BASE is provided (or use /tmp).
    Returns resolved paths.
    """
    hf_cache_base = os.environ.get("HF_CACHE_BASE", "/tmp/hf")
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
    """
    Match `Benchmark.update_stats` semantics for (tp, fp, tn, fn).
    """
    if is_cache_hit:
        if label_id_set != -1:
            cache_response_correct = label_id_set == getattr(response_metadata, "id_set", -999999)
        elif use_llm_judge:
            # Not used in this script; kept for parity.
            cache_response_correct = label_response == cache_response
        else:
            cache_response_correct = answers_have_same_meaning_static(label_response, cache_response)

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
        "--device",
        default="cpu",
        help="Device for live BGE embedding calculation (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.002,
        help="Per-step sleep to allow VerifiedDecisionPolicy background updates (match benchmark.py).",
    )
    parser.add_argument(
        "--similarity-evaluator",
        choices=["benchmark_id_set", "string"],
        default="benchmark_id_set",
        help="How to evaluate correctness (match benchmark.py run-combination).",
    )
    parser.add_argument("--max-capacity", type=int, default=200_000)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save per-sample results for curve plotting.",
    )
    args = parser.parse_args()

    # Mirror: must be set before importing HF libs in a fresh process.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cache_paths = _ensure_hf_cache_env()

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

    # Build vCache with benchmark engines (offline eval)
    inference_engine = BenchmarkInferenceEngine()
    
    # Use live BGE embeddings to match the splitter's base model
    shared_embedder = EmbeddingModel(device=args.device)
    embedding_engine = BGEEmbeddingEngine(embedding_model=shared_embedder)

    if args.similarity_evaluator == "string":
        similarity_evaluator = StringComparisonSimilarityEvaluator()
    else:
        similarity_evaluator = BenchmarkComparisonSimilarityEvaluator()

    config = VCacheConfig(
        inference_engine=inference_engine,
        embedding_engine=embedding_engine,
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=args.max_capacity,
        ),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        eviction_policy=NoEvictionPolicy(),
        similarity_evaluator=similarity_evaluator,
    )

    policy = VerifiedDecisionPolicy(delta=args.delta)
    vcache = VCache(config=config, policy=policy)

    hits = 0
    tp = fp = tn = fn = 0
    n = 0
    per_sample_results = []

    t0 = time.time()
    pbar = tqdm(rows, desc="Evaluating", unit="samples")
    for r in pbar:
        prompt = r["prompt"]
        system_prompt = r.get("output_format", "") or ""
        id_set = _get_id_set(r)

        label_response = r[args.llm_col]

        # Inject ground truth response for the benchmark engine
        inference_engine.set_next_response(label_response)

        is_hit, resp, resp_meta, nn_meta = vcache.infer_with_cache_info(
            prompt=prompt,
            system_prompt=system_prompt,
            id_set=id_set,
        )

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

        # Track per-sample result
        per_sample_results.append({
            "sample_index": n,
            "is_hit": is_hit,
            "running_hit_rate": hits / n,
            "tp": d_tp,
            "fp": d_fp,
            "tn": d_tn,
            "fn": d_fn
        })

        # Update progress bar stats
        pbar.set_postfix({
            "hits": f"{hits}/{n}",
            "hit_rate": f"{(hits/n):.1%}"
        })

        # VerifiedDecisionPolicy updates cache metadata asynchronously; a tiny sleep (as in benchmark.py)
        # prevents the evaluation loop from outrunning the background worker and biasing toward EXPLORE.
        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    elapsed = time.time() - t0

    print(f"dataset={args.dataset}")
    print(f"columns: embedding={args.embedding_col} llm={args.llm_col}")
    print(f"delta={args.delta} n={n} time={elapsed:.2f}s")
    print(f"hit_rate={hits}/{n} ({(hits/max(1,n)):.1%})")
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")

    if args.output_json:
        # Ensure parent directory exists (e.g., "results/...")
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        output_data = {
            "args": vars(args),
            "summary": {
                "n": n,
                "hits": hits,
                "hit_rate": hits / max(1, n),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "total_time": elapsed
            },
            "per_sample": per_sample_results
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output_json}")

    # Clean shutdown (VerifiedDecisionPolicy uses background threads)
    time.sleep(0.1)
    vcache.vcache_policy.shutdown()


if __name__ == "__main__":
    main()


