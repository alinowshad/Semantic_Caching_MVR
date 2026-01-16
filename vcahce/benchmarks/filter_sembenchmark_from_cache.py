from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


BANNED_SOURCE_PATH_SUBSTRINGS = (
    "/data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_train.json",
    "/data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_val.json",
    "/data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_test.json",
)


DEFAULT_PAIRS_JSON_PATHS = BANNED_SOURCE_PATH_SUBSTRINGS


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


def _iter_string_leaves(x: Any) -> Iterable[str]:
    if x is None:
        return
    if isinstance(x, str):
        yield x
        return
    if isinstance(x, (int, float, bool)):
        return
    if isinstance(x, dict):
        for v in x.values():
            yield from _iter_string_leaves(v)
        return
    if isinstance(x, (list, tuple, set)):
        for v in x:
            yield from _iter_string_leaves(v)
        return


def _keep_example(
    example: Dict[str, Any],
    banned: tuple[str, ...],
    banned_sentences: set[str] | None = None,
    prompt_key: str = "prompt",
    match_substring: bool = False,
    text_extract: str = "after_question",
) -> bool:
    if banned_sentences:
        prompt = example.get(prompt_key)
        if isinstance(prompt, str):
            p = prompt.strip()
            extracted = _extract_prompt_text(p, text_extract)
            if match_substring:
                for s in banned_sentences:
                    if s and s in extracted:
                        return False
            else:
                if extracted in banned_sentences:
                    return False

    for v in example.values():
        for s in _iter_string_leaves(v):
            for b in banned:
                if b in s:
                    return False
    return True


def _load_banned_sentences_from_pairs_json(paths: list[str]) -> set[str]:
    banned: set[str] = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in pairs file: {p}")
        for row in data:
            if not isinstance(row, dict):
                continue
            s1 = row.get("sentence_1")
            s2 = row.get("sentence_2")
            if isinstance(s1, str) and s1.strip():
                banned.add(s1.strip())
            if isinstance(s2, str) and s2.strip():
                banned.add(s2.strip())
    return banned


def _prompt_hits_banned(prompt: Any, banned_sentences: set[str], match_substring: bool) -> bool:
    if not banned_sentences or not isinstance(prompt, str):
        return False
    p = prompt.strip()
    if not p:
        return False
    if match_substring:
        for s in banned_sentences:
            if s and s in p:
                return True
        return False
    return p in banned_sentences


def _extract_prompt_text(prompt: str, mode: str) -> str:
    p = prompt.strip()
    if mode == "none":
        return p
    if mode == "after_question":
        if "?" in p:
            return p.split("?", 1)[1].strip()
        return p
    raise ValueError(f"Unknown text_extract mode: {mode}")


def _first_substring_match(haystack: str, needles: set[str]) -> str | None:
    for s in needles:
        if s and s in haystack:
            return s
    return None


def _iter_extracted_prompts(
    ds: Dataset,
    *,
    prompt_key: str,
    text_extract: str,
) -> Iterable[str]:
    for ex in ds:
        p = ex.get(prompt_key)
        if not isinstance(p, str):
            continue
        yield _extract_prompt_text(p, text_extract)


def _debug_print_not_found_banned(
    *,
    ds: Dataset,
    banned_sentences: set[str],
    prompt_key: str,
    text_extract: str,
    match_substring: bool,
    print_n: int,
    max_len: int,
) -> None:
    if not banned_sentences:
        return

    if not match_substring:
        prompt_set: set[str] = set(_iter_extracted_prompts(ds, prompt_key=prompt_key, text_extract=text_extract))
        not_found = [s for s in banned_sentences if s not in prompt_set]
        found_n = len(banned_sentences) - len(not_found)
    else:
        remaining = set(banned_sentences)
        for extracted in _iter_extracted_prompts(ds, prompt_key=prompt_key, text_extract=text_extract):
            m = _first_substring_match(extracted, remaining)
            if m is not None:
                remaining.discard(m)
                if not remaining:
                    break
        not_found = list(remaining)
        found_n = len(banned_sentences) - len(not_found)

    mode = "substring" if match_substring else "exact"
    print(
        f"[DEBUG] Banned sentences coverage ({mode} over extracted prompts): found={found_n}/{len(banned_sentences)} not_found={len(not_found)}"
    )
    if int(print_n) <= 0:
        return

    if not not_found:
        return

    for i, s in enumerate(not_found[: int(print_n)], start=1):
        preview = s if len(s) <= int(max_len) else s[: int(max_len) - 3] + "..."
        print(f"  [NOT_FOUND {i}] len={len(s)} preview={preview}")
        print(f"             repr={repr(s[: int(max_len)])}")


@dataclass(frozen=True)
class MatchExample:
    prompt: str
    extracted_text: str
    matched_sentence: str


def _format_match_context(prompt_text: str, matched: str, max_len: int) -> str:
    idx = prompt_text.find(matched)
    if idx < 0:
        s = prompt_text
        return s if len(s) <= max_len else s[: max_len - 3] + "..."

    before = prompt_text[:idx]
    after = prompt_text[idx + len(matched) :]

    # show context around the match
    keep_each_side = max(0, (max_len - len(matched) - 10) // 2)
    b = before[-keep_each_side:]
    a = after[:keep_each_side]

    prefix = "..." if len(before) > len(b) else ""
    suffix = "..." if len(after) > len(a) else ""
    return f"{prefix}{b}[{matched}]{a}{suffix}"


def _export_csv(ds: Dataset, out_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = ds.to_pandas()
    df.to_csv(out_path, index=False)


def _export_jsonl(ds: Dataset, out_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="vCache/SemBenchmarkClassification",
        help="HF dataset id (default: vCache/SemBenchmarkClassification)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to load (default: train). Use 'all' to load all splits. You can also pass train[:1000] etc.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional: limit to first N samples using HF slicing syntax (e.g. train[:N]). Ignored when --split=all.",
    )
    parser.add_argument(
        "--pairs-json",
        action="append",
        default=None,
        help="Path to a local pairs JSON file (list of {sentence_1, sentence_2, ...}). Can be passed multiple times.",
    )
    parser.add_argument(
        "--use-default-pairs-json",
        action="store_true",
        help="Use default pairs JSON files (balanced_train/val/test) without specifying --pairs-json repeatedly.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Which HF dataset field to match against the banned sentence list (default: prompt).",
    )
    parser.add_argument(
        "--match-substring",
        action="store_true",
        help="If set, remove rows where banned sentence is a substring of prompt (default: exact match after strip).",
    )
    parser.add_argument(
        "--text-extract",
        choices=["none", "after_question"],
        default="after_question",
        help="How to extract the text body from the prompt before matching (default: after_question).",
    )
    parser.add_argument(
        "--debug-print-removed",
        type=int,
        default=5,
        help="Print up to N examples of removed prompts with matched sentence evidence.",
    )
    parser.add_argument(
        "--debug-max-len",
        type=int,
        default=240,
        help="Max length of printed prompt/context in debug output.",
    )
    parser.add_argument(
        "--debug-print-not-found",
        type=int,
        default=0,
        help="Print up to N banned sentences that are NOT found in the loaded dataset prompts (after text extraction).",
    )
    parser.add_argument(
        "--debug-count-not-found",
        action="store_true",
        help="Only print the count of banned sentences NOT found in the loaded dataset prompts (no examples).",
    )
    parser.add_argument(
        "--hf-cache-base",
        default=os.environ.get("HF_CACHE_BASE", None),
        help="Base dir for HF caches (datasets/hub/transformers). Overrides HF_CACHE_BASE.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (will fail if data/model are not already cached locally).",
    )
    parser.add_argument(
        "--out-dir",
        required=False,
        default=None,
        help="Output directory for save_to_disk() filtered dataset.",
    )
    parser.add_argument(
        "--no-save-to-disk",
        action="store_true",
        help="Do not write datasets.save_to_disk() (arrow) output. Useful when disk space is limited and you only need CSV/JSONL exports.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV export path for the filtered dataset (for benchmark.py custom dataset loading).",
    )
    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional JSONL export path for the filtered dataset.",
    )
    args = parser.parse_args()

    if bool(args.no_save_to_disk) and not (args.out_csv or args.out_jsonl):
        raise ValueError("--no-save-to-disk requires --out-csv and/or --out-jsonl (otherwise nothing would be saved).")
    if (not bool(args.no_save_to_disk)) and not args.out_dir:
        raise ValueError("--out-dir is required unless you pass --no-save-to-disk.")

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cache_paths = _ensure_hf_cache_env(args.hf_cache_base)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    banned_sentences: set[str] = set()
    pairs_json_paths: list[str] = []
    if args.pairs_json:
        pairs_json_paths.extend(list(args.pairs_json))
    if args.use_default_pairs_json:
        pairs_json_paths.extend(list(DEFAULT_PAIRS_JSON_PATHS))

    if pairs_json_paths:
        banned_sentences = _load_banned_sentences_from_pairs_json(pairs_json_paths)
        print(f"[OK] Loaded banned sentences from pairs-json: n={len(banned_sentences)}")

    split_arg = None
    if args.split != "all":
        split_arg = args.split
        if args.max_samples is not None and "[" not in split_arg:
            # Mirror eval_sembenchmark_verified_splitter.py behavior: use HF split slicing.
            split_arg = f"{split_arg}[:{int(args.max_samples)}]"

    ds = load_dataset(
        args.dataset,
        split=split_arg,
        cache_dir=cache_paths["DATASETS_CACHE"],
        token=hf_token,
    )

    # Optional: show which banned sentences never appear in the loaded prompts.
    if banned_sentences and (bool(args.debug_count_not_found) or int(args.debug_print_not_found) > 0):
        if isinstance(ds, DatasetDict):
            # Check coverage across all splits combined.
            combined_not_found: set[str] = set(banned_sentences)
            for split_name, split_ds in ds.items():
                remaining = set(combined_not_found)
                for extracted in _iter_extracted_prompts(
                    split_ds,
                    prompt_key=args.prompt_key,
                    text_extract=args.text_extract,
                ):
                    if not remaining:
                        break
                    if bool(args.match_substring):
                        m = _first_substring_match(extracted, remaining)
                        if m is not None:
                            remaining.discard(m)
                    else:
                        if extracted in remaining:
                            remaining.discard(extracted)
                combined_not_found = remaining

            mode = "substring" if args.match_substring else "exact"
            found_n = len(banned_sentences) - len(combined_not_found)
            print(
                f"[DEBUG] Banned sentences coverage across all splits ({mode}): found={found_n}/{len(banned_sentences)} not_found={len(combined_not_found)}"
            )
            if int(args.debug_print_not_found) > 0:
                for i, s in enumerate(list(combined_not_found)[: int(args.debug_print_not_found)], start=1):
                    preview = s if len(s) <= int(args.debug_max_len) else s[: int(args.debug_max_len) - 3] + "..."
                    print(f"  [NOT_FOUND {i}] len={len(s)} preview={preview}")
                    print(f"             repr={repr(s[: int(args.debug_max_len)])}")
        else:
            _debug_print_not_found_banned(
                ds=ds,
                banned_sentences=banned_sentences,
                prompt_key=args.prompt_key,
                text_extract=args.text_extract,
                match_substring=bool(args.match_substring),
                print_n=int(args.debug_print_not_found),
                max_len=int(args.debug_max_len),
            )

    def _build_debug_examples(split_ds: Dataset) -> list[MatchExample]:
        examples: list[MatchExample] = []
        if not banned_sentences or int(args.debug_print_removed) <= 0:
            return examples
        for ex in split_ds:
            prompt = ex.get(args.prompt_key)
            if not isinstance(prompt, str):
                continue
            extracted = _extract_prompt_text(prompt, args.text_extract)

            matched: str | None = None
            if bool(args.match_substring):
                matched = _first_substring_match(extracted, banned_sentences)
            else:
                if extracted in banned_sentences:
                    matched = extracted

            if matched is None:
                continue
            examples.append(MatchExample(prompt=prompt, extracted_text=extracted, matched_sentence=matched))
            if len(examples) >= int(args.debug_print_removed):
                break
        return examples

    if isinstance(ds, DatasetDict):
        n0 = {k: int(v.num_rows) for k, v in ds.items()}

        debug_by_split: dict[str, list[MatchExample]] = {}
        if banned_sentences and int(args.debug_print_removed) > 0:
            for split_name, split_ds in ds.items():
                debug_by_split[split_name] = _build_debug_examples(split_ds)

        if banned_sentences:
            match_counts = {}
            for split_name, split_ds in ds.items():
                c = 0
                for ex in split_ds:
                    p = ex.get(args.prompt_key)
                    if not isinstance(p, str):
                        continue
                    extracted = _extract_prompt_text(p, args.text_extract)
                    if bool(args.match_substring):
                        if _first_substring_match(extracted, banned_sentences) is not None:
                            c += 1
                    else:
                        if extracted in banned_sentences:
                            c += 1
                match_counts[split_name] = c
            mode = "substring" if args.match_substring else "exact"
            print(f"[OK] Prompt {mode}-match counts by split (before filtering): {match_counts}")

        ds_filtered = ds.filter(
            _keep_example,
            fn_kwargs={
                "banned": BANNED_SOURCE_PATH_SUBSTRINGS,
                "banned_sentences": banned_sentences,
                "prompt_key": args.prompt_key,
                "match_substring": bool(args.match_substring),
                "text_extract": args.text_extract,
            },
        )
        n1 = {k: int(v.num_rows) for k, v in ds_filtered.items()}
        removed = {k: n0[k] - n1.get(k, 0) for k in n0.keys()}
        print(f"[OK] Loaded: {args.dataset} split=all rows={n0}")
        print(f"[OK] Filtered rows={n1} (removed {removed})")

        if banned_sentences and int(args.debug_print_removed) > 0:
            for split_name, examples in debug_by_split.items():
                if not examples:
                    continue
                print(f"\n[DEBUG] Removed prompt examples (split={split_name}, showing up to {int(args.debug_print_removed)}):")
                for i, ex in enumerate(examples, start=1):
                    ctx = _format_match_context(ex.extracted_text, ex.matched_sentence, int(args.debug_max_len))
                    p_preview = ex.prompt if len(ex.prompt) <= int(args.debug_max_len) else ex.prompt[: int(args.debug_max_len) - 3] + "..."
                    print(f"  Example {i}:")
                    print(f"    matched_sentence: {ex.matched_sentence[: int(args.debug_max_len)]}")
                    print(f"    extracted_text_ctx: {ctx}")
                    print(f"    original_prompt: {p_preview}")
    else:
        n0 = int(getattr(ds, "num_rows", 0))

        debug_examples = _build_debug_examples(ds) if isinstance(ds, Dataset) else []

        if banned_sentences:
            c = 0
            for ex in ds:
                p = ex.get(args.prompt_key)
                if not isinstance(p, str):
                    continue
                extracted = _extract_prompt_text(p, args.text_extract)
                if bool(args.match_substring):
                    if _first_substring_match(extracted, banned_sentences) is not None:
                        c += 1
                else:
                    if extracted in banned_sentences:
                        c += 1
            mode = "substring" if args.match_substring else "exact"
            print(f"[OK] Prompt {mode}-match count (before filtering): {c}")

        ds_filtered = ds.filter(
            _keep_example,
            fn_kwargs={
                "banned": BANNED_SOURCE_PATH_SUBSTRINGS,
                "banned_sentences": banned_sentences,
                "prompt_key": args.prompt_key,
                "match_substring": bool(args.match_substring),
                "text_extract": args.text_extract,
            },
        )

        n1 = int(getattr(ds_filtered, "num_rows", 0))
        removed = n0 - n1
        print(f"[OK] Loaded: {args.dataset} split={args.split} rows={n0}")
        print(f"[OK] Filtered rows={n1} (removed {removed})")

        if banned_sentences and debug_examples:
            print(f"\n[DEBUG] Removed prompt examples (showing up to {int(args.debug_print_removed)}):")
            for i, ex in enumerate(debug_examples, start=1):
                ctx = _format_match_context(ex.extracted_text, ex.matched_sentence, int(args.debug_max_len))
                p_preview = ex.prompt if len(ex.prompt) <= int(args.debug_max_len) else ex.prompt[: int(args.debug_max_len) - 3] + "..."
                print(f"  Example {i}:")
                print(f"    matched_sentence: {ex.matched_sentence[: int(args.debug_max_len)]}")
                print(f"    extracted_text_ctx: {ctx}")
                print(f"    original_prompt: {p_preview}")

    if not bool(args.no_save_to_disk):
        out_dir = os.path.abspath(args.out_dir)
        if os.path.exists(out_dir) and os.listdir(out_dir):
            raise FileExistsError(f"Output directory is not empty: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        ds_filtered.save_to_disk(out_dir)
        print(f"[OK] Saved filtered dataset -> {out_dir}")

    if args.out_csv:
        out_csv_path = os.path.abspath(args.out_csv)
        if isinstance(ds_filtered, DatasetDict):
            base_dir = os.path.dirname(out_csv_path)
            base_name = os.path.splitext(os.path.basename(out_csv_path))[0]
            for split_name, split_ds in ds_filtered.items():
                out_path = os.path.join(base_dir, f"{base_name}_{split_name}.csv")
                _export_csv(split_ds, out_path)
                print(f"[OK] Saved CSV ({split_name}) -> {out_path}")
        else:
            _export_csv(ds_filtered, out_csv_path)
            print(f"[OK] Saved CSV -> {out_csv_path}")

    if args.out_jsonl:
        if isinstance(ds_filtered, DatasetDict):
            base_dir = os.path.dirname(os.path.abspath(args.out_jsonl))
            base_name = os.path.splitext(os.path.basename(args.out_jsonl))[0]
            for split_name, split_ds in ds_filtered.items():
                out_path = os.path.join(base_dir, f"{base_name}_{split_name}.jsonl")
                _export_jsonl(split_ds, out_path)
                print(f"[OK] Saved JSONL ({split_name}) -> {out_path}")
        else:
            _export_jsonl(ds_filtered, args.out_jsonl)
            print(f"[OK] Saved JSONL -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
