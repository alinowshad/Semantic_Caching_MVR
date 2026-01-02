import argparse
import random

import torch
from tensordict.tensordict import TensorDict

from embedding_model import EmbeddingModel
from MaxSimGenerator import MaxSimGenerator
from MaxSimEnv import MaxSimEnv
from AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy


def _load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Inspect whether punctuation-based action masking provides valid choices, and what the policy samples."
    )
    parser.add_argument("--prompts_file", type=str, required=True, help="txt file: one prompt per line")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional Lightning checkpoint (.ckpt). If provided, loads trained policy weights from it.",
    )
    parser.add_argument(
        "--unsafe_ckpt_load",
        action="store_true",
        help=(
            "If set, loads the checkpoint with torch.load(weights_only=False). "
            "This can execute arbitrary code during unpickling; use ONLY for trusted checkpoints."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--max_segments", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true", help="Enable per-step debug printing from the policy")
    parser.add_argument("--debug_topk", type=int, default=8, help="Top-k logits to log per step (per side)")
    parser.add_argument("--debug_n_samples", type=int, default=1, help="How many samples in the batch to log")
    parser.add_argument(
        "--debug_log_path",
        type=str,
        default=None,
        help="If set, appends JSONL debug records here (one dict per line).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    prompts = _load_lines(args.prompts_file)

    embedding_model = EmbeddingModel(device=device)
    gen = MaxSimGenerator(prompts=prompts, max_len=args.max_len, embedding_model=embedding_model)
    env = MaxSimEnv(generator=gen, max_segments=args.max_segments, embedding_model=embedding_model, device=device)
    policy_hidden_dim = 128
    policy = AdaptedPointerNetworkPolicy(
        env,
        embedding_dim=policy_hidden_dim,
        hidden_dim=policy_hidden_dim,
        max_segments=args.max_segments,
    ).to(device)

    # Optionally load trained weights
    if args.ckpt_path:
        # PyTorch 2.6+ defaults torch.load(weights_only=True) which uses a restricted unpickler.
        # Lightning .ckpt files may include references to project classes (e.g., MaxSimEnv),
        # so we allowlist them for safe weights-only loading.
        try:
            if args.unsafe_ckpt_load:
                ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
            else:
                try:
                    ckpt = torch.load(args.ckpt_path, map_location="cpu")  # weights_only=True by default
                except Exception:
                    # Retry with safe globals allowlisted.
                    # Lightning checkpoints may include references to project classes AND TorchRL spec objects.
                    try:
                        import inspect
                        from torch.serialization import safe_globals
                        import torchrl.data.tensor_specs as tensor_specs

                        torchrl_spec_classes = [
                            obj
                            for obj in vars(tensor_specs).values()
                            if inspect.isclass(obj) and getattr(obj, "__module__", "") == tensor_specs.__name__
                        ]

                        allow = [
                            # project classes
                            MaxSimEnv,
                            MaxSimGenerator,
                            EmbeddingModel,
                            AdaptedPointerNetworkPolicy,
                            # torchrl tensor spec classes (Composite/Bounded/Unbounded/etc.)
                            *torchrl_spec_classes,
                        ]

                        with safe_globals(allow):
                            ckpt = torch.load(args.ckpt_path, map_location="cpu")
                    except Exception as e2:
                        raise e2
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint safely: {args.ckpt_path}\n"
                f"- If this is your own checkpoint and you trust it, rerun with --unsafe_ckpt_load\n"
                f"- Otherwise, consider exporting just the state_dict.\n"
                f"Original error: {e}"
            ) from e

        state = ckpt.get("state_dict", ckpt)
        # Lightning checkpoints store keys like "policy.encoder_layers.0...."
        policy_state = {}
        for k, v in state.items():
            if k.startswith("policy."):
                policy_state[k[len("policy.") :]] = v
        missing, unexpected = policy.load_state_dict(policy_state, strict=False)
        print(f"[CKPT] Loaded policy weights from {args.ckpt_path}")
        print(f"[CKPT] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

    # Sample one batch from generator (returns cpu tensordict)
    td: TensorDict = gen(batch_size=args.batch_size)
    # Move to device for policy/env
    td = td.to(device)

    with torch.no_grad():
        out = policy(
            td,
            env,
            phase="train",
            decode_type="sampling",
            debug=args.debug,
            debug_topk=args.debug_topk,
            debug_n_samples=args.debug_n_samples,
            debug_log_path=args.debug_log_path,
            debug_print=True,
        )

    actions = out["actions"]  # [batch, 2*max_segments] interleaved [A0,B0,A1,B1,...]
    rewards = out["reward"]
    info = out.get("info", {})

    tok = embedding_model.tokenizer
    ids_a = td["input_ids_a"]
    ids_b = td["input_ids_b"]
    la = td["length_a"].long()
    lb = td["length_b"].long()

    # Recompute valid punctuation positions at step 0 (current boundary = 0)
    # This matches policy masking logic (punctuation/[SEP]/[EOS] only).
    valid_ids = getattr(policy, "_valid_split_ids", None)
    if valid_ids is None:
        policy._init_punctuation_ids(ids_a.device)
        valid_ids = policy._valid_split_ids

    range_a = torch.arange(ids_a.size(1), device=device).unsqueeze(0).expand(args.batch_size, -1)
    range_b = torch.arange(ids_b.size(1), device=device).unsqueeze(0).expand(args.batch_size, -1)
    is_punct_a = torch.isin(ids_a, valid_ids)
    is_punct_b = torch.isin(ids_b, valid_ids)

    # Match policy's "effective length" behavior: exclude trailing [SEP]/[EOS] from the selectable range.
    sep_id = getattr(tok, "sep_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    last_idx_a = (la - 1).clamp(min=0)
    last_idx_b = (lb - 1).clamp(min=0)
    last_tok_a = ids_a.gather(1, last_idx_a.unsqueeze(1)).squeeze(1)
    last_tok_b = ids_b.gather(1, last_idx_b.unsqueeze(1)).squeeze(1)
    is_last_special_a = torch.zeros_like(last_tok_a, dtype=torch.bool)
    is_last_special_b = torch.zeros_like(last_tok_b, dtype=torch.bool)
    if sep_id is not None:
        is_last_special_a |= last_tok_a == sep_id
        is_last_special_b |= last_tok_b == sep_id
    if eos_id is not None:
        is_last_special_a |= last_tok_a == eos_id
        is_last_special_b |= last_tok_b == eos_id
    eff_len_a = (la - is_last_special_a.long()).clamp(min=2)
    eff_len_b = (lb - is_last_special_b.long()).clamp(min=2)

    # Match policy behavior: disallow selecting the final token position (eff_len-1) as an explicit split.
    valid0_a = is_punct_a & (range_a > 0) & (range_a < (eff_len_a - 1).unsqueeze(1))
    valid0_b = is_punct_b & (range_b > 0) & (range_b < (eff_len_b - 1).unsqueeze(1))

    print("\n=== Batch summary ===")
    print("policy info keys:", sorted(list(info.keys()))[:20], "..." if len(info.keys()) > 20 else "")
    for k in ["avg_valid_a", "avg_valid_b", "frac_any_valid_a", "frac_any_valid_b", "frac_zero_log_likelihood"]:
        if k in info:
            v = info[k]
            try:
                v = float(v)
            except Exception:
                pass
            print(f"{k}: {v}")

    print("\n=== Per-sample inspection (step 0 candidates) ===")
    for i in range(args.batch_size):
        a_candidates = torch.where(valid0_a[i])[0].tolist()
        b_candidates = torch.where(valid0_b[i])[0].tolist()
        a0 = int(actions[i, 0].item())
        b0 = int(actions[i, 1].item())

        # decode candidate tokens (small cap for printing)
        def _tok_at(seq, idx):
            try:
                return tok.convert_ids_to_tokens(int(seq[idx].item()))
            except Exception:
                return str(int(seq[idx].item()))

        print(f"\n-- sample {i} reward={float(rewards[i].item()):.6f}")
        print(f"A: len={int(la[i].item())} n_candidates(step0)={len(a_candidates)} chosen={a0} token={_tok_at(ids_a[i], a0)}")
        if a_candidates:
            show = a_candidates[:12]
            print("  A candidates (first 12):", [(j, _tok_at(ids_a[i], j)) for j in show])
        else:
            print("  A candidates: NONE -> policy will often fallback to length-1 at many steps")

        print(f"B: len={int(lb[i].item())} n_candidates(step0)={len(b_candidates)} chosen={b0} token={_tok_at(ids_b[i], b0)}")
        if b_candidates:
            show = b_candidates[:12]
            print("  B candidates (first 12):", [(j, _tok_at(ids_b[i], j)) for j in show])
        else:
            print("  B candidates: NONE -> policy will often fallback to length-1 at many steps")


if __name__ == "__main__":
    main()


