import importlib
import os
import re
import sys

import torch
from tensordict.tensordict import TensorDict

from .AdaptedPointerNetworkPolicy import AdaptedPointerNetworkPolicy
from .embedding_model import EmbeddingModel
from .MaxSimEnv import MaxSimEnv, get_segments_from_token_pointers
from .MaxSimGenerator import MaxSimGenerator

from rl4co.envs.common.base import RL4COEnvBase

def _register_checkpoint_module_aliases() -> None:
    """
    Backward-compat for old checkpoints saved with top-level module names.

    Example failure:
      ModuleNotFoundError: No module named 'MaxSimEnv'
    """

    aliases = {
        "MaxSimEnv": "vcache.vcache_core.splitter.MaxSimEnv",
        "MaxSimGenerator": "vcache.vcache_core.splitter.MaxSimGenerator",
        "MaxSimSplitter": "vcache.vcache_core.splitter.MaxSimSplitter",
        "embedding_model": "vcache.vcache_core.splitter.embedding_model",
        "AdaptedPointerNetworkPolicy": "vcache.vcache_core.splitter.AdaptedPointerNetworkPolicy",
    }

    for old_name, new_name in aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except Exception:
            # Best-effort; only needed if a checkpoint references this name.
            continue


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve `checkpoint_path` into an actual checkpoint file on disk.

    Supported inputs:
    - A checkpoint file path (returned as-is if exists)
    - A directory path (auto-pick newest `epoch=*-step=*.ckpt`, else `last.ckpt`)
    - A non-existent file path whose parent is a directory (falls back to newest in parent)
    """
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError(f"checkpoint_path must be a non-empty str, got: {checkpoint_path!r}")

    p = os.path.abspath(os.path.expanduser(checkpoint_path))

    def _pick_from_dir(d: str) -> str:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Checkpoint directory not found: {d}")

        try:
            entries = os.listdir(d)
        except OSError as e:
            raise FileNotFoundError(f"Could not list checkpoint directory: {d} ({e})") from e

        epoch_pat = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
        epoch_files: list[tuple[int, int, str]] = []
        for name in entries:
            m = epoch_pat.match(name)
            if not m:
                continue
            epoch_files.append((int(m.group(1)), int(m.group(2)), os.path.join(d, name)))

        if epoch_files:
            epoch_files.sort(key=lambda t: (t[0], t[1]))
            return epoch_files[-1][2]

        last_ckpt = os.path.join(d, "last.ckpt")
        if os.path.isfile(last_ckpt):
            return last_ckpt

        preview = ", ".join(sorted(entries)[:20])
        raise FileNotFoundError(
            f"No checkpoint found in directory: {d}. "
            f"Expected `epoch=*-step=*.ckpt` or `last.ckpt`. "
            f"Found (first 20): {preview}"
        )

    if os.path.isdir(p):
        return _pick_from_dir(p)

    if os.path.isfile(p):
        return p

    parent = os.path.dirname(p) or "."
    if os.path.isdir(parent):
        # Helpful fallback when a specific epoch file was referenced but rotated away.
        return _pick_from_dir(parent)

    raise FileNotFoundError(f"Checkpoint path not found: {p}")
class MaxSimSplitter:
    def __init__(self, checkpoint_path, device="cuda", embedding_model=None):
        # Normalize device early so all downstream `.to(...)` calls are consistent.
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        resolved_ckpt = _resolve_checkpoint_path(checkpoint_path)
        print(f"正在加载分句模型: {resolved_ckpt} ...")
        
        # 1. 初始化组件
        if embedding_model is not None:
            print("MaxSimSplitter: 复用外部传入的 EmbeddingModel")
            self.embedding_model = embedding_model
        else:
            print("MaxSimSplitter: 未传入 EmbeddingModel，正在创建新实例...")
            self.embedding_model = EmbeddingModel() 

        # Ensure the underlying LM is on the requested device *before* constructing the env.
        # RL4CO env init may call the generator once (prints [GEN]) and we want it to reflect the real device.
        if hasattr(self.embedding_model, "model") and hasattr(self.embedding_model.model, "to"):
            self.embedding_model.model.to(self.device)
        
        # Dummy Generator/Env
        # NOTE: MaxSimEnv init may call the generator once (which prints `[GEN] ... device=...`).
        # Move the LM to the target device *before* constructing the env so that call uses CUDA.
        self.generator = MaxSimGenerator(
            prompts=["dummy"], max_len=512, embedding_model=self.embedding_model
        )
        if hasattr(self.generator.lm, "to"):
            self.generator.lm.to(self.device)
        # Ensure the RL4CO env itself is created on the intended device; otherwise it may move
        # the shared LM back to CPU during init.
        self.env = MaxSimEnv(
            generator=self.generator,
            max_segments=4,
            embedding_model=self.embedding_model,
            device=self.device,
        )

        # 2. 初始化 Policy
        self.policy = AdaptedPointerNetworkPolicy(self.env, embedding_dim=768, hidden_dim=128, max_segments=4)
        
        # =================================================================
   
        original_setstate = RL4COEnvBase.__setstate__

      
        def safe_setstate(obj, state):
            try:
                original_setstate(obj, state)
            except (TypeError, RuntimeError) as e:
              
                pass

     
        RL4COEnvBase.__setstate__ = safe_setstate

        try:
            _register_checkpoint_module_aliases()
            checkpoint = torch.load(resolved_ckpt, map_location='cpu', weights_only=False)
        finally:
          
            RL4COEnvBase.__setstate__ = original_setstate
        # =================================================================

        # 4. 提取并加载权重
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            new_state_dict = {k.replace("policy.", ""): v for k, v in state_dict.items() if k.startswith("policy.")}
            self.policy.load_state_dict(new_state_dict)
        else:
            print("警告: Checkpoint 中没有 state_dict")
        
        # 5. 最后再移动到目标设备
        self.policy.to(self.device)
        self.policy.eval()
        print("分句模型加载完成。")
    def split_pair_return_segments(self, text_a, text_b):
        """
        输入：Query 和 Cache Candidate
        输出：RL模型优化后的 A片段列表 和 B片段列表
        """
        # 1. 构造输入 (Joint Input)
        inputs_a = self.generator.tokenizer([text_a], return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(self.device)
        inputs_b = self.generator.tokenizer([text_b], return_tensors='pt', padding='max_length', truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            embeds_a = self.generator.lm(**inputs_a).last_hidden_state
            embeds_b = self.generator.lm(**inputs_b).last_hidden_state
        
        td = TensorDict({
            "token_embeddings_a": embeds_a,
            "token_embeddings_b": embeds_b,
            "attention_mask_a": inputs_a['attention_mask'],
            "attention_mask_b": inputs_b['attention_mask'],
            "length_a": inputs_a['attention_mask'].sum(dim=1),
            "length_b": inputs_b['attention_mask'].sum(dim=1),
            "input_ids_a": inputs_a['input_ids'],
            "input_ids_b": inputs_b['input_ids'],
        }, batch_size=1)

        # 2. Greedy Decoding 获取最佳切分动作
        with torch.no_grad():
            out = self.policy(td, None, phase="test", select_best=True, decode_type="greedy")
        
        actions = out['actions'][0] # [2 * max_segments]

        # 3. 解析动作 -> 文本片段
        # NOTE: In this repo, actions are interleaved: [A0, B0, A1, B1, ...]
        # See `inspect_punctuation_cases.py` and `MaxSimEnv._step` (full-plan interleaved layout).
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0: 2 * max_segments: 2].tolist()
        pointers_b = actions[1: 2 * max_segments: 2].tolist()
        
        # Reconstruct segments in **token-index space** (pointers are token positions).
        # This avoids the previous mismatch where pointers (token indices) were applied
        # to `prompt.lower().split()` (word indices).
        segments_a = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=inputs_a["input_ids"][0],
            attention_mask=inputs_a.get("attention_mask", None)[0]
            if inputs_a.get("attention_mask", None) is not None
            else None,
            pointers=pointers_a,
        )
        segments_b = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=inputs_b["input_ids"][0],
            attention_mask=inputs_b.get("attention_mask", None)[0]
            if inputs_b.get("attention_mask", None) is not None
            else None,
            pointers=pointers_b,
        )
        
        return segments_a, segments_b

    def debug_split_pair(self, text_a: str, text_b: str) -> dict:
        """
        Debug helper to inspect the splitter output.

        Returns a dict containing:
        - pointers_a / pointers_b: raw token-index pointers (inclusive end boundaries)
        - segments_a / segments_b: decoded segments reconstructed from token pointers
        - length_a / length_b: effective token lengths (sum of attention_mask)
        """
        inputs_a = self.generator.tokenizer(
            [text_a],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)
        inputs_b = self.generator.tokenizer(
            [text_b],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            embeds_a = self.generator.lm(**inputs_a).last_hidden_state
            embeds_b = self.generator.lm(**inputs_b).last_hidden_state

        td = TensorDict(
            {
                "token_embeddings_a": embeds_a,
                "token_embeddings_b": embeds_b,
                "attention_mask_a": inputs_a["attention_mask"],
                "attention_mask_b": inputs_b["attention_mask"],
                "length_a": inputs_a["attention_mask"].sum(dim=1),
                "length_b": inputs_b["attention_mask"].sum(dim=1),
                "input_ids_a": inputs_a["input_ids"],
                "input_ids_b": inputs_b["input_ids"],
            },
            batch_size=1,
        )

        with torch.no_grad():
            out = self.policy(
                td, None, phase="test", select_best=True, decode_type="greedy", debug=True
            )

        actions = out["actions"][0]
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device)
        total = int(actions.numel())
        if total % 2 != 0:
            raise ValueError(f"Expected even number of action entries (A/B interleaved), got {total}")
        max_segments = total // 2
        pointers_a = actions[0 : 2 * max_segments : 2].tolist()
        pointers_b = actions[1 : 2 * max_segments : 2].tolist()

        segments_a = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=inputs_a["input_ids"][0],
            attention_mask=inputs_a["attention_mask"][0],
            pointers=pointers_a,
        )
        segments_b = get_segments_from_token_pointers(
            tokenizer=self.generator.tokenizer,
            input_ids=inputs_b["input_ids"][0],
            attention_mask=inputs_b["attention_mask"][0],
            pointers=pointers_b,
        )

        return {
            "pointers_a": pointers_a,
            "pointers_b": pointers_b,
            "segments_a": segments_a,
            "segments_b": segments_b,
            "length_a": int(inputs_a["attention_mask"][0].sum().item()),
            "length_b": int(inputs_b["attention_mask"][0].sum().item()),
            "policy_info": out.get("info", {}),
        }