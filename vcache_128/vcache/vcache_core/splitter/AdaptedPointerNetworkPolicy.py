import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .embedding_model import EmbeddingModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.mha_ab = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.mha_ba = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.norm_ab = nn.LayerNorm(hidden_dim)
        self.norm_ba = nn.LayerNorm(hidden_dim)
        self.ff_ab = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ff_ba = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        pad_mask_a: torch.Tensor | None = None,
        pad_mask_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A attends to B
        attn_a, _ = self.mha_ab(query=emb_a, key=emb_b, value=emb_b, key_padding_mask=pad_mask_b)
        a_out = self.norm_ab(emb_a + attn_a)
        a_out = self.norm_ab(a_out + self.ff_ab(a_out))

        # B attends to A
        attn_b, _ = self.mha_ba(query=emb_b, key=emb_a, value=emb_a, key_padding_mask=pad_mask_a)
        b_out = self.norm_ba(emb_b + attn_b)
        b_out = self.norm_ba(b_out + self.ff_ba(b_out))

        return a_out, b_out

class AdaptedPointerNetworkPolicy(nn.Module):
    """
    用于双文本分割的指针网络策略。

    该策略使用交叉注意力Transformer对两个输入文本进行编码，
    并使用一个自回归的LSTM解码器来依次预测两个文本的分割边界。
    """
    def __init__(self,   
                 env, 
                 embedding_dim=128,    
                 hidden_dim=128,        
                 max_segments=6,       
                 nhead=4,               
                 num_encoder_layers=2  
                ):
        super().__init__()
        self.env = env
        self.hidden_dim = hidden_dim
        self.max_segments = max_segments
      
        self.train_decode_type = "sampling"
        self.val_decode_type = "greedy"
        self.test_decode_type = "greedy"
        

        # --- 编码器 (Encoder) ---
    
        self.encoder_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim=hidden_dim, nhead=nhead)
            for _ in range(num_encoder_layers)
        ])

        # --- 解码器 (Decoder) ---
      
        self.decoder_cell = nn.LSTMCell(hidden_dim, hidden_dim) 
        

        self.attention_linear_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.attention_linear_encoder = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        
       
        self.decoder_start_input = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        if hasattr(env, 'device'):
            self._device = env.device
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
       
        if hasattr(env, 'reward_lm'):
          
            self.embedding_model = env.reward_lm
            self.tokenizer = env.reward_lm.tokenizer
            self.lm = env.reward_lm.model
           
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        elif hasattr(env, 'generator') and hasattr(env.generator, 'embedding_model'):
      
            self.embedding_model = env.generator.embedding_model
            self.tokenizer = env.generator.embedding_model.tokenizer
            self.lm = env.generator.embedding_model.model
            self.lm.to(self._device)
            self._device = next(self.lm.parameters()).device
        else:
            
            self.embedding_model = EmbeddingModel()
            self.embedding_model.model.to(self._device)
            for param in self.embedding_model.model.parameters():
                param.requires_grad = False
            self.embedding_model.model.eval()
            self.tokenizer = self.embedding_model.tokenizer
            self.lm = self.embedding_model.model
            self._device = next(self.lm.parameters()).device
        
        self.split_words = [
            # Coordinating conjunctions
            "and",
            "or",
            "but",
            "nor",
            "yet",
            "so",
            # Subordinating conjunctions / complementizers
            "because",
            "although",
            "though",
            "while",
            "whereas",
            "if",
            "unless",
            "since",
            "after",
            "before",
            "when",
            "whenever",
            "once",
            "until",
            "as",
            # Relative / wh- words
            "which",
            "that",
            "who",
            "whom",
            "whose",
            "where",
            # Discourse connectives
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "nevertheless",
            "consequently",
            "meanwhile",
            "instead",
            "otherwise",
            "likewise",
            "similarly",
        ]
    def _init_punctuation_ids(self, device):
        """
        初始化有效的分割token ID集合。
        包含: 基础标点 + 带空格的标点变体。

        Important: we do NOT include [SEP]/[EOS] here. Allowing [SEP] as a split point
        on early steps leads to a degenerate policy where it immediately selects the
        end-of-sequence token and all subsequent steps have no valid "future" positions.
        Instead, we keep special end tokens in a separate set and only allow them
        on the final step.
        """
      
        if hasattr(self, "_valid_split_ids") and self._valid_split_ids is not None:
            if self._valid_split_ids.device == device:
                return
        
      
        punct_chars = {",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；"}
        valid_ids = set()

        # Track special end tokens separately (allowed only at final step)
        end_ids = set()
        if getattr(self.tokenizer, "sep_token_id", None) is not None:
            end_ids.add(int(self.tokenizer.sep_token_id))
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            end_ids.add(int(self.tokenizer.eos_token_id))
            
  
        for char in punct_chars:
            # Case A: 纯标点
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if ids: valid_ids.update(ids)
            
            # Case B: 带空格前缀
            ids_space = self.tokenizer.encode(" " + char, add_special_tokens=False)
            if ids_space: valid_ids.update(ids_space)

        # Add connector words as split markers (single-token only).
        for w in getattr(self, "split_words", []) or []:
            # Include both raw and space-prefixed forms for different tokenizer families.
            for ww in {w, w.capitalize()}:
                for prefix in ("", " "):
                    ids = self.tokenizer.encode(prefix + ww, add_special_tokens=False)
                    # Only accept markers that map to a single token id to avoid mid-word splits.
                    if isinstance(ids, list) and len(ids) == 1:
                        valid_ids.add(ids[0])

     
        sorted_ids = sorted(list(valid_ids))
        self._valid_split_ids = torch.tensor(sorted_ids, device=device, dtype=torch.long)
        self._end_split_ids = (
            torch.tensor(sorted(list(end_ids)), device=device, dtype=torch.long)
            if end_ids
            else None
        )
    @property
    def device(self):
        """获取模型所在的设备"""
        if hasattr(self, '_device'):
            return self._device
      
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Make checkpoint loading robust when `input_proj` is created lazily.

        Older/newer checkpoints may contain `input_proj.weight`/`input_proj.bias`, but this
        module historically set `self.input_proj = None` until the first forward pass.
        When loading with `strict=True`, those keys become "unexpected".

        Fix: if the incoming state_dict contains `input_proj.weight`, instantiate a matching
        `nn.Linear` before delegating to `nn.Module.load_state_dict`.
        """
        try:
            w_key = "input_proj.weight"
            if isinstance(state_dict, dict) and w_key in state_dict:
                w = state_dict[w_key]
                if isinstance(w, torch.Tensor) and w.ndim == 2:
                    out_features, in_features = int(w.shape[0]), int(w.shape[1])
                    need_replace = False
                    if self.input_proj is None:
                        need_replace = True
                    elif isinstance(self.input_proj, nn.Identity):
                        need_replace = True
                    elif isinstance(self.input_proj, nn.Linear):
                        need_replace = (
                            self.input_proj.in_features != in_features
                            or self.input_proj.out_features != out_features
                        )
                    else:
                        # Unknown module type: be conservative and leave it as-is.
                        need_replace = False

                    if need_replace:
                        self.input_proj = nn.Linear(in_features, out_features)
        except Exception:
            # Never fail loading due to this compatibility shim.
            pass

        # PyTorch >=2.1 supports `assign`; keep compatibility with older versions.
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def forward(self, td, env=None, phase: str = "train", select_best: bool = False, **kwargs):
        """与 RL4CO 的 REINFORCE 调用约定兼容的入口。
        根据阶段/标志选择采样或贪心解码，返回包含 reward 与 log_likelihood 的字典。
        """
       
        decode_override = kwargs.pop("decode_type", None)
        if decode_override is not None:
            decode_type = decode_override
        else:
            decode_type = "greedy" if (select_best or phase != "train") else "sampling"

        actions, log_likelihood, info = self._forward(td, decode_type=decode_type, **kwargs)
        # 计算奖励
        used_env = env if env is not None else getattr(self, 'env', None)
        if used_env is None:
            raise RuntimeError("Environment instance is required to compute reward.")
        reward = used_env.get_reward(td, actions)

        return {
            "actions": actions,
            "log_likelihood": log_likelihood,
            "reward": reward,
            "info": info,
        }

    def _forward(self, td, decode_type="sampling", **kwargs):
        """
        实现自回归解码的核心逻辑 。
        """
        debug: bool = bool(kwargs.pop("debug", False))
       
     
        target_device = next(self.lm.parameters()).device
        if getattr(self, "_current_module_device", None) != target_device:
            self.to(target_device)
            self._current_module_device = target_device
        self._device = target_device
        
   
        try:
            td = td.to(target_device)
        except Exception:
            for k in ['token_embeddings_a','token_embeddings_b','attention_mask_a','attention_mask_b',
                      'input_ids_a','input_ids_b','length_a','length_b']:
                if k in td and isinstance(td[k], torch.Tensor):
                    td[k] = td[k].to(target_device)
        self._init_punctuation_ids(td['input_ids_a'].device)
        
     
        input_dim = td['token_embeddings_a'].size(-1)
        if input_dim != self.hidden_dim:
            raise ValueError(
                f"token_embeddings_a last-dim ({input_dim}) must equal policy hidden_dim "
                f"({self.hidden_dim}). You are likely using a backbone with hidden_size={input_dim}. "
                "Either switch to a 128-dim model (e.g., prajjwal1/bert-tiny) or reintroduce a projection."
            )

        embedded_a = td['token_embeddings_a']
        embedded_b = td['token_embeddings_b']
        
        mask_a = ~td['attention_mask_a'].bool()
        mask_b = ~td['attention_mask_b'].bool()
        
        batch_size = embedded_a.size(0)
        seq_len_a = embedded_a.size(1)
        seq_len_b = embedded_b.size(1)
        
        # --- 2. Encoder (Cross-Attention) ---
        for layer in self.encoder_layers:
            embedded_a, embedded_b = layer(
                emb_a=embedded_a, 
                emb_b=embedded_b, 
                pad_mask_a=mask_a, 
                pad_mask_b=mask_b
            )
        encoder_outputs_a = embedded_a
        encoder_outputs_b = embedded_b
        
        # --- 3. Decoder 初始化 ---
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        h, c = (torch.zeros(batch_size, self.decoder_cell.hidden_size, device=self.device),
                torch.zeros(batch_size, self.decoder_cell.hidden_size, device=self.device))
        
        pointers = []
        log_probs = []
        # Debug stats to diagnose "loss not improving" (e.g., degenerate action space => logp ~ 0)
        valid_counts_a_steps = []
        valid_counts_b_steps = []
        had_any_valid_a_steps = []
        had_any_valid_b_steps = []
        
        # 初始边界设为 0 (对应 [CLS])
        current_bA = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        current_bB = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Optional per-step debug telemetry (batch_size may be >1 during training;
        # for debug we typically run bs=1 and read the first item).
        debug_steps = [] if debug else None

        # Precompute base punctuation mask (special end tokens are handled per-step)
        is_punct_base_a = torch.isin(td["input_ids_a"], self._valid_split_ids)
        is_punct_base_b = torch.isin(td["input_ids_b"], self._valid_split_ids)

        # --- 4. 解码循环 ---
        for step in range(self.max_segments):
            # (a) LSTM Step
            h, c = self.decoder_cell(decoder_input, (h, c))
            query_vec = self.attention_linear_decoder(h)

            # Allow [SEP]/[EOS] only on the final step to prevent early degeneration.
            if step == self.max_segments - 1 and getattr(self, "_end_split_ids", None) is not None:
                is_end_a = torch.isin(td["input_ids_a"], self._end_split_ids)
                is_end_b = torch.isin(td["input_ids_b"], self._end_split_ids)
                is_punct_global_a = is_punct_base_a | is_end_a
                is_punct_global_b = is_punct_base_b | is_end_b
            else:
                is_punct_global_a = is_punct_base_a
                is_punct_global_b = is_punct_base_b

            #  指针 A 选择
            range_a = torch.arange(seq_len_a, device=self.device).expand(batch_size, -1)
            
            # 1. 必须是标点符号 (预计算结果)
            # 2. 必须严格在当前边界之后 (range_a > current_bA)
            # 3. 必须在有效长度内 (range_a < length_a)
       
            
            is_future_a = range_a > current_bA.unsqueeze(1)
            is_in_length_a = range_a < td['length_a'].unsqueeze(1)
            valid_slots_a = is_punct_global_a & is_future_a & is_in_length_a
            mask_ptr_a = ~valid_slots_a
            valid_counts_a_steps.append(valid_slots_a.sum(dim=1))
            had_any_valid_a_steps.append(valid_slots_a.any(dim=1))

            inp_a = query_vec.unsqueeze(2)
            ctx_a = self.attention_linear_encoder(encoder_outputs_a.permute(0, 2, 1))
            V_exp = self.V.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            attn_scores_a = torch.bmm(V_exp, torch.tanh(inp_a + ctx_a)).squeeze(1)
            
       
            mask_fill_val = float('-inf') 
            attn_scores_a = attn_scores_a.masked_fill(mask_ptr_a, mask_fill_val)

            # === 指针 B 选择 ===
            range_b = torch.arange(seq_len_b, device=self.device).expand(batch_size, -1)
            
            is_future_b = range_b > current_bB.unsqueeze(1)
            is_in_length_b = range_b < td['length_b'].unsqueeze(1)
            
            valid_slots_b = is_punct_global_b & is_future_b & is_in_length_b
            mask_ptr_b = ~valid_slots_b
            valid_counts_b_steps.append(valid_slots_b.sum(dim=1))
            had_any_valid_b_steps.append(valid_slots_b.any(dim=1))

            inp_b = query_vec.unsqueeze(2)
            ctx_b = self.attention_linear_encoder(encoder_outputs_b.permute(0, 2, 1))
            attn_scores_b = torch.bmm(V_exp, torch.tanh(inp_b + ctx_b)).squeeze(1)
            attn_scores_b = attn_scores_b.masked_fill(mask_ptr_b, mask_fill_val)

            # === 采样/Argmax ===
            pointer_a = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            pointer_b = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            logp_a = torch.zeros(batch_size, device=self.device)
            logp_b = torch.zeros(batch_size, device=self.device)

            has_valid_a = valid_slots_a.any(dim=1)
            has_valid_b = valid_slots_b.any(dim=1)

            # Snapshot pre-sampling/argmax debug info for batch[0]
            if debug and debug_steps is not None:
                try:
                    # Use masked scores (after -inf fill) to reflect the real choice space
                    topk_k = min(5, attn_scores_a.size(1))
                    topk_a = torch.topk(attn_scores_a[0], k=topk_k).indices.detach().cpu().tolist()
                    topk_kb = min(5, attn_scores_b.size(1))
                    topk_b = torch.topk(attn_scores_b[0], k=topk_kb).indices.detach().cpu().tolist()
                    debug_steps.append(
                        {
                            "step": int(step),
                            "valid_count_a": int(valid_slots_a[0].sum().item()),
                            "valid_count_b": int(valid_slots_b[0].sum().item()),
                            "topk_pos_a": topk_a,
                            "topk_pos_b": topk_b,
                            # Fill chosen pointers later after selection
                        }
                    )
                except Exception:
                    debug_steps.append({"step": int(step), "error": "failed_to_collect_topk"})

            # 处理 A
            if has_valid_a.any():
                rows = has_valid_a
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_a[rows])
                    act = dist.sample()
                    pointer_a[rows] = act
                    logp_a[rows] = dist.log_prob(act)
                else:
                    pointer_a[rows] = torch.argmax(attn_scores_a[rows], dim=1)

            if (~has_valid_a).any():
                rows = ~has_valid_a
              
                pointer_a[rows] = td['length_a'][rows] - 1
             

            # 处理 B
            if has_valid_b.any():
                rows = has_valid_b
                if decode_type == "sampling":
                    dist = Categorical(logits=attn_scores_b[rows])
                    act = dist.sample()
                    pointer_b[rows] = act
                    logp_b[rows] = dist.log_prob(act)
                else:
                    pointer_b[rows] = torch.argmax(attn_scores_b[rows], dim=1)
            
            if (~has_valid_b).any():
                rows = ~has_valid_b
                pointer_b[rows] = td['length_b'][rows] - 1

            if debug and debug_steps is not None:
                # Record selected pointers and whether we had to fall back (batch[0])
                try:
                    debug_steps[-1].update(
                        {
                            "pointer_a": int(pointer_a[0].item()),
                            "pointer_b": int(pointer_b[0].item()),
                            "fallback_a": bool((~has_valid_a)[0].item()),
                            "fallback_b": bool((~has_valid_b)[0].item()),
                        }
                    )
                except Exception:
                    pass

            pointers.append(torch.stack([pointer_a, pointer_b], dim=1))
            log_probs.append(logp_a + logp_b)

            # ---------------------------------------------------------------------
            # Fast feedback embedding: reuse token-level embeddings already computed
            # outside the RL loop (MaxSimSplitter.split_pair_return_segments does one
            # LM forward per text). We slice the span corresponding to the chosen
            # boundary and mean-pool it, matching MaxSimEnv's "FAST VERSION":
            #   real_start = (prev + 1) if prev > 0 else 1   # skip [CLS] at index 0
            #   end        = pointer + 1                    # inclusive boundary token
            # ---------------------------------------------------------------------
            token_emb_a = td["token_embeddings_a"]  # [B, L, H] on self.device already
            length_a = td["length_a"].long()  # [B]

            feedback_emb_a_list = []
            for b in range(batch_size):
                s_a = int(current_bA[b].item())
                e_a = int(pointer_a[b].item())
                la = int(length_a[b].item())
                la = max(1, la)

                # Policy semantics: skip [CLS] (idx 0), boundary token is inclusive.
                real_start_a = (s_a + 1) if s_a > 0 else 1
                real_end_a = e_a + 1

                # Clip to effective length to avoid pooling padded tail.
                real_start_a = min(max(0, real_start_a), la)
                real_end_a = min(max(0, real_end_a), la)

                if real_end_a <= real_start_a:
                    # Match previous behavior: empty segment -> zeros embedding.
                    emb_val = torch.zeros(
                        token_emb_a.size(-1), device=token_emb_a.device, dtype=token_emb_a.dtype
                    )
                else:
                    emb_val = token_emb_a[b, real_start_a:real_end_a, :].mean(dim=0)

                feedback_emb_a_list.append(emb_val)

            feedback_tensor_a = torch.stack(feedback_emb_a_list, dim=0).to(self.device)
            
            # As we enforce input_dim == hidden_dim, feedback embedding dim must match too.
            if feedback_tensor_a.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"feedback embedding last-dim ({feedback_tensor_a.size(-1)}) must equal hidden_dim "
                    f"({self.hidden_dim}). Check embedding model hidden_size."
                )
            decoder_input = feedback_tensor_a

            # 更新状态
            current_bA = pointer_a
            current_bB = pointer_b

     
        actions = torch.stack(pointers, 1).view(batch_size, -1)
        log_likelihood = torch.stack(log_probs, 1).sum(dim=1)

        info = {}
        try:
            # [batch, steps] -> scalar stats
            vca = torch.stack(valid_counts_a_steps, dim=1).float()
            vcb = torch.stack(valid_counts_b_steps, dim=1).float()
            hva = torch.stack(had_any_valid_a_steps, dim=1)
            hvb = torch.stack(had_any_valid_b_steps, dim=1)

            # These are high-signal indicators:
            # - If avg_valid_* ~ 1.0 and frac_any_valid_* ~ 1.0, then logp tends to 0 => loss ~ 0
            # - If frac_any_valid_* is low, you fall back to (length-1) often => logp still ~ 0
            info.update(
                {
                    "avg_valid_a": vca.mean().detach(),
                    "avg_valid_b": vcb.mean().detach(),
                    "min_valid_a": vca.min().detach(),
                    "min_valid_b": vcb.min().detach(),
                    "frac_any_valid_a": hva.float().mean().detach(),
                    "frac_any_valid_b": hvb.float().mean().detach(),
                    "mean_log_likelihood": log_likelihood.detach().mean(),
                    "frac_zero_log_likelihood": (log_likelihood.detach().abs() < 1e-8).float().mean(),
                }
            )
        except Exception:
            # Never break training due to debug stats
            pass

        if debug and debug_steps is not None:
            # Python-native list of dicts for easy printing/logging
            info["debug_steps"] = debug_steps

        return actions, log_likelihood, info