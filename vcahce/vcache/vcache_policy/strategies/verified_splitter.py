import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

from vcache.config import VCacheConfig
from vcache.inference_engine import InferenceEngine
from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.similarity_evaluator import SimilarityEvaluator
from vcache.vcache_policy.strategies.verified import _Action, _Algorithm
from vcache.vcache_policy.vcache_policy import VCachePolicy

class CallbackQueue(queue.Queue):
    """
    Same helper as in VerifiedDecisionPolicy: serialize cache updates in one worker thread.
    """

    def __init__(self, callback_function):
        super().__init__()
        self.callback_function = callback_function
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        while True:
            if self._stop_event.is_set():
                break
            try:
                item = self.get(timeout=1)
                if item is None:
                    break
                self.callback_function(item)
                self.task_done()
            except queue.Empty:
                continue

    def start(self):
        self.worker_thread.start()

    def stop(self):
        if self.worker_thread.is_alive():
            self.put(None)
            self.worker_thread.join()


class VerifiedSplitterDecisionPolicy(VCachePolicy):
    """
    VerifiedDecisionPolicy variant where the similarity score is computed via:

      (query_prompt, cached_prompt) -> RL splitter (MaxSimSplitter) -> MaxSim similarity.

    The explore/exploit decision + Bayesian thresholding logic is reused from the original
    verified policy via `_Algorithm`.
    """

    def __init__(
        self,
        delta: float = 0.01,
        splitter=None,
        device: str = "cpu",
        candidate_selection: str = "top_k",
        candidate_k: int = 10,
        use_cached_candidate_segments: bool = False,
        score_mode: str = "avg_full_and_maxsim",
    ):
        self.bayesian = _Algorithm(delta=delta)
        self.similarity_evaluator: Optional[SimilarityEvaluator] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.cache: Optional[Cache] = None
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.callback_queue: Optional[CallbackQueue] = None

        # RL splitter instance (expected: vcache.vcache_core.splitter.MaxSimSplitter.MaxSimSplitter)
        self.splitter = splitter
        self.device = device
        # How to choose which cached prompts to score with MaxSim:
        # - "top_k": get k candidates from the vector DB (fast), then rerank by MaxSim (accurate).
        # - "all": score against all cached prompts by MaxSim (slow, but MaxSim everywhere).
        self.candidate_selection = candidate_selection
        self.candidate_k = candidate_k
        # If True, cache a candidate's MaxSim segment-embedding tensor in metadata at insert time
        # (or lazily on first use), and at query time only segment the query (single-text forward).
        self.use_cached_candidate_segments = bool(use_cached_candidate_segments)
        # How to compute the *final* similarity score used by the Bayesian thresholding:
        # - "avg_full_and_maxsim": 0.5 * (full cosine + symmetric MaxSim), both mapped to [0,1]
        # - "legacy_weighted": legacy behavior (coarse weight ~0, (row+col)/2 mapped to [0,1])
        self.score_mode = str(score_mode).lower().strip()

    @staticmethod
    def _maxsim_from_tensors(query_tensor, corpus_tensor, *, score_mode: str = "avg_full_and_maxsim") -> float:
        """
        Compute symmetric MaxSim similarity in raw cosine space (typically [-1, 1]) given:
          - query_tensor:  [S_q+1, H] (last row is full embed)
          - corpus_tensor: [S_c+1, H] (last row is full embed)

        NOTE: We intentionally treat the full embedding (last row) as an additional "segment"
        for MaxSim, so it participates in the MaxSim row/col aggregation.

        NOTE: The policy previously combined a separate "full embedding cosine" term with MaxSim
        and optionally mapped/clamped scores. This implementation intentionally returns only the
        symmetric MaxSim (avg of row/col) with no mapping/clamping.
        """
        import torch
        import torch.nn.functional as F

        dev = query_tensor.device
        qt = query_tensor.to(dtype=torch.float32)
        ct = corpus_tensor.to(dtype=torch.float32, device=qt.device)

        # MaxSim over *all* rows, including the last "full embedding" row.
        query_vectors = qt
        corpus_vectors = ct
        if query_vectors.shape[0] > 0 and corpus_vectors.shape[0] > 0:
            qn = F.normalize(query_vectors, p=2, dim=-1)
            cn = F.normalize(corpus_vectors, p=2, dim=-1)
            cos = torch.mm(qn, cn.T)
            # Symmetric MaxSim in [-1,1]
            max_cos_sim_row = torch.max(cos, dim=1).values  # [Q]
            max_cos_sim_col = torch.max(cos, dim=0).values  # [C]

            # Weighted version (training-style). Cache policy has no learned per-segment weights,
            # so we use uniform weights (equivalent to mean, but keeps the weighting structure).
            query_weights = torch.ones(max_cos_sim_row.shape[0], device=dev, dtype=torch.float32)
            corpus_weights = torch.ones(max_cos_sim_col.shape[0], device=dev, dtype=torch.float32)
            row_score = torch.sum(max_cos_sim_row * query_weights) / (torch.sum(query_weights) + 1e-8)  # A->B
            col_score = torch.sum(max_cos_sim_col * corpus_weights) / (torch.sum(corpus_weights) + 1e-8)  # B->A
        else:
            row_score = torch.tensor(0.0, device=dev)
            col_score = torch.tensor(0.0, device=dev)

        # Weighted mix between row/col (uniform by default).
        mix = torch.softmax(torch.tensor([0.0, 0.0], device=dev, dtype=torch.float32), dim=0)
        raw = mix[0] * row_score + mix[1] * col_score
        return float(raw.item())

    def _maybe_cache_candidate_segments(self, embedding_id: int) -> None:
        """
        Ensure metadata for `embedding_id` has `cached_maxsim_tensor` populated.
        Safe no-op if caching is disabled or prerequisites are missing.
        """
        if not self.use_cached_candidate_segments:
            return
        if self.cache is None or self.splitter is None:
            return
        try:
            meta = self.cache.get_metadata(embedding_id=embedding_id)
        except Exception:
            return
        if meta is None:
            return
        if getattr(meta, "cached_maxsim_tensor", None) is not None:
            return

        cached_prompt = getattr(meta, "prompt", "") or ""
        if not cached_prompt:
            return
        try:
            # One-time compute per candidate prompt
            cand_tensor = self.splitter.split_text_return_maxsim_tensor(cached_prompt)
            meta.cached_maxsim_tensor = cand_tensor.detach()
            self.cache.update_metadata(embedding_id=embedding_id, embedding_metadata=meta)
        except Exception as e:
            # Keep serving even if caching fails for an entry
            self.logger.warning(f"Candidate segment caching failed for embedding_id={embedding_id}: {e}")

    def __cache_add(self, prompt: str, response: str, id_set: int) -> int:
        """
        Add (prompt,response) to cache and optionally precompute candidate segment embeddings.
        Returns embedding_id.
        """
        if self.cache is None:
            return -1
        embedding_id = self.cache.add(prompt=prompt, response=response, id_set=id_set)
        if embedding_id >= 0:
            self._maybe_cache_candidate_segments(embedding_id=embedding_id)
        return embedding_id

    def setup(self, config: VCacheConfig):
        self.similarity_evaluator = config.similarity_evaluator
        self.inference_engine = config.inference_engine
        self.cache = Cache(
            embedding_engine=config.embedding_engine,
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=config.embedding_metadata_storage,
                vector_db=config.vector_db,
            ),
            eviction_policy=config.eviction_policy,
        )

        self.callback_queue = CallbackQueue(callback_function=self.__perform_cache_update)
        self.callback_queue.start()
        self.executor = ThreadPoolExecutor(max_workers=64)

    def shutdown(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.callback_queue:
            self.callback_queue.stop()

    def _maxsim_similarity(self, query: str, candidate: str) -> float:
        """
        Compute MaxSim similarity in [0, 1] using the provided splitter + its embedding model.
        """
        if self.splitter is None:
            raise ValueError(
                "VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided."
            )

        try:
            import torch
            import torch.nn.functional as F
        except Exception as e:
            raise RuntimeError(
                "VerifiedSplitterDecisionPolicy requires torch to compute MaxSim similarity."
            ) from e

        query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors(query, candidate)
        return self._maxsim_from_tensors(query_tensor, corpus_tensor, score_mode=getattr(self, "score_mode", "avg_full_and_maxsim"))

    def _maxsim_similarity_from_encoded(self, query_enc: dict, candidate_prompt: str) -> float:
        """
        MaxSim similarity where the query has already been encoded once (token embeddings + pooled embedding).
        We still need to encode the candidate prompt once to get its token embeddings for the RL splitter.
        """
        if self.splitter is None:
            raise ValueError("VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided.")
        import torch
        import torch.nn.functional as F

        # Encode candidate once (token-level). This is the part you may want to report separately.
        t0 = time.time()
        cand_enc = self.splitter.encode_text(candidate_prompt)
        dt = time.time() - t0
        # Best-effort accumulate timing for reporting (does not affect correctness)
        try:
            self._timing_candidate_encode_s = getattr(self, "_timing_candidate_encode_s", 0.0) + float(dt)
            self._timing_candidate_encode_n = getattr(self, "_timing_candidate_encode_n", 0) + 1
        except Exception:
            pass

        query_tensor, corpus_tensor = self.splitter.split_pair_return_maxsim_tensors_from_encoded(query_enc, cand_enc)
        return self._maxsim_from_tensors(query_tensor, corpus_tensor, score_mode=getattr(self, "score_mode", "avg_full_and_maxsim"))

    def _select_nn_by_maxsim(
        self, prompt: str
    ) -> Tuple[Optional[EmbeddingMetadataObj], Optional[float]]:
        """
        Select the nearest-neighbor metadata object using MaxSim similarity.

        Returns:
            (best_metadata, best_similarity) where similarity is in [0, 1].
        """
        if self.cache is None:
            return None, None

        candidates: list[EmbeddingMetadataObj] = []

        if self.candidate_selection == "all":
            candidates = self.cache.get_all_embedding_metadata_objects()
        elif self.candidate_selection == "top_k":
            knn = self.cache.get_knn(prompt=prompt, k=max(1, int(self.candidate_k)))
            for _db_score, embedding_id in knn:
                try:
                    candidates.append(self.cache.get_metadata(embedding_id=embedding_id))
                except Exception:
                    continue
        else:
            raise ValueError(
                f"Unknown candidate_selection={self.candidate_selection!r}. Use 'top_k' or 'all'."
            )

        if not candidates:
            return None, None

        best_meta: Optional[EmbeddingMetadataObj] = None
        best_s: float = -1.0

        for meta in candidates:
            cached_prompt = getattr(meta, "prompt", "") or ""
            if not cached_prompt:
                # Can't compute MaxSim without cached prompt text; skip.
                continue
            try:
                s = self._maxsim_similarity(prompt, cached_prompt)
            except Exception as e:
                self.logger.warning(f"MaxSim similarity failed for one candidate: {e}")
                continue
            if s > best_s:
                best_s = s
                best_meta = meta

        if best_meta is None:
            return None, None
        return best_meta, best_s

    def process_request(
        self, prompt: str, system_prompt: Optional[str], id_set: int
    ) -> Tuple[bool, str, EmbeddingMetadataObj]:
        if self.inference_engine is None or self.cache is None:
            raise ValueError("Policy has not been setup")

        # Optimized path: encode query ONCE and reuse it for:
        # - KNN selection (pooled embedding)
        # - MaxSim/RL (token embeddings)
        if self.splitter is None:
            raise ValueError("VerifiedSplitterDecisionPolicy requires `splitter` (MaxSimSplitter) to be provided.")

        query_enc = self.splitter.encode_text(prompt)
        query_knn_emb = query_enc["pooled_knn"]  # torch tensor on device
        # HNSW runs on CPU; pass a CPU list[float]
        query_knn_emb_cpu = query_knn_emb.detach().float().cpu().tolist()

        nn_metadata, similarity_score = self._select_nn_by_maxsim_with_query(prompt, query_enc, query_knn_emb_cpu)
        if nn_metadata is None or similarity_score is None:
            response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self.__cache_add(prompt=prompt, response=response, id_set=id_set)
            return False, response, EmbeddingMetadataObj(embedding_id=-1, response="")

        action = self.bayesian.select_action(similarity_score=similarity_score, metadata=nn_metadata)

        match action:
            case _Action.EXPLOIT:
                return True, nn_metadata.response, nn_metadata
            case _Action.EXPLORE:
                response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
                self.__update_cache(
                    response=response,
                    nn_metadata=nn_metadata,
                    similarity_score=similarity_score,
                    embedding_id=nn_metadata.embedding_id,
                    prompt=prompt,
                    label_id_set=id_set,
                )
                return False, response, nn_metadata

    def _select_nn_by_maxsim_with_query(
        self, prompt: str, query_enc: dict, query_knn_emb_cpu: list[float]
    ) -> Tuple[Optional[EmbeddingMetadataObj], Optional[float]]:
        """
        Like `_select_nn_by_maxsim`, but reuses the already-encoded query:
          - uses pooled embedding for vector DB KNN
          - uses query token embeddings for MaxSim/RL scoring
        """
        if self.cache is None:
            return None, None

        candidates: list[EmbeddingMetadataObj] = []
        if self.candidate_selection == "all":
            candidates = self.cache.get_all_embedding_metadata_objects()
        elif self.candidate_selection == "top_k":
            knn = self.cache.get_knn_from_embedding(
                embedding=query_knn_emb_cpu, k=max(1, int(self.candidate_k))
            )
            for _db_score, embedding_id in knn:
                try:
                    candidates.append(self.cache.get_metadata(embedding_id=embedding_id))
                except Exception:
                    continue
        else:
            raise ValueError(
                f"Unknown candidate_selection={self.candidate_selection!r}. Use 'top_k' or 'all'."
            )

        if not candidates:
            return None, None

        # If we are using cached candidate segment tensors, compute the query tensor ONCE.
        query_tensor = None
        if self.use_cached_candidate_segments and self.splitter is not None:
            try:
                query_tensor = self.splitter.split_text_return_maxsim_tensor_from_encoded(query_enc)
            except Exception as e:
                self.logger.warning(f"Query single-text segmentation failed; falling back to pair MaxSim path: {e}")
                query_tensor = None

        best_meta: Optional[EmbeddingMetadataObj] = None
        best_s: float = -1.0

        for meta in candidates:
            cached_prompt = getattr(meta, "prompt", "") or ""
            if not cached_prompt:
                continue
            try:
                if query_tensor is not None:
                    cand_tensor = getattr(meta, "cached_maxsim_tensor", None)
                    if cand_tensor is None:
                        # Lazy-fill cache for this candidate (still avoids re-encoding it next time)
                        try:
                            cand_tensor = self.splitter.split_text_return_maxsim_tensor(cached_prompt)
                            meta.cached_maxsim_tensor = cand_tensor.detach()
                            self.cache.update_metadata(embedding_id=meta.embedding_id, embedding_metadata=meta)
                        except Exception:
                            cand_tensor = None
                    if cand_tensor is None:
                        # Fall back to existing encoded-pair path for correctness
                        s = self._maxsim_similarity_from_encoded(query_enc, cached_prompt)
                    else:
                        s = self._maxsim_from_tensors(query_tensor, cand_tensor, score_mode=getattr(self, "score_mode", "avg_full_and_maxsim"))
                else:
                    s = self._maxsim_similarity_from_encoded(query_enc, cached_prompt)
            except Exception as e:
                self.logger.warning(f"MaxSim similarity failed for one candidate: {e}")
                continue
            if s > best_s:
                best_s = s
                best_meta = meta

        if best_meta is None:
            return None, None
        return best_meta, best_s

    def __update_cache(
        self,
        response: str,
        nn_metadata: EmbeddingMetadataObj,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        label_id_set: int,
    ) -> None:
        if self.executor is None:
            raise ValueError("Executor not initialized. Call setup() first.")

        self.executor.submit(
            self.__submit_for_background_update,
            response,
            similarity_score,
            embedding_id,
            prompt,
            nn_metadata.response,
            label_id_set,
            nn_metadata.id_set,
        )

    def __submit_for_background_update(
        self,
        new_response: str,
        similarity_score: float,
        embedding_id: int,
        prompt: str,
        cached_response: str,
        label_id_set: int,
        nn_id_set: int,
    ):
        if self.similarity_evaluator is None or self.callback_queue is None:
            return

        should_have_exploited = self.similarity_evaluator.answers_similar(
            a=new_response, b=cached_response, id_set_a=label_id_set, id_set_b=nn_id_set
        )

        self.callback_queue.put(
            (
                should_have_exploited,
                new_response,
                similarity_score,
                embedding_id,
                prompt,
                label_id_set,
            )
        )

    def __perform_cache_update(self, update_args: tuple) -> None:
        (
            should_have_exploited,
            new_response,
            similarity_score,
            embedding_id,
            prompt,
            id_set,
        ) = update_args

        if self.cache is None:
            return

        try:
            latest_metadata_object = self.cache.get_metadata(embedding_id=embedding_id)
        except (ValueError, KeyError):
            return

        if latest_metadata_object is None:
            return

        try:
            self.bayesian.add_observation_to_metadata(
                similarity_score=similarity_score,
                is_correct=should_have_exploited,
                metadata=latest_metadata_object,
            )
        except (ValueError, KeyError):
            return

        if not should_have_exploited:
            self.__cache_add(prompt=prompt, response=new_response, id_set=id_set)

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metadata_object
            )
        except (ValueError, KeyError):
            return


