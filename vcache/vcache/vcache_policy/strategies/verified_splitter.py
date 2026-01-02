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

        segments_a, segments_b = self.splitter.split_pair_return_segments(query, candidate)

        # Keep embeddings + similarity math on the policy device to avoid GPU->CPU->GPU round trips.
        # NOTE: We preserve the exact per-text behavior of `get_embedding_tensor` (batch_size=1 => no pad)
        # while batching by using masked-mean pooling in `get_embeddings_tensor`.
        emb = self.splitter.embedding_model
        # Use the embedding model's *actual* parameter device to avoid cuda vs cuda:0 mismatches
        # that would trigger expensive `model.to(device)` moves in hot loops.
        try:
            dev = next(emb.model.parameters()).device
        except Exception:
            dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device

        # Batch embeddings: 2 model forwards instead of (len(segments_a)+len(segments_b)+2) forwards.
        a_texts = list(segments_a) + [query]
        b_texts = list(segments_b) + [candidate]
        a_emb = emb.get_embeddings_tensor(a_texts, device=dev).to(dtype=torch.float32)  # [Sa+1, D]
        b_emb = emb.get_embeddings_tensor(b_texts, device=dev).to(dtype=torch.float32)  # [Sb+1, D]

        seg_a_t = a_emb[:-1, :] if a_emb.shape[0] > 1 else a_emb[:0, :]
        full_a_t = a_emb[-1:, :]
        seg_b_t = b_emb[:-1, :] if b_emb.shape[0] > 1 else b_emb[:0, :]
        full_b_t = b_emb[-1:, :]

        # query_tensor/corpus_tensor follow MaxSimEnv.raw_score_text convention:
        # [sentence_embeds..., full_embed]
        query_tensor = torch.cat([seg_a_t, full_a_t], dim=0)
        corpus_tensor = torch.cat([seg_b_t, full_b_t], dim=0)

        # Weights mimic MaxSimEnv: score_weights_raw = [-1e9, 0, 0] => coarse ~0, row/col ~0.5 each
        weights = torch.softmax(torch.tensor([-1e9, 0.0, 0.0], device=dev, dtype=torch.float32), dim=0)
        w_coarse, w_row, w_col = weights.tolist()

        coarse = F.cosine_similarity(query_tensor[-1:, :], corpus_tensor[-1:, :]).squeeze()

        query_sentence = query_tensor[:-1, :]
        corpus_sentence = corpus_tensor[:-1, :]
        if query_sentence.shape[0] > 0 and corpus_sentence.shape[0] > 0:
            qn = F.normalize(query_sentence, p=2, dim=-1)
            cn = F.normalize(corpus_sentence, p=2, dim=-1)
            cos = torch.mm(qn, cn.T)
            row_score = torch.max(cos, dim=1).values.mean()
            col_score = torch.max(cos, dim=0).values.mean()
        else:
            row_score = torch.tensor(0.0)
            col_score = torch.tensor(0.0)

        raw = (w_coarse * coarse) + (w_row * row_score) + (w_col * col_score)

        # Map [-1, 1] -> [0, 1] and clip
        s01 = float(torch.clamp((raw + 1.0) / 2.0, 0.0, 1.0).item())
        return s01

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

        # If cache is empty, this will return (None, None)
        nn_metadata, similarity_score = self._select_nn_by_maxsim(prompt)
        if nn_metadata is None or similarity_score is None:
            response = self.inference_engine.create(prompt=prompt, system_prompt=system_prompt)
            self.cache.add(prompt=prompt, response=response, id_set=id_set)
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
            self.cache.add(prompt=prompt, response=new_response, id_set=id_set)

        try:
            self.cache.update_metadata(
                embedding_id=embedding_id, embedding_metadata=latest_metadata_object
            )
        except (ValueError, KeyError):
            return


