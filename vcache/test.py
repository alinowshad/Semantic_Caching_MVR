from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.silicon_flow import SiliconFlowInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.silicon_flow import (
    SiliconFlowEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.similarity_evaluator.strategies.string_comparison import (
    StringComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.verified import (
    VerifiedDecisionPolicy,
)
from vcache.vcache_policy.vcache_policy import VCachePolicy

vcache_policy: VCachePolicy = VerifiedDecisionPolicy(delta=0.02)
vcache_config: VCacheConfig = VCacheConfig(
    inference_engine=SiliconFlowInferenceEngine(),
    embedding_engine=SiliconFlowEmbeddingEngine(),
    vector_db=HNSWLibVectorDB(
        similarity_metric_type=SimilarityMetricType.COSINE,
        max_capacity=100000,
    ),
    embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
    similarity_evaluator=StringComparisonSimilarityEvaluator,
)
vcache: VCache = VCache(vcache_config, vcache_policy)
