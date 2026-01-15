# Instruction to run the codes

## Code Versions and Directories

There are two versions of the code 128 and 768:

We have two different directory for the code ones is for training and one is caching inference (caching system)

---

## Training

The training code is similar to the original training code I have just made some modification for better results and resolving some bugs, furthermore I have added the calibrator and also the training procedure has been changed, 
it is now based on the pair of labeled samples, you can run the training code by executing the following command:

```bash
 python RL4COTrainer.py   --train_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_train.json   --val_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_val.json   --test_pairs_json /data1/wuyinjun/semantic_cache_dataset/dataset/semantic_prompt_cache_benchmark2500_pairs_balanced_test.json   --checkpoint_dir /data2/ali/checkpoints_seperate   --policy_mode separate   --debug_policy   --debug_policy_log_path /data2/ali/checkpoints_seperate/policy_debug.jsonl   --debug_policy_every_n_epochs 1   --debug_policy_batch_size 8   --debug_policy_topk 10   --debug_policy_n_samples 2
```

I have added the debug policy in case you want to check the results of the segmentation, you can run this command in both of the directory of the two versions.

---

## Caching Inference (Caching System)

The caching system algorithm is taken from the original "https://github.com/vcache-project/vCache/tree/master" repo, it includes all the baseline and I have added our method as one of the baselines (the name is verified splitter), you can try to run the caching experiments like this:

```bash
python benchmarks/eval_sembenchmark_verified_splitter.py --dataset vCache/SemBenchmarkClassification --llm-col response_llama_3_8b --delta 0.02 --similarity-evaluator string --sleep 0.1 --splitter-checkpoint /data2/ali/checkpoints_seperate --candidate-selection top_k --candidate-k 1 --splitter-device cuda --use-cached-candidate-segments --output-json results/verified_splitter_cuda_cachedcandsegments.json```

this experiments is for benchmark "SemBenchmarkClassification" dataset in our method. You need to put the sleep and play with it since it is important for sync, but it should not be included in the latency, this is a system problem not the method.

and for runing the original vcache model you can try this command:

```bash
python benchmarks/eval_sembenchmark_verified.py   --dataset vCache/SemBenchmarkClassification   --llm-col response_llama_3_8b   --delta 0.02   --similarity-evaluator string   --sleep 0.02   --device cuda --output-json results_verified.json
```

---



### Caching Inference (Caching System + HNSW-Multi)

So since the hnsw libraries are with same (our local one and original) it is better to completely create a new directory and conda environment for this caching system, otherwise if you run this installation it would delete the old HNSW which is used for the baseline evaluations.

You need to install the following local library (adjust the path based on your system)

```bash
python -m pip install -e vcache/vcache_core/cache/embedding_store/hnswlib
```


This is just for verification that we are actually using the HNSW-Multi (Multivector HNSW)

### 3) Verify the custom APIs are present

```bash
python - <<'PY'
import hnswlib
p = hnswlib.Index(space="ip", dim=4)
missing = []
if not hasattr(p, "knn_query_skipping_duplicates_with_parent"):
    missing.append("knn_query_skipping_duplicates_with_parent")
print("hnswlib OK" if not missing else f"hnswlib MISSING: {missing}")
PY
```

If this prints `hnswlib MISSING: ...`, you’re not importing the custom fork (or it didn’t build).

---

This is the command that is used for evaluation of the script, please pay attention to the arugments (--candidate-selection multivector_top_k \ --candidate-k 10) and for all the evaluation please set the **--sleep** (IT IS VERY IMPORTANT)

## Run the evaluation (CUDA)

From the repo root:

```bash
python benchmarks/eval_sembenchmark_verified_splitter.py \
  --dataset vCache/SemBenchmarkClassification \
  --llm-col response_llama_3_8b \
  --delta 0.02 \
  --similarity-evaluator string \
  --sleep 0.1 \
  --splitter-checkpoint /data2/ali/checkpoints_seperate \
  --candidate-selection multivector_top_k \
  --candidate-k 10 \
  --splitter-device cuda \
  --output-json results/verified_splitter_cuda_multivector.json



## Baselines and Benchmark Scripts

in the benchmark folder there is file called "/vcahce/benchmarks/benchmark.py" in this file there are all the based line that is the original vcache paper, you can create scripts like eval_sembenchmark_verified* and run those baselines on those datasets.

---

## Environment Variables (Required)

Since our server is not connected to vpn you need to set the proper environmental variables before running any command in you system.

```bash
expoert HF_ENDPOINT= https://hf-mirror.com

export HF_CACHE_BASE=YOUR_CACHING_PATH

export HF_TOKEN=YOUR_hf_token (you can easily create on hf, this is needed since our code usually makes multiple requests) 
```

run these commands on you system before you run the code
