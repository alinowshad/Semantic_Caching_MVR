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



## New Training Algorithm
We modified our training algorithm logic whihc is now more coherent with the inference time. 


```bash
  python /root/Semantic_Caching_MVR-main/RL_SemanticCashing_768/RL4COTrainer.py \
  --devices 4 \
  --nn_similarity_mode full \
  --nn_similarity_dtype float32 \
  --ddp_find_unused_parameters \
  --train_parquet /root/LMArena/train_30k.parquet \
  --val_parquet /root/LMArena/train_3k.parquet \
  --test_parquet /root/LMArena/test_1k.parquet \
  --parquet_text_column prompt \
  --train_sampling_mode anchor_nn \
  --nn_warmup_epochs 10 \
  --nn_candidate_topk 10 \
  --nn_rebuild_every_n_epochs 1 \
  --label_mode id_set \
  --train_data_size 30000 \
  --batch_size 64 \
  --accumulate_grad_batches 2 \
  --lr 1e-4 \
  --max_epochs 200 \
  --check_val_every_n_epoch 5 \
  --checkpoint_dir /root/data/checkpointsv3 \
  --policy_mode separate \
  --split_on_space \
  --split_words_before \
  --debug_policy \
  --debug_policy_log_path /root/data/checkpointsv3/policy_debug.jsonl \
  --debug_policy_every_n_epochs 1 \
  --debug_policy_batch_size 8 \
  --debug_policy_topk 10 \
  --debug_policy_n_samples 2 \
  --bce_auto_balance
```

basically some of the hyperparameters are the same as before, however, first is regarding the train data, to make a training faster you can choose a small subset of training data lik 2k-5k (based on the performance gains), you can just randomly select this data from the original parquet file of the dataset. As you can see here for example I choosed 10k subset (but this is quite large and training becomes very slow according to the logic we are choosing). Next thing is that you need to you the "**--train_sampling_mode anchor_nn**" this hyperparameter which is the correct trainig logic and the most important parameter. Next hyperparameter **--nn_warmup_epochs 5** this is the warmup stage since in the initial stages the segmentation model is not well trained yet, we use the single vector for tuning and finding the nearest neighbours (nn) so here we are saying basically for the first 5 epoch use the warmup (single vector, so that the model learns and have stable trainig). This is also another important parameter which effects the speed of the training a lot **--nn_candidate_topk 50**, so in genearal in our new training logic we select a random prompt x and then we find all the y samples that their nearest neighbour is the prompt x, and then we do the prompt segmentation and bce loss accordingly, however this requires us to build a matrix of similarity for all the training samples at each step to find the NN, for multi-vector this operation is expensive (after warmup) so I set this parameter so that in the large datasets we choose the top N single vector nearest neighbour and then we select the most nearest with the multivector operations, so you can adjust this parameter accordingly if your dataset is smaller you can raise this parameter, if it is larger you can decrease this parameter. These are also **--policy_mode separate   --split_on_space   --split_words_before** also important as you know we choose the seperate structure and maybe you can play with the trainig to whether use the space segmentation or not (based on the performance observed). You can keep the debug log this will help you to analyze the training to see if it is going well. Moreover, this part we say how many GPUs we use for faster training and also whether we build the full NxN similarity matrix and put it on the ram (depends on the memory available)  **--devices 4  --nn_similarity_mode full  --nn_similarity_dtype float32  --ddp_find_unused_parameters **. Furthermore, the last important thing is that **--bce_auto_balance**, since we are using and updating the model on the nearest neigbour we are possibly favouring the positive class more, so this will create an imbalance problem, I set this parameter to automatically weight the classes accordingly to make it balance. However, you can double check everything to ensure that everything is correct or making sure that I didn't miss anything.
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

#### 1) Verify the custom APIs are present

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

#### 2) Run the evaluation (CUDA)

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

```

to test a group of candidate-k and delta,run command like this:
```bash
poetry run python benchmarks/eval_sembenchmark_verified_splitter.py \
  --dataset /home/zhengzishan/Semantic_Caching_MVR/vcahce/datasets/filtered_sembenchmark_train.csv \
  --llm-col response_llama_3_8b \
  --deltas 0.01 0.015 0.02 0.03 0.05 0.07 0.08 \
  --candidate-selection multivector_top_k \
  --candidate-ks 5 10 \
  --splitter-checkpoint ~/checkpoints_words/epoch=29-step=1620.ckpt \
  --splitter-device cuda:3 \
  --similarity-evaluator string \
  --sleep 0.1 \
  --output-json results/local_verified_splitter.json \
  --benchmark-output-dir results/benchmark_compat \
  --benchmark-run-index 1
```
results/benchmark_compat directory is for the convenience of drawing graphs with benchmark.py.

## Baselines and Benchmark Scripts

in the benchmark folder there is file called "/vcahce/benchmarks/benchmark.py" in this file there are all the based line that is the original vcache paper, you can create scripts like eval_sembenchmark_verified* and run those baselines on those datasets.

vcahce/benchmarks/eval_sembenchmark_verified_splitter.py (the original version,without new hnswlib) has been intergrated into vcahce/benchmarks/benchmark*.py to automatically generate various comparison graphs.So you can just run benchmark*.py on three datasets.

---

## Datasets
The dataset I used for inference has had the training data removed which is located at /data1/wuyinjun/semantic_cache_dataset/dataset/filtered_sembenchmark_classification.csv,/data1/wuyinjun/semantic_cache_dataset/dataset/filtered_SemBenchmarkLmArena_train.csv,/data1/wuyinjun/semantic_cache_dataset/dataset/filtered_SemBenchmarksqArena_train.csv.All datasets for training is also at 
/data1/wuyinjun/semantic_cache_dataset/dataset/.

## Environment Variables (Required)

Since our server is not connected to vpn you need to set the proper environmental variables before running any command in you system.

```bash
expoert HF_ENDPOINT= https://hf-mirror.com

export HF_CACHE_BASE=YOUR_CACHING_PATH

export HF_TOKEN=YOUR_hf_token (you can easily create on hf, this is needed since our code usually makes multiple requests) 
```

run these commands on you system before you run the code
