# RL4CO module overview

- **AdaptedPointerNetworkPolicy.py**
  Pointer-network style policy for two texts: a cross-attention encoder aligns both sequences and an autoregressive LSTM decoder selects punctuation-constrained cut positions for A and B
- **MaxSimEnv.py**
  RL environment defining specs, reset/step, and reward. It reconstructs segments from chosen indices and scores them via cosine similarities with a frozen embedding model.

- **MaxSimGenerator.py**
  Data generator that samples two prompts and returns token-level embeddings, masks, lengths, and ids as a TensorDict for the env/policy; truncation controlled by `max_len`.

- **embedding_model.py**
  Thin wrapper around HuggingFace AutoModel/Tokenizer providing sentence- and token-level embeddings

- **RL4COTrainer.py**
  Training script
