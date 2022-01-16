+ : Key different from BERT
@ : Value different from BERT

### ALBERT

inner_group_num (int, optional, defaults to 1) – The number of inner repetition of attention and ffn.
num_hidden_groups (int, optional, defaults to 1) – Number of groups for the hidden layers, parameters in the same group are shared.

'''json
AlbertConfig {
  @ "attention_probs_dropout_prob": 0,
  +@ "bos_token_id": 2, 
  +@ "classifier_dropout_prob": 0.1,
  "embedding_size": 128,
  +@ "eos_token_id": 3,
  "hidden_act": "gelu_new",
  @ "hidden_dropout_prob": 0,
  @ "hidden_size": 4096,
  "initializer_range": 0.02,
  +@ "inner_group_num": 1,
  @ "intermediate_size": 16384,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  @ "num_attention_heads": 64,
  +@ "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30000
}
'''

### BERT
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30522
}

### Reformer
ReformerConfig {
  "attention_head_size": 64,
  "attn_layers": [
    "local",
    "lsh",
    "local",
    "lsh",
    "local",
    "lsh"
  ],
  "axial_norm_std": 1.0,
  "axial_pos_embds": true,
  "axial_pos_embds_dim": [
    64,
    192
  ],
  "axial_pos_shape": [
    11,
    11
  ],
  "chunk_size_lm_head": 0,
  "eos_token_id": 2,
  "feed_forward_size": 512,
  "hash_seed": null,
  "hidden_act": "relu",
  "hidden_dropout_prob": 0.05,
  "hidden_size": 256,
  "initializer_range": 0.02,
  "is_decoder": true,
  "layer_norm_eps": 1e-12,
  "local_attention_probs_dropout_prob": 0.05,
  "local_attn_chunk_length": 64,
  "local_num_chunks_after": 0,
  "local_num_chunks_before": 1,
  "lsh_attention_probs_dropout_prob": 0.0,
  "lsh_attn_chunk_length": 64,
  "lsh_num_chunks_after": 0,
  "lsh_num_chunks_before": 1,
  "max_position_embeddings": 4096,
  "model_type": "reformer",
  "num_attention_heads": 12,
  "num_buckets": null,
  "num_hashes": 1,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "tie_word_embeddings": false,
  "vocab_size": 320
}