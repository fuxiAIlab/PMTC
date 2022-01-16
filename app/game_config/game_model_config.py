from easydict import EasyDict as edict

NUM_HIDDEN_LAYERS = 6
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 768
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_HEADS = 12

# general config
game_config = edict()
game_config.num_hidden_layers = NUM_HIDDEN_LAYERS
game_config.intermediate_size = INTERMEDIATE_SIZE
game_config.max_position_embeddings = 1024
game_config.attention_window = 1024

# longformer config
longformer_config = edict()
longformer_config.attention_window = [32, 32, 32, 32, 32, 32]

# transformer_xl config
transformer_xl_config = edict()
transformer_xl_config.n_layer = NUM_HIDDEN_LAYERS
transformer_xl_config.d_inner = INTERMEDIATE_SIZE
transformer_xl_config.d_model = INTERMEDIATE_SIZE  # Dimensionality of the model’s hidden states.
transformer_xl_config.d_embed = INTERMEDIATE_SIZE  # Dimensionality of the embeddings

# albert config
albert_config = edict()
albert_enlarge_ratio = 1
albert_config.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
albert_config.hidden_size = HIDDEN_SIZE * albert_enlarge_ratio
albert_config.inner_group_num = 1
albert_config.num_hidden_groups = 1
albert_config.intermediate_size = INTERMEDIATE_SIZE * albert_enlarge_ratio
albert_config.num_attention_heads = ATTENTION_HEADS * albert_enlarge_ratio
albert_config.num_hidden_layers = NUM_HIDDEN_LAYERS

# reformer config
reformer_config = edict()
reformer_config.axial_pos_value = 64
reformer_config.axial_pos_embds_value = 32
reformer_config.hidden_size = int(HIDDEN_SIZE / 4)
reformer_config.axial_pos_embds_dim = [reformer_config.axial_pos_embds_value,
                                       int(reformer_config.hidden_size - reformer_config.axial_pos_embds_value)]
reformer_config.feed_forward_size = int(INTERMEDIATE_SIZE / 4)
reformer_config.num_hidden_layers = NUM_HIDDEN_LAYERS
reformer_config.num_attention_heads = int(ATTENTION_HEADS / 4)
reformer_config.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
