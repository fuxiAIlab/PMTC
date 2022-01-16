from easydict import EasyDict as edict

# training config
train_config = edict()
train_config.epoch = 20
train_config.finetune_epoch = 5
train_config.union_recommend_epoch = 5
train_config.epoch_debug = 2
train_config.lr = 1e-3
train_config.batch_size = 32
train_config.finetune_batch_size = {
    'bot_detect': 1,
    'churn_predict': 1,
    'buy_time_predict': 2,
    'union_recommend': 1,
}
train_config.lr_scheduler_patience = 5
train_config.lr_scheduler_min_lr = 1e-7
train_config.max_seq_len_debug = 256  # Only for training, not related to embedding
train_config.max_seq_len = 400  # TODO, 这个好像只在bot detection里面用到

# Base learner tokenizer config
tokenizer_config = {}

# bot_detect
tokenizer_config['bot_detect'] = edict()
tokenizer_config['bot_detect'].unk_percent = 0.98
tokenizer_config['bot_detect'].unk_percent_debug = 0.95
tokenizer_config['bot_detect'].max_vocab = 10000
tokenizer_config['bot_detect'].max_vocab_debug = 5000

# map_preload
tokenizer_config['map_preload'] = edict()
tokenizer_config['map_preload'].unk_percent = 0.98
tokenizer_config['map_preload'].unk_percent_debug = 0.95
tokenizer_config['map_preload'].max_vocab = 10000
tokenizer_config['map_preload'].max_vocab_debug = 5000
