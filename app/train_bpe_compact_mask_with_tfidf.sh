#!/usr/bin/env bash


max_text_length=${1:-512} # max text length
epoch=${2:-2}
batch_size=${3:-1}
logging_steps=${4:-20}
ngpu=${5:-1}
model_type=${6:-'bert'}
large_batch=${7:-1024}
use_time_embed=${8:-1}
ga_steps=${9:-1}
pretrain_task=${10:-'mlm'}
mlm_probability=${11:-0.15}
dynamic_mask_update_alpha=${12:-0.5}
mask_softmax_t=${13:-0.5}
debugN=${14:-0}

export CUDA_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=0

echo use_time_embed: $use_time_embed


# 重新跑之前检查一下参数！！！！！！！ MASK prob之类的
# -------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------
# DEBUG, NO Time Embed
# bash train_bpe_compact_mask_with_tfidf.sh 512 3 2 5 1 bert 512 0 4 mlm 0.15 0.5 1 50
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
# bash train_bpe_compact_mask_with_tfidf.sh 512 3 8 20 1 bert 40000 0 32 mlm 0.15 0.5 0.5 0
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 一天能跑完的版本
# -------------------------------------------------------------------
# bash train_bpe_compact_mask_with_tfidf.sh 512 7 8 20 1 bert 40000 0 32 mlm 0.16 0.5 3 200000
# -------------------------------------------------------------------

output_dir=../bert_model/game
max_token_vocab_size=50000
bpe_tokenizer_path=../static/bpe_new.str

h5_data_file_path=../data/pre_train/game_player_behavior_sequence.h5

# other settings
max_time_gap=1024
is_timestamp_rnn=0
clm_sample_n=1
mask_type=tf_idf # posterior_prob, tf_idf, posterior_prob_with_tf_idf_warmup
idf_path=../static/bpe_token_idf_debug-20000.json
learning_rate=5e-4

echo "large_batch $large_batch"
echo "h5_data_file_path is set to $h5_data_file_path"
echo "max_time_gap $max_time_gap"

# [Mask Language Modeling]
python3.6 -m torch.distributed.launch --nproc_per_node=$ngpu train_compact_bpe.py \
  --output_dir=$output_dir \
  --model_type=$model_type \
  --do_train \
  --h5_data_file=$h5_data_file_path \
  --num_train_epochs 1 \
  --per_device_train_batch_size $batch_size \
  --block_size $max_text_length \
  --outer_epoch $epoch \
  --large_batch $large_batch \
  --logging_steps $logging_steps \
  --use_time_embed $use_time_embed \
  --max_time_gap $max_time_gap \
  --debugN $debugN \
  --overwrite_output_dir \
  --gradient_accumulation_steps $ga_steps \
  --max_token_vocab_size $max_token_vocab_size \
  --bpe_tokenizer_path $bpe_tokenizer_path \
  --use_bpe_tokenzier 1 \
  --is_timestamp_rnn $is_timestamp_rnn \
  --pretrain_task $pretrain_task \
  --clm_sample_n $clm_sample_n \
  --mlm_probability $mlm_probability \
  --dynamic_mask_update_alpha $dynamic_mask_update_alpha \
  --mask_type $mask_type \
  --mask_softmax_t $mask_softmax_t \
  --idf_path $idf_path \
  --learning_rate $learning_rate


