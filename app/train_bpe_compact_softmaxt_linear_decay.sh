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
mask_softmax_t=${13:-0.1}
debugN=${14:-0}

export CUDA_LAUNCH_BLOCKING=1
export PYTHONHASHSEED=0

echo use_time_embed: $use_time_embed

# 重新跑之前检查一下参数！！！！！！！ MASK prob之类的
# -------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------
# bash train_bpe_compact_softmaxt_linear_decay.sh 512 3 2 5 1 bert 512 0 4 mlm 0.15 0.5 0.00001 50
# small DEBUG
# bash train_bpe_compact_softmaxt_linear_decay.sh 512 3 2 5 1 bert 512 0 8 mlm 0.15 0.5 0.00001 200
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
# 128GB内存版本：
# bash train_bpe_compact_softmaxt_linear_decay.sh 512 3 8 20 1 bert 80000 0 32 mlm 0.15 0.5 0.1 0
# 64GB内存版本 (latest: 85704)
# bash train_bpe_compact_softmaxt_linear_decay.sh 512 3 8 20 1 bert 40000 0 32 mlm 0.15 0.5 0.1 0
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 一天内能跑完的版本
# -------------------------------------------------------------------
# bash train_bpe_compact_softmaxt_linear_decay.sh 512 3 8 20 1 bert 40000 0 32 mlm 0.15 0.5 0.1 200000
# -------------------------------------------------------------------

output_dir=../bert_model/game
max_token_vocab_size=50000
bpe_tokenizer_path=../static/bpe_new.str

# [UBUNTU] or [DANLU]
if [ "$(hostid)" == 007f0101 ]
then
#   bpe_tokenizer_path=../data/sample_processed/cn_char_bert_2048_timestamp_gamelog_50w_50000.str
    h5_data_file_path=/media/iamlxb3/2D97AD940A9AD661/data/game_bert/cn_char_bert_4096_origin_timestamp_gamelog.h5
    #h5_data_file_path=../data/sample_processed/cn_char_bert_4096_origin_timestamp_gamelog_debug.h5
else
    #bpe_tokenizer_path=../data/sample_processed/bpe_new.str
    #h5_data_file_path=../data/sample_processed/cn_char_bert_4096_origin_timestamp_gamelog_debug.h5
    h5_data_file_path=../data/sample_processed/cn_char_bert_4096_origin_timestamp_gamelog.h5
fi

# other settings
max_time_gap=1024
is_timestamp_rnn=0
clm_sample_n=1
mask_type=posterior_prob # posterior_prob, tf_idf, posterior_prob_with_tf_idf_warmup
softmax_t_decay_mode=linear
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
  --mask_softmax_t $mask_softmax_t \
  --softmax_t_decay_mode $softmax_t_decay_mode \
  --mask_type $mask_type \
  --learning_rate $learning_rate

