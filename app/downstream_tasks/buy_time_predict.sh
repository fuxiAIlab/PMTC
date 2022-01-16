# bash buy_time_predict.sh
set -e

test_n_split=5
task_name=buy_time_predict

# Data:
debug_N_range=(256)

task_data_path=../../data/downstream_tasks/predict_pay_time.h5

export PYTHONHASHSEED=0

for ((i=0;i<${#debug_N_range[@]};++i))
do
  debug_N=${debug_N_range[i]}
  echo "train_size: $debug_N"
  python3.6 main.py --task_data_path $task_data_path \
                    --task_name $task_name \
                    --pretrain_models bpe_bow \
                                      game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_mlmprob-0.15_mask_type-normal_t-0.5_alpha-0.5_DEBUG50 \
                    --compare_finetune 0 \
                    --debug_N $debug_N \
                    --feature_choices '0 0 1' \
                    --is_debug 1 \
                    --test_n_split $test_n_split \
                    --gpus 0
done