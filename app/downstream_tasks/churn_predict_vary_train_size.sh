# bash churn_predict_vary_train_size.sh

set -e

task_data_path=../../data/downstream_tasks/churn_prediction.h5
task_name=churn_predict
test_n_split=5
is_debug=0
gpus=0

debug_N_range=(128 256 512 1024)

save_embedding_cache=0
pretrain_models1=(game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_mlmprob-0.15_mask_type-normal_t-0.5_alpha-0.5_DEBUG50)

for ((n=0;n<${#pretrain_models1[@]};++n))
do
  pretrain_models=${pretrain_models1[n]}
  echo "pretrain_models: $pretrain_models"
  # General
  for ((i=0;i<${#debug_N_range[@]};++i))
  do
    echo "train_size: ${debug_N_range[i]}"
    python3.6 main.py --task_data_path $task_data_path \
                      --task_name $task_name \
                      --pretrain_models $pretrain_models \
                      --compare_finetune 0 \
                      --debug_N ${debug_N_range[i]} \
                      --feature_choices '0 0 1' \
                      --is_debug $is_debug \
                      --test_n_split $test_n_split \
                      --save_embedding_cache $save_embedding_cache \
                      --gpus $gpus
  done
done

# for non pretrain
for ((i=0;i<${#debug_N_range[@]};++i))
do
  echo "train_size: ${debug_N_range[i]}"
  python3.6 main.py --task_data_path $task_data_path \
                    --task_name $task_name \
                    --compare_finetune 0 \
                    --debug_N ${debug_N_range[i]} \
                    --feature_choices '1 0 0' \
                    --is_debug $is_debug \
                    --test_n_split $test_n_split \
                    --save_embedding_cache $save_embedding_cache \
                    --gpus $gpus
done