# bash clustering.sh

set -e

debug_N=0
test_n_split=5
task_name=clustering

task_data_path=../../data/downstream_tasks/role_id_clustering.h5

pretrain_models="game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_mlmprob-0.15_mask_type-normal_t-0.5_alpha-0.5_DEBUG50"

python3.6 main.py --task_data_path $task_data_path \
                  --task_name $task_name \
                  --pretrain_models $pretrain_models \
                  --compare_finetune 0 \
                  --debug_N $debug_N \
                  --feature_choices '0 0 1' \
                  --is_debug 0 \
                  --test_n_split $test_n_split \
                  --gpus 0