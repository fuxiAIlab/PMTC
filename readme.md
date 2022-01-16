# Unsupervised representation learning of Player Behavioral Data with Confidence Guided Masking

## Examples of Behavior Sequence Data

Typical examples of proactive and passive/systems logs in Justice Online:

![3](https://tva1.sinaimg.cn/large/008i3skNly1gwkc1sce0vj324r0ogwlb.jpg)

We demonstrate the results of the deployed PMTC-enhanced bot detection system. Each figure presents a cluster of similar behavioral sequences, with each color representing an event id:

![1](https://tva1.sinaimg.cn/large/008i3skNly1gwkbzrqks4j31s10kd7a6.jpg)

## Requirements

We provide a docker enviroment with all packages installed, where you can run all below scripts readily.

The tested python version is 3.6.9. For package versions, please refer to requirements.txt.

To successfully run our code in the docker container, you need at least 1 gpu and install nvidia-container-toolkit first, for details, please refer to https://dev.to/et813/install-docker-and-nvidia-container-m0j. 

To inspect the pre-training and probing procedure/code, simply clone the whole repository or just the dockerfile, and run the following line to build and run.
```shell script
sudo docker build -t behavior_sequence_pre_training . # BUILD
sudo docker run -it --gpus all behavior_sequence_pre_training bash # RUN iteratively
```
Docker repository: iamlxb3/behavior_sequence_pre_training_via_mtc

All code has been tested with minimum computation resources:
- GPU: NVIDIA GeForce 1060 TI
- RAM: 16GB
- CPU: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz


## Run pre-training experiments

### Pre-training scripts' arguments
Below we show important input arguments for the pre-training scripts (e.g. train_bpe_compact.sh).
  ```python
    max_text_length: maximum length of sequence
    epoch: training epoch
    batch_size: training batch size
    logging_steps: determines logging frequency
    ngpu: the number of GPUs to use
    model_type: transformer model type, valid options: [bert, longformer, reformer]
    large_batch: the maximum data to read in memory at a time
    ga_steps: gradident accumaltion step
    pretrain_task: pre-training task, default is Masked Language Modeling (MLM)
    mlm_probability: MLM masking ratio
    mask_softmax_t: T, temperature for sotftmax
    dynamic_mask_update_alpha: decaying coefficient alpha
    debugN: to speed debugging, the number of sequence used for pre-training
  ```
Scripts of pre-training for various models and masking strategies discussed in our paper:
- (Model) Bert_bpe + (Masking Strategy) uniform
  ```shell script
  bash train_bpe_compact.sh 512 3 2 5 1 bert 512 0 4 mlm 0.15 50
  ```
- (Model) Longformer_ws + (Masking Strategy) uniform
  ```shell script
  bash train_bpe_compact.sh 512 3 2 5 1 longformer 512 0 4 mlm 0.15 50
  ```
- (Model) Reformer_ws + (Masking Strategy) uniform
  ```shell script
  bash train_bpe_compact.sh 512 3 2 5 1 reformer 512 0 4 mlm 0.15 50
  ```
  * In order to run reformer successfully, you have to modify one line of the transformers source code.
  Change line 151 in transformers/modeling_reformer.py to
  ```python
  weight.expand(list((batch_size,)) + self.axial_pos_shape + list(weight.shape[-1:])) for weight in self.weights
  ```
- (Model) Bert_bpe + (Masking Strategy) itf-idf (T=3.0)
  ```shell script
  bash train_bpe_compact_mask_with_tfidf.sh 512 3 2 5 1 bert 512 0 4 mlm 0.15 0.5 3 50
  ```
- (Model) Bert_bpe + (Masking Strategy) MTC (T=0.0001)
  ```shell script
  bash train_bpe_compact_adjust_mask_with_prob.sh 512 3 2 5 1 bert 512 0 4 mlm 0.15 0.5 0.0001 50
  ```
- (Model) Bert_bpe + (Masking Strategy) MTC (T=a.t.c, average token confidence)
  ```shell script
  bash train_bpe_compact_part_prob_auto_t.sh 512 3 2 5 1 bert 512 0 4 mlm 0.16 0.5 0.0 0.0 50
  ```
- (Model) Bert_bpe + (Masking Strategy) PMTC (T=a.t.c, p = 0.1)
  ```shell script
  bash train_bpe_compact_part_prob_auto_t.sh 512 3 2 5 1 bert 512 0 4 mlm 0.16 0.5 0.0 0.1 50
  ```
- (Model) Bert_bpe + (Masking Strategy) PMTC (T=a.t.c, p linear increase)
  ```shell script
  bash train_bpe_compact_part_prob_auto_t_linear_increase.sh 512 3 2 5 1 bert 512 0 4 mlm 0.16 0.5 0.0 50
  ```
The pre-trained model will be save to '../bert_model/' with a specific name.

## Run downstream probing tasks
After models are pre-trained, you can either extract fixed features from them or directly fine-tune them on serveal downstream tasks.

**Because the data needs to be kept confidential, we can only provide a small amount of encrypted data. We provide these data only to accurately demonstrate the experimental process and present model/training hyperparameters, not to reproduce the results discussed in our paper.**

Before running any downstream tasks, you have to specify the model name (e.g. Bert_bpe_uniform_MLM) in each script. You can do it by passing model names to the 'pretrain_models' argument.

To inspect the training process of downstream tasks, go to the downstream_tasks directory and run the following scripts.

- Bot detection task (use encoder as the feature extrator)
  ```shell script
  bash bot_detect_vary_train_size.sh
  ```
- Bot detection task (fine-tune sequence encoder)
  ```shell script
  bash bot_detect_vary_train_size_finetune.sh
  ```
- Churn prediction task (use encoder as the feature extrator)
  ```shell script
  bash churn_predict_vary_train_size.sh
  ```
- Churn prediction task (fine-tune sequence encoder)
  ```shell script
  bash churn_predict_vary_train_size_finetune.sh
  ```
- Purchase timing prediction task (use encoder as the feature extrator)
  ```shell script
  bash buy_time_predict.sh
  ```
- Purchase timing prediction task (fine-tune sequence encoder)
  ```shell script
  bash buy_time_predict_finetune.sh
  ```
- Similar player inducing task (use encoder as the feature extrator)
  ```shell script
  bash clustering.sh
  ```

## License

Our project is under the GPL License.
