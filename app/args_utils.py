from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_white_space_tokenzier: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to use white space tokenzier"},
    )
    use_bpe_tokenzier: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to use bpe tokenzier"},
    )
    bpe_tokenizer_path: Optional[str] = field(
        default=None,
    )
    remove_cache: Optional[int] = field(
        default=1,
        metadata={"help": "Whether to remove cached features"},
    )
    use_time_embed: Optional[int] = field(
        default=1,
        metadata={"help": "Whether to use time embedding"},
    )
    max_time_gap: Optional[int] = field(
        default=1024,
        metadata={"help": "Dimension for time embedding"},
    )
    debugN: Optional[int] = field(
        default=None,
        metadata={"help": "debugN"},
    )
    max_token_vocab_size: Optional[int] = field(
        default=None,
        metadata={"help": "max token vocab size"},
    )
    add_mean_pool_to_train: int = field(
        default=0
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    h5_data_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input h5 data file."}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pretrain_task: str = field(
        default=None, metadata={"help": "The name of the pre-train task, valid ones include: mlm, clm, ..."}
    )
    clm_sample_n: int = field(
        default=None, metadata={"help": "The number of CLM training samples to create from the original sample"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    outer_epoch: int = field(
        default=1
    )
    large_batch: int = field(
        default=1024
    )
    seperate_design_id: int = field(
        default=0
    )
    use_sinusoidal: int = field(
        default=0
    )
    is_gate_design_id: int = field(
        default=0
    )
    is_gate_timestamp: int = field(
        default=0
    )
    is_multi_task: int = field(
        default=0
    )
    is_task0: int = field(
        default=1,
        metadata={"help": "Input: masked log_id, design_id, timestamps, Ouput: log_id, Task: MLM"}
    )
    is_task1: int = field(
        default=0,
        metadata={"help": "Input: masked log_id, Ouput: log_id, Task: MLM"}
    )
    is_task2: int = field(
        default=0,
        metadata={
            "help": "Input: log_id, masked design_id, masked timestamps,"
                    " Ouput: design_id, timstamp, Task: MLM + regression"}
    )
    is_task3: int = field(
        default=0,
        metadata={"help": "Input: masked timestamp, Ouput: timestamp, Task: Regression"}
    )
    is_task4: int = field(
        default=0,
        metadata={"help": "Input: masked design_id, Ouput: design_id, Task: "}
    )
    task0_mlm_prob: float = field(
        default=0.15
    )
    task1_mlm_prob: float = field(
        default=0.15
    )
    task2_mlm_prob: float = field(
        default=0.15
    )
    task3_mlm_prob: float = field(
        default=0.15
    )
    task4_mlm_prob: float = field(
        default=0.15
    )
    is_curriculum_task: int = field(
        default=0,
        metadata={"help": "Whether to use Curriculum Learning for task"}
    )
    task_curriculum_update_step: int = field(
        default=0,
    )
    task_curriculum_alpha: float = field(
        default=2.0
    )
    is_curriculum_within_task: int = field(
        default=0,
        metadata={"help": "Whether to use Curriculum Learning for each single task"}
    )
    is_timestamp_rnn: int = field(
        default=0
    )
    bpe_cr_update_coeff: float = field(
        default=1.01
    )
    bpe_cr_adjust_tol: float = field(
        default=0.0
    )
    use_random_mlm_probability: int = field(
        default=0
    )
    mlm_prob_min: float = field(
        default=0.15
    )
    mlm_prob_max: float = field(
        default=0.5
    )
    is_anti_curr: int = field(
        default=0
    )
    dynamic_mask_update_alpha: float = field(
        default=0.5,
        metadata={"help": "Controls the speed of updating prob, "
                          "if close to 1.0, only the latest predicted sample will affect the prob"}
    )
    is_train_time_freq_adapter: int = field(
        default=0
    )
    mask_type: str = field(
        default='normal'
    )
    mask_softmax_t: float = field(
        default=0.5
    )
    idf_path: str = field(
        default=''
    )
    tf_idf_warmup_decay: float = field(
        default=None
    )
    is_record_snapshot: int = field(
        default=0
    )
    snapshot_gap: int = field(
        default=1000
    )
    shuffle_epoch: int = field(
        default=0
    )
    is_record_mask_ratio: int = field(
        default=0
    )
    softmax_t_decay_mode: str = field(
        default=None
    )
    part_prob_percent: float = field(
        default=None
    )
    part_prob_min: float = field(
        default=None
    )
    part_prob_max: float = field(
        default=None
    )