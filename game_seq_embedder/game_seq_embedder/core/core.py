from .embedder import BehaviorSequenceEmbedder
from .bert_tokenizer_custom import WhiteSpaceTokenizer
from ..custom_models.bert_model import BertForMaskedLMTimeEmbed
from ..custom_models.bert_model_multi_task import BertForMaskedLMTimeEmbedMultiTask

from .config import BPE_TOKENIZER_PATH
from .config import BPE_GAME_ID_CN_CHAR_MAP_PATH

from .utils import load_save_json

from ..transformers import (AutoConfig, AutoModelWithLMHead)
from tokenizers import Tokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ipdb
import torch
import pandas as pd
import os
import glob


def _init_tokenizer(meta_config, model_dir, seperate_design_id, cn_char_map_path=None, tokenizer_path=None):
    if cn_char_map_path is None:
        cn_char_map_path = BPE_GAME_ID_CN_CHAR_MAP_PATH

    if tokenizer_path is None:
        tokenizer_path = BPE_TOKENIZER_PATH

    if meta_config['tokenizer'] == 'bpe':
        embed_tokenizer = Tokenizer.from_file(tokenizer_path)
        embed_tokenizer.cls_token_id = 0
        embed_tokenizer.pad_token_id = 1
        embed_tokenizer.sep_token_id = 2
        embed_tokenizer.unk_token_id = 3
        embed_tokenizer.mask_token_id = 4
        # tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
        # tokenizer.added_tokens_encoder = {}
        # tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
        # tokenizer.mask_token = '[MASK]'
        embed_tokenizer._pad_token = '[PAD]'
        embed_tokenizer.use_bpe = True
        embed_tokenizer.game_id_cn_char_map = load_save_json(cn_char_map_path, 'load', encoding='utf-8')
        embed_tokenizer.vocab = {'[PAD]': embed_tokenizer.pad_token_id}
        return embed_tokenizer
    elif meta_config['tokenizer'] == 'whitespace':
        if seperate_design_id:
            behave_vocab_path = glob.glob(os.path.join(model_dir, 'ws_behave_*.vocab'))[0]
            design_vocab_path = glob.glob(os.path.join(model_dir, 'ws_design_*.vocab'))[0]
            behave_tokenizer = WhiteSpaceTokenizer(vocab_path=behave_vocab_path)
            design_tokenizer = WhiteSpaceTokenizer(vocab_path=design_vocab_path)
            return behave_tokenizer, design_tokenizer
        else:
            vocab_paths = glob.glob(os.path.join(model_dir, '*.vocab'))
            # print(f"model_dir: {model_dir}, vocab_paths: {vocab_paths}")
            assert len(vocab_paths) == 1
            vocab_path = vocab_paths[0]
            embed_tokenizer = WhiteSpaceTokenizer(vocab_path=vocab_path)
            embed_tokenizer.use_bpe = False
            embed_tokenizer.resort_vocab()  # TODO, maybe remove in the future
            return embed_tokenizer
    else:
        raise NotImplemented


def _auto_load_model(model_bin_path, model_config):
    model = AutoModelWithLMHead.from_pretrained(
        model_bin_path,
        config=model_config,
    )
    print(f"Load model params from {model_bin_path}")
    return model


def _init_model(meta_config, model_bin_path, model_config, use_time_embed):
    if meta_config['model'] == 'bert':
        if use_time_embed:
            if 'task0_mlm_prob' in meta_config:
                model = BertForMaskedLMTimeEmbedMultiTask(model_config)
            else:
                model = BertForMaskedLMTimeEmbed(model_config)
            load_result = model.load_state_dict(torch.load(model_bin_path))
            print(load_result)
            print(f"Load model params from {model_bin_path}")
        else:
            model = _auto_load_model(model_bin_path, model_config)
    else:
        model = _auto_load_model(model_bin_path, model_config)
    model_name = model._get_name()
    print(f"Load model-{model_name} SUCCESS")
    return model


def init_model_and_tokenizer(model_dir, cn_char_map_path=None, tokenizer_path=None):
    model_config_path = os.path.join(model_dir, 'config.json')
    model_config = AutoConfig.from_pretrained(model_config_path)
    model_bin_path = os.path.join(model_dir, 'pytorch_model.bin')
    meta_config_path = os.path.join(model_dir, 'meta_config.json')
    meta_config = load_save_json(meta_config_path, 'load')
    print(f"Load meta config from {meta_config_path}, meta_config: {meta_config}")

    # use bpe
    if meta_config['tokenizer'] == 'bpe':
        use_bpe = True
    else:
        use_bpe = False

    # use time embed
    use_time_embed = meta_config['is_time_embed']
    use_sinusoidal = meta_config.get('use_sinusoidal', 0)
    seperate_design_id = meta_config.get('seperate_design_id', 0)
    use_sinusoidal = True if use_sinusoidal else False
    seperate_design_id = True if seperate_design_id else False
    if seperate_design_id:
        assert not use_bpe

    # init_tokenizer
    if seperate_design_id:
        behave_tokenizer, design_tokenizer = _init_tokenizer(meta_config,
                                                             model_dir,
                                                             seperate_design_id,
                                                             cn_char_map_path=cn_char_map_path,
                                                             tokenizer_path=tokenizer_path)
        tokenizer = (behave_tokenizer, design_tokenizer)
        embed_tokenizer = None
    else:
        embed_tokenizer = _init_tokenizer(meta_config, model_dir, False,
                                          cn_char_map_path=cn_char_map_path,
                                          tokenizer_path=tokenizer_path)
        tokenizer = (embed_tokenizer,)

    infer_with_prob = meta_config.get('infer_with_prob', False)
    if infer_with_prob:
        train_predict_prob_path = os.path.join(model_dir, 'train_predict_prob.csv')
        train_predict_prob_df = pd.read_csv(train_predict_prob_path)
        label_indices = train_predict_prob_df['label_index'].values
        probs = train_predict_prob_df['acc_prob'].values
        label_index_probs = dict(zip(label_indices, probs))
        assert embed_tokenizer is not None
        vocab_length = embed_tokenizer.get_vocab_size()
        assert len(probs) <= vocab_length
        token_weights = torch.zeros((vocab_length,))
        for label_index, prob in label_index_probs.items():
            token_weights[label_index] = prob
        print(
            f"[Embedder Token weights] Load token weights from {train_predict_prob_path}, shape: {token_weights.shape},"
            f" avg: {torch.mean(token_weights)}")
    else:
        token_weights = None

    model_name = model_config.architectures[0]
    model = _init_model(meta_config, model_bin_path, model_config, use_time_embed)
    print(model)

    return model, tokenizer, use_time_embed, use_bpe, use_sinusoidal, seperate_design_id, token_weights


def init_behavior_sequence_embedder(model_dir: str,
                                    is_finetune: bool = False,
                                    embedding_cache_dir: str = None,
                                    cn_char_map_path: str = None,
                                    tokenizer_path: str = None
                                    ):
    model, embed_tokenizer, use_time_embed, use_bpe, use_sinusoidal, seperate_design_id, token_weights \
        = init_model_and_tokenizer(model_dir, cn_char_map_path=cn_char_map_path, tokenizer_path=tokenizer_path)

    if len(embed_tokenizer) == 1:
        embed_tokenizer = embed_tokenizer[0]
        behave_tokenizer, design_tokenizer = None, None
    else:
        behave_tokenizer, design_tokenizer = embed_tokenizer
        embed_tokenizer = None

    # init embedder
    embedder = BehaviorSequenceEmbedder(embed_tokenizer, model,
                                        use_time_embed=use_time_embed,
                                        use_bpe=use_bpe,
                                        use_sinusoidal=use_sinusoidal,
                                        seperate_design_id=seperate_design_id,
                                        behave_tokenizer=behave_tokenizer,
                                        design_tokenizer=design_tokenizer,
                                        is_finetune=is_finetune,
                                        embedding_cache_dir=embedding_cache_dir,
                                        token_weights=token_weights)
    return embedder
