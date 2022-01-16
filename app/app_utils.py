import os
import json
import ipdb
import math
import shutil
import hashlib
import time
import copy
import h5py
import ntpath
import random
import collections
import numpy as np
import torch

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def set_randseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding=encoding) as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def get_all_indices(h5_data_file_path, debug_N, is_print_pad_ratio=False, is_shuffle=False):
    hdf5_file = h5py.File(h5_data_file_path, 'r')
    all_indices = []
    all_keys = sorted(hdf5_file.keys())
    total_num = 0
    for key in all_keys:
        data = hdf5_file[key]

        if is_print_pad_ratio:
            data_arr = np.array(data)
            data_arr = data_arr[data_arr == b'[PAD]']
            pad_ratio = len(data_arr) / (data.shape[0] * data.shape[1])
            print(f"Key: {key}, Pad ratio: {pad_ratio}")

        shape = data.shape
        total_num += shape[0]
        all_indices.extend([f'{key}_{x}' for x in range(shape[0])])

    if debug_N:
        random.shuffle(all_indices)
        all_indices, total_num = all_indices[:debug_N], debug_N
    else:
        if is_shuffle:
            random.shuffle(all_indices)

    return all_indices, total_num


def load_cn_char_arr_creator(cn_char_npy_dir, read_npy_from_disk):
    def load_cn_char_arr(data_with_byte_arr):
        if read_npy_from_disk:
            line_hash = hashlib.md5(data_with_byte_arr).hexdigest()
            cnchar_file_path = os.path.join(cn_char_npy_dir, f'{line_hash}.npy')
            if os.path.isfile(cnchar_file_path):
                temp_line = np.load(cnchar_file_path)
            else:
                temp_line = [x.decode('utf-8') for x in data_with_byte_arr]
                np.save(cnchar_file_path, np.array(temp_line))
        else:
            temp_line = [x.decode('utf-8') for x in data_with_byte_arr]
        return temp_line

    return load_cn_char_arr


def check_batch_hdf5_dataset_size(hdf5_file_path,
                                  all_indices,
                                  step_size,
                                  large_batch=1024):
    cp_all_indices = copy.deepcopy(all_indices)
    random.shuffle(cp_all_indices)
    for step_i in range(step_size):
        t1 = time.time()
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        # 'nsh_2020-03-01'
        # hdf5_file['nsh_2020-03-01']
        next_indices = cp_all_indices[step_i * large_batch:(step_i + 1) * large_batch]
        yield len(next_indices)


def hdf5_load_dataset(hdf5_file_path, all_indices, step_size,
                      large_batch=1024,
                      is_decode_utf8=False,
                      cn_char_npy_dir='../data/cn_char_npy_files',
                      read_npy_from_disk=False):
    # 这里的shuffle在数据量大的时候有用，保证每个epoch的数据不一样（每个日期都读一点），
    # 但是当一次性一个large_batch_data读完整个数据集的时候就没有用了，
    # 所以结束的时候再shuffle一下

    random.shuffle(all_indices)

    # init thread pool
    thread_pool = ThreadPoolExecutor(max_workers=6)

    for step_i in range(step_size):
        t1 = time.time()

        hdf5_file = h5py.File(hdf5_file_path, 'r')
        # 'nsh_2020-03-01'
        # hdf5_file['nsh_2020-03-01']
        next_indices = all_indices[step_i * large_batch:(step_i + 1) * large_batch]
        next_data = collections.defaultdict(lambda: [])
        for x in next_indices:
            *dataset_name, index = x.split('_')
            dataset_name = '_'.join(dataset_name)
            next_data[dataset_name].append(int(index))
        large_batch_data = []

        print(f"Load dataset from hdf5 step {step_i}, size next indices: {len(next_indices)}")
        for dataset_name, dataset_indices in next_data.items():

            if dataset_name == 'nsh_2020-04-04':
                print(f"Skip for {dataset_name}")
                continue

            if is_decode_utf8:
                temp_indices_data = hdf5_file[dataset_name][sorted(dataset_indices)]
                temp_indices_data_str = list(
                    thread_pool.map(load_cn_char_arr_creator(cn_char_npy_dir, read_npy_from_disk), temp_indices_data))
                # temp_indices_data_str = []
                # for i, temp_line in enumerate(temp_indices_data):
                #     line_hash = hashlib.md5(temp_line).hexdigest()
                #     cnchar_file_path = os.path.join(cn_char_npy_dir, f'{line_hash}.npy')
                #     if os.path.isfile(cnchar_file_path):
                #         temp_line = np.load(cnchar_file_path)
                #     else:
                #         temp_line = [x.decode('utf-8') for x in temp_line]
                #         np.save(cnchar_file_path, np.array(temp_line))
                #     temp_indices_data_str.append(temp_line)

                temp_indices_data_str = np.stack(temp_indices_data_str)
                large_batch_data.append(temp_indices_data_str)
            else:
                large_batch_data.append(hdf5_file[dataset_name][sorted(dataset_indices)])
            print(f"Read from {dataset_name} done, size: {len(dataset_indices)}")

        large_batch_data = np.concatenate(large_batch_data).astype(str)

        hdf5_file.close()

        large_batch_indices = list(range(len(large_batch_data)))
        random.shuffle(large_batch_indices)
        large_batch_data = large_batch_data[large_batch_indices]

        t2 = time.time()
        # Read Data step_i-0, Total time: 2.59651517868042
        print(f"Read Data step_i-{step_i}, Total data load time: {t2 - t1}")
        yield large_batch_data


def load_dataset_from_hdf5_by_indices(hdf5_file_path, indices, is_decode_utf8=False):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    next_data = collections.defaultdict(lambda: [])
    for x in indices:
        *dataset_name, index = x.split('_')
        dataset_name = '_'.join(dataset_name)
        next_data[dataset_name].append(int(index))
    large_batch_data = []
    for dataset_name, dataset_indices in next_data.items():
        if is_decode_utf8:
            temp_indices_data = hdf5_file[dataset_name][sorted(dataset_indices)]
            temp_indices_data_str = []
            for i, temp_line in enumerate(temp_indices_data):
                temp_line = [x.decode('utf-8') for x in temp_line]
                temp_indices_data_str.append(temp_line)
            temp_indices_data_str = np.stack(temp_indices_data_str)
            large_batch_data.append(temp_indices_data_str)
        else:
            large_batch_data.append(hdf5_file[dataset_name][sorted(dataset_indices)])
    large_batch_data = np.concatenate(large_batch_data).astype(str)
    hdf5_file.close()
    return large_batch_data


def _is_init_list_consecutive(int_list):
    return sorted(int_list) == list(range(min(int_list), max(int_list) + 1))


def change_trainer_output_dir(origin_output_dir, model_args, logger, data_args):
    output_name = ntpath.basename(origin_output_dir)
    output_base_dir = os.path.dirname(origin_output_dir)
    output_name += f'_{model_args.model_type}'
    if model_args.use_time_embed:
        if data_args.use_sinusoidal:
            output_name += '_time_embed_sin'
        else:
            output_name += '_time_embed'
    else:
        output_name += '_NO_time_embed'

    if model_args.use_white_space_tokenzier:
        output_name += '_whitespace'
    else:
        output_name += '_BPE'

    if data_args.seperate_design_id:
        output_name += '_sep_des_id'

    if data_args.is_gate_design_id:
        output_name += f'_design_gate'
    if data_args.is_gate_timestamp:
        output_name += f'_time_gate'
    if model_args.add_mean_pool_to_train:
        output_name += f'_train_mean_pool'

    if model_args.model_type not in {'reformer', 'longformer'}:
        if data_args.is_task0:
            output_name += f'_t0'
        if data_args.is_task1:
            output_name += f'_t1'
        if data_args.is_task2:
            output_name += f'_t2'
        if data_args.is_task3:
            output_name += f'_t3'
        if data_args.is_task4:
            output_name += f'_t4'

        if data_args.is_curriculum_task:
            output_name += f'_curr_task'
            if data_args.task_curriculum_update_step:
                output_name += f'_step_{data_args.task_curriculum_update_step}'
            if data_args.task_curriculum_alpha:
                output_name += f'_alpha_{data_args.task_curriculum_alpha}'

        if data_args.is_curriculum_within_task:
            output_name += f'_curr_mask'
            if data_args.task_curriculum_update_step:
                output_name += f'_step_{data_args.task_curriculum_update_step}'

            if data_args.bpe_cr_update_coeff:
                output_name += f'_{data_args.bpe_cr_update_coeff}'
            if data_args.bpe_cr_adjust_tol:
                output_name += f'_{data_args.bpe_cr_adjust_tol}'

        if data_args.pretrain_task:
            output_name += f'_{data_args.pretrain_task}'

    output_name += f'_mlen_{data_args.block_size}'

    if model_args.model_type not in {'reformer', 'longformer'}:
        if data_args.use_random_mlm_probability:
            output_name += f'_random_mlm_{data_args.mlm_prob_min}_{data_args.mlm_prob_max}'
        elif data_args.is_anti_curr:
            output_name += f'_anti_curr_mlm_{data_args.mlm_prob_min}_{data_args.mlm_prob_max}'
        else:
            if data_args.mlm_probability:
                output_name += f'_mlmprob-{data_args.mlm_probability}'

    if data_args.mask_type:
        output_name += f'_mask_type-{data_args.mask_type}'

    if data_args.mask_softmax_t and data_args.softmax_t_decay_mode is None:
        output_name += f'_t-{data_args.mask_softmax_t}'

    if data_args.dynamic_mask_update_alpha:
        output_name += f'_alpha-{data_args.dynamic_mask_update_alpha}'

    if data_args.tf_idf_warmup_decay:
        output_name += f'_decay-{data_args.tf_idf_warmup_decay}'

    if data_args.is_record_snapshot:
        output_name += f'_record_snapshot'

    if data_args.softmax_t_decay_mode:
        output_name += f'_decay-{data_args.softmax_t_decay_mode}'

    if data_args.part_prob_percent is not None:
        output_name += f'_part_percent-{data_args.part_prob_percent}'

    if model_args.debugN:
        output_name += f'_DEBUG{model_args.debugN}'

    new_output_dir = os.path.join(output_base_dir, output_name)
    logger.info(f"Set new training save dir {new_output_dir} !!!!!!!!!!!!!!!!!!!!!")
    if not os.path.isdir(new_output_dir):
        os.makedirs(new_output_dir)
        logger.info(f"Make new dir {new_output_dir}")

    return new_output_dir


def load_save_white_space_tokenizer_vocab(tokenizer,
                                          h5_file_path,
                                          local_rank,
                                          model_save_dir,
                                          vocab_base_dir='../bert_model/vocab_hashes',
                                          debug_N=None,
                                          max_token_vocab_size=None,
                                          max_read_file=5,
                                          mode='all'
                                          ):
    assert mode in {'all', 'behave', 'design'}

    hdf5_file = h5py.File(h5_file_path, 'r')
    hdf5_file_all_shapes = []
    for data in hdf5_file.values():
        hdf5_file_all_shapes.extend([str(x) for x in data.shape])

    hash_list = sorted(list(hdf5_file.keys())) + [ntpath.basename(h5_file_path)] + hdf5_file_all_shapes + [
        str(debug_N)] + [str(max_read_file)]

    if mode in {'behave', 'design'}:
        hash_list = hash_list + [mode]

    ws_tokenzier_hash_value = hash(tuple(sorted(hash_list)))

    print(f"ws_tokenzier_hash_value: {ws_tokenzier_hash_value}")
    if not os.path.isdir(vocab_base_dir):
        if local_rank in {-1, 0}:
            os.makedirs(vocab_base_dir)
            print(f"Make new dir {vocab_base_dir} done")

    if mode != 'all':
        ws_tokenzier_vocab_path = os.path.join(vocab_base_dir, f'ws_{mode}_{ws_tokenzier_hash_value}.vocab')
        vocab_path_in_model_dir = os.path.join(model_save_dir, f'ws_{mode}_{ws_tokenzier_hash_value}.vocab')
    else:
        ws_tokenzier_vocab_path = os.path.join(vocab_base_dir, f'ws_{ws_tokenzier_hash_value}.vocab')
        vocab_path_in_model_dir = os.path.join(model_save_dir, f'ws_{ws_tokenzier_hash_value}.vocab')

    if local_rank in {-1, 0}:
        if os.path.isfile(ws_tokenzier_vocab_path):
            tokenizer.vocab = {**load_save_json(ws_tokenzier_vocab_path, 'load'), **tokenizer.vocab}
            print(f"Load White Space Tokenizer from {ws_tokenzier_vocab_path}")
        else:
            for file_i, (file_name, data) in enumerate(hdf5_file.items()):
                print(f"Building vocab from {file_name} ...")
                data_tqdm = tqdm(data, total=len(data))
                for data_i, line in enumerate(data_tqdm):
                    if debug_N:
                        if data_i >= debug_N:
                            print(f"Trigger early stopping for {file_name} because debug_N {debug_N}")
                            break
                    line = line.astype('str')
                    line = [x for x in line if x != '[PAD]']
                    no_time_gap_line = [x for i, x in enumerate(line) if (i + 1) % 3 != 0]

                    if mode == 'all':
                        tokenizer.add_vocab_from_list(no_time_gap_line)
                    elif mode == 'behave':
                        tokenizer.add_vocab_from_list(no_time_gap_line[::2])
                    elif mode == 'design':
                        tokenizer.add_vocab_from_list(no_time_gap_line[1::2])

                    data_tqdm.set_description(f"Total vocab size :{len(tokenizer.vocab)}")

                if debug_N:
                    break

                if file_i >= max_read_file:
                    print("Reach Num max_read_file, break")
                    break
            tokenizer.squeeze_vocab_by_freq(max_token_vocab_size)
            print(f"[WHITE SPACE VOCAB] Load vocab done, total: {len(tokenizer.vocab)}, local_rank: {local_rank}")
            load_save_json(ws_tokenzier_vocab_path, 'save', data=tokenizer.vocab)
    else:
        while not os.path.isfile(ws_tokenzier_vocab_path):
            time.sleep(5)
            print(f"[WHITE SPACE VOCAB] vocab path {ws_tokenzier_vocab_path} not found, local_rank: {local_rank}")
        tokenizer.vocab = {**load_save_json(ws_tokenzier_vocab_path, 'load'), **tokenizer.vocab}

    hdf5_file.close()
    tokenizer.resort_vocab()
    vocab_values = sorted(tokenizer.vocab.values())
    assert vocab_values[0] == 0
    assert vocab_values[-1] == min(len(tokenizer.vocab), max_token_vocab_size) - 1
    assert _is_init_list_consecutive(vocab_values)

    # move vocab to
    if local_rank in {-1, 0}:
        shutil.copy(ws_tokenzier_vocab_path, vocab_path_in_model_dir)
        print(f"Copy vocab path from {ws_tokenzier_vocab_path} to {vocab_path_in_model_dir}")


def create_ws_tokenizer(data_args, training_args, model_args, new_output_dir, mode):
    from game_embedder.bert_tokenizer_custom import WhiteSpaceTokenizer
    from game_tokenizer.game_tokenizer import create_func1, create_func2
    tokenizer = WhiteSpaceTokenizer()
    tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
    tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)

    if mode == 'design':
        max_token_vocab_size = model_args.max_token_vocab_size // 4
    else:
        max_token_vocab_size = model_args.max_token_vocab_size

    load_save_white_space_tokenizer_vocab(tokenizer,
                                          data_args.h5_data_file_path,
                                          training_args.local_rank,
                                          new_output_dir,
                                          debug_N=model_args.debugN,
                                          max_token_vocab_size=max_token_vocab_size,
                                          mode=mode)
    tokenizer.max_len = tokenizer.max_vocab_index  # max vocab index
    print(
        f"Create vocab for mode-{mode} tokenzier done! Vocab size: {tokenizer.max_len}, Max vocab size: {max_token_vocab_size}")
    return tokenizer


def get_input_for_task(inputs):
    new_inputs = {}
    new_inputs['input_ids'] = inputs[0]
    new_inputs['labels'] = inputs[1]

    assert new_inputs['input_ids'].shape[1] % 3 == 0
    cut_index1 = int(new_inputs['input_ids'].shape[1] / 3)
    cut_index2 = cut_index1 * 2
    behave_ids = new_inputs['input_ids'][:, :cut_index1]
    design_ids = new_inputs['input_ids'][:, cut_index1:cut_index2]
    timestamps = new_inputs['input_ids'][:, cut_index2:]

    new_inputs['time_gaps'] = timestamps
    new_inputs['input_ids'] = behave_ids
    new_inputs['design_ids'] = design_ids

    if isinstance(new_inputs['labels'], tuple):
        assert new_inputs['labels'][0].shape == new_inputs['labels'][0].shape == \
               behave_ids.shape == design_ids.shape == timestamps.shape
    else:
        assert new_inputs['labels'].shape == behave_ids.shape == design_ids.shape == timestamps.shape
    return new_inputs


def get_target_loss(task_name, inputs, training_step_func, model, is_loss_backward=True):
    task0_inputs = inputs[task_name]
    inputs = get_input_for_task(task0_inputs)

    if task_name in {'task0', 'task1'}:
        inputs['output_log_id'] = True
        inputs['output_design_id'] = False
        inputs['output_timestamp'] = False
    elif task_name == 'task2':
        inputs['output_log_id'] = False
        inputs['output_design_id'] = True
        inputs['output_timestamp'] = True
    elif task_name == 'task3':
        inputs['output_log_id'] = False
        inputs['output_design_id'] = False
        inputs['output_timestamp'] = True
    elif task_name == 'task4':
        inputs['output_log_id'] = False
        inputs['output_design_id'] = True
        inputs['output_timestamp'] = False

    task_loss = training_step_func(model, inputs, is_loss_backward=is_loss_backward)
    return task_loss
