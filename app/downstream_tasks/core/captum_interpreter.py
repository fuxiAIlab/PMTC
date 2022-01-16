import os
import ipdb
import torch
import pickle
import numpy as np
import collections
from tqdm import tqdm
import pandas as pd

from captum.attr import IntegratedGradients
from captum.attr import visualization


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

class CaptumInterpreter:
    def __init__(self, base_dir, task):
        self.base_save_path = os.path.join(base_dir, 'captum_interpreter', task)
        if not os.path.isdir(self.base_save_path):
            os.makedirs(self.base_save_path)
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.log_id_trans_dict_path = os.path.join(project_dir, 'static', 'log_object_id_translation.json')
        self.log_id_trans_dict = load_save_json(self.log_id_trans_dict_path, 'load')
        self.model = None
        self.ig = None
        self.task = task

        # self.neg_attr = {'token': [], 'weights': []}
        self.pos_token_attr = collections.defaultdict(lambda: [])
        self.neg_token_attr = collections.defaultdict(lambda: [])
        self.pos_position_attr = collections.defaultdict(lambda: [])
        self.neg_position_attr = collections.defaultdict(lambda: [])
        self.deltas = []
        self.vis_data_records = []

    def load_test_model(self, model):
        model.eval()
        self.ig = IntegratedGradients(model)
        self.model = model
        print(f"[Captum Interpreter] load model success!")

    def compute_attribute(self, test_samples, test_labels, test_bpe_seqs, batch_size=4):
        steps = len(test_samples) // batch_size
        attributions = []
        model_output_probs = []
        for step in tqdm(range(steps), total=steps):
            # all_zero_samples = torch.zeros_like(test_samples[0:2]).to('cuda')
            # all_zero_results = self.model(all_zero_samples) # the output is around 0.45 after training 1 epoch
            step_samples = test_samples[step * batch_size: (step + 1) * batch_size].to(self.model.device)
            model_output_prob = self.model(step_samples)
            model_output_probs.append(model_output_prob)
            step_attributions, step_delta = self.ig.attribute(
                step_samples,
                baselines=None,
                return_convergence_delta=True)
            attributions.append(step_attributions.to('cpu'))  # batch_size x max_seq_length
            self.deltas.extend(step_delta.tolist())
        attributions = torch.cat(attributions)
        model_output_probs = torch.cat(model_output_probs)

        assert attributions.shape[0] == model_output_probs.shape[0]

        # compute the sum attribution of the last dimension
        attributions = torch.sum(attributions, dim=2)

        for i, attribution in enumerate(attributions):
            attribution = attribution[attribution != 0.0]
            attribution = attribution / torch.norm(attribution)
            attr_score = float(attribution.sum())
            convergence_delta = self.deltas[i]
            label = int(test_labels[i])
            attr_label = 1
            bpe_seq = test_bpe_seqs[i]
            bpes_seq = [str(x) for x in bpe_seq]
            predict_prob = float(model_output_probs[i])
            predict_label = 0 if predict_prob < 0.5 else 1

            # update vis_data_records
            self.vis_data_records.append(visualization.VisualizationDataRecord(
                attribution,
                predict_prob,
                predict_label,
                label,
                attr_label,
                attr_score,
                bpes_seq,
                convergence_delta))

            for pos_i, (x, x_attr) in enumerate(zip(bpe_seq, attribution)):
                if label == 0:
                    self.neg_token_attr[tuple(x)].append(float(x_attr))
                    self.neg_position_attr[len(attribution) - pos_i].append(float(x_attr))
                elif label == 1:
                    self.pos_token_attr[tuple(x)].append(float(x_attr))
                    self.pos_position_attr[len(attribution) - pos_i].append(float(x_attr))

    def save_all_to_disk(self):
        to_save_dicts = [(self.neg_token_attr, 'negative_label_token.csv'),
                         (self.neg_position_attr, 'negative_label_pos.csv'),
                         (self.pos_token_attr, 'positive_label_token.csv'),
                         (self.pos_position_attr, 'positive_label_pos.csv')]
        for tmp_dict, save_name in to_save_dicts:
            tmp_dict = {k: np.mean(v) for k, v in tmp_dict.items()}
            save_df = pd.DataFrame(tmp_dict.items())

            if save_name.startswith('negative'):
                save_df = save_df.sort_values(1, ascending=True)
            elif save_name.startswith('positive'):
                save_df = save_df.sort_values(1, ascending=False)
            else:
                raise Exception

            save_path = os.path.join(self.base_save_path, save_name)
            if save_name.split('.')[0].endswith('pos'):
                save_df.columns = ['pos', 'attr']
            elif save_name.split('.')[0].endswith('token'):
                save_df.columns = ['token', 'attr']
                trans_log_ids = []
                for i, row in save_df.iterrows():
                    token = row['token']
                    trans_log_id = [self.log_id_trans_dict.get(x, '') for x in token]
                    trans_log_ids.append(trans_log_id)
                save_df['trans_log_id'] = trans_log_ids
                save_df = save_df[['token', 'attr', 'trans_log_id']]
            else:
                raise Exception
            save_df.to_csv(save_path, index=False)
            print(f"Save df to {save_path}")

        # save deltas
        deltas_save_path = os.path.join(self.base_save_path, 'deltas.txt')
        with open(deltas_save_path, 'w') as f:
            for x in self.deltas:
                f.write(str(x) + '\n')
        print(f"Save deltas to {deltas_save_path}")

        # save vis_data_records
        vis_data_records_save_path = os.path.join(self.base_save_path, 'captum_vis_records.pkl')
        pickle.dump(self.vis_data_records, open(vis_data_records_save_path, 'wb'))
        print(f"Save captum vis record to {vis_data_records_save_path}")

        # visualization.visualize_text(random_vis_data_records_ig)
