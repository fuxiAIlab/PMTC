import math
import ipdb
import torch
import warnings
from torch import nn
from transformers.modeling_bert import add_start_docstrings, add_start_docstrings_to_callable, \
    add_code_sample_docstrings
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import logger
from transformers.modeling_bert import MaskedLMOutput
from transformers.modeling_bert import CrossEntropyLoss
from transformers.modeling_bert import MSELoss
from transformers.modeling_bert import BERT_START_DOCSTRING
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers.modeling_bert import _TOKENIZER_FOR_DOC
from transformers.modeling_bert import _CONFIG_FOR_DOC
from transformers.modeling_bert import BertPredictionHeadTransform

from .bert_model import BertModelTimeEmbed

BertLayerNorm = torch.nn.LayerNorm


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.config = config

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.is_task0 or config.is_task1:
            self.behave_id_decoder = nn.Linear(config.hidden_size, config.behave_vocab_size, bias=False)
            self.behave_id_decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.behave_id_decoder = None

        if config.is_task4 or config.is_task2:
            self.design_id_decoder = nn.Linear(config.hidden_size, config.design_vocab_size, bias=False)
            self.design_id_decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.design_id_decoder = None

        if config.is_task3 or config.is_task2:
            self.timestamp_decoder = nn.Linear(config.hidden_size, 1, bias=True)

        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        # self.decoder.bias = self.bias

    def forward(self, hidden_states,
                output_log_id=True,
                output_design_id=False,
                output_timestamp=False):
        hidden_states = self.transform(hidden_states)
        forward_dict = {}
        if output_log_id:
            forward_dict['behave_id_hidden'] = self.behave_id_decoder(hidden_states)
        if output_design_id:
            forward_dict['design_id_hidden'] = self.design_id_decoder(hidden_states)
        if output_timestamp:
            forward_dict['timestamp_hidden'] = self.timestamp_decoder(hidden_states)
        return forward_dict


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output,
                output_log_id=True,
                output_design_id=False,
                output_timestamp=False):
        prediction_dict = self.predictions(sequence_output,
                                           output_log_id=output_log_id,
                                           output_design_id=output_design_id,
                                           output_timestamp=output_timestamp
                                           )
        return prediction_dict


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLMTimeEmbedMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModelTimeEmbed(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """

        if self.cls.predictions.behave_id_decoder is not None:
            self._tie_or_clone_weights(self.cls.predictions.behave_id_decoder,
                                       self.bert.embeddings.behave_word_embeddings)
            print("Tie weights for behave input/ouput embedding")

        if self.cls.predictions.design_id_decoder is not None:
            self._tie_or_clone_weights(self.cls.predictions.design_id_decoder,
                                       self.bert.embeddings.design_word_embeddings)
            print("Tie weights for design input/ouput embedding")

        #
        # output_embeddings = self.get_output_embeddings()
        # if output_embeddings is not None and self.config.tie_word_embeddings:
        #     self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            design_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            time_gaps=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            output_log_id=True,
            output_design_id=False,
            output_timestamp=False,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        assert labels is not None
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_ids.shape == time_gaps.shape

        outputs = self.bert(
            input_ids,
            design_ids=design_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            time_gaps=time_gaps,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_dict = self.cls(sequence_output,
                                   output_log_id=output_log_id,
                                   output_design_id=output_design_id,
                                   output_timestamp=output_timestamp)

        if isinstance(labels, tuple):
            assert output_design_id and output_timestamp
            design_id_labels, timestamp_labels = labels
        else:
            if output_log_id:
                assert not output_design_id
                assert not output_timestamp
                behave_id_labels = labels
            elif output_design_id:
                assert not output_log_id
                assert not output_timestamp
                design_id_labels = labels
            elif output_timestamp:
                assert not output_log_id
                assert not output_design_id
                timestamp_labels = labels

        total_loss = torch.tensor(0.0).to(self.device)
        if 'behave_id_hidden' in prediction_dict:
            log_id_lm_loss = CrossEntropyLoss()(
                prediction_dict['behave_id_hidden'].view(-1, self.config.behave_vocab_size),
                behave_id_labels.view(-1))
            total_loss += log_id_lm_loss

        if 'design_id_hidden' in prediction_dict:
            design_id_lm_loss = CrossEntropyLoss()(
                prediction_dict['design_id_hidden'].view(-1, self.config.design_vocab_size),
                design_id_labels.view(-1))
            total_loss += design_id_lm_loss

        if 'timestamp_hidden' in prediction_dict:
            timestamp_gt_labels = timestamp_labels.view(-1)
            keep_indices = torch.where(timestamp_gt_labels != -100)
            timestamp_gt_labels = timestamp_gt_labels[keep_indices].float()
            timestamp_predict_labels = prediction_dict['timestamp_hidden'].view(-1)[keep_indices]
            timestamp_loss = MSELoss()(timestamp_predict_labels,
                                       timestamp_gt_labels)
            total_loss += timestamp_loss

        output = (prediction_dict,) + outputs[2:]
        return ((total_loss,) + output)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
