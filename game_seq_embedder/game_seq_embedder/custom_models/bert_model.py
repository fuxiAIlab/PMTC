import math
import torch
import warnings

from ..transformers.modeling_bert import add_start_docstrings, add_start_docstrings_to_callable, \
    add_code_sample_docstrings
from ..transformers.modeling_bert import BertOnlyMLMHead
from ..transformers.modeling_bert import BertPreTrainedModel
from ..transformers.modeling_bert import logger
from ..transformers.modeling_bert import MaskedLMOutput
from ..transformers.modeling_bert import CrossEntropyLoss
from ..transformers.modeling_bert import BERT_START_DOCSTRING
from ..transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from ..transformers.modeling_bert import _TOKENIZER_FOR_DOC
from ..transformers.modeling_bert import _CONFIG_FOR_DOC

# from transformers.modeling_bert import BertEmbeddings
from ..transformers.modeling_bert import BertEncoder
from ..transformers.modeling_bert import BertPooler
from ..transformers.modeling_bert import BaseModelOutputWithPooling

from torch import nn

BertLayerNorm = torch.nn.LayerNorm


def positionalencoding1d_batch(seq_len, hidden_size):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    pe = torch.zeros(seq_len, hidden_size)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, hidden_size, 2, dtype=torch.float) *
                          -(math.log(10000.0) / hidden_size)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def positionalencoding1d(length, d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))

    before_sin = position.float() * div_term
    tempv = div_term[-1] * 86400
    print(f"tempv: {tempv}")

    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class BertEmbeddingsTimeEmbed(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        if hasattr(config, 'use_sinusoidal'):
            self.use_sinusoidal = config.use_sinusoidal
        else:
            self.use_sinusoidal = False

        if hasattr(config, 'seperate_design_id'):
            self.seperate_design_id = config.seperate_design_id
        else:
            self.seperate_design_id = False

        # gate for design_id
        if hasattr(config, 'is_gate_design_id'):
            assert self.seperate_design_id
            self.is_gate_design_id = True
            self.design_id_gate_w = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.is_gate_design_id = False
            self.design_id_gate_w = None

        # gate for timestamp
        if hasattr(config, 'is_gate_timestamp'):
            assert self.seperate_design_id
            self.is_gate_timestamp = True
            self.timestamp_gate_w = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.is_gate_timestamp = False
            self.timestamp_gate_w = None

        self.hidden_size = config.hidden_size

        if self.seperate_design_id:
            self.behave_word_embeddings = nn.Embedding(config.behave_vocab_size, config.hidden_size,
                                                       padding_idx=config.pad_token_id)
            self.word_embeddings = self.behave_word_embeddings
            self.design_word_embeddings = nn.Embedding(config.design_vocab_size, config.hidden_size,
                                                       padding_idx=config.pad_token_id)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        if self.use_sinusoidal:
            self.time_gap_embeddings = None
            self.sinusoidal_time_embeddings = positionalencoding1d_batch(86400, self.hidden_size)
        else:
            self.time_gap_embeddings = nn.Embedding(config.time_gap_max_size, config.hidden_size)
            self.sinusoidal_time_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, time_gaps=None,
                design_ids=None):

        device = self.position_ids.device

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            if self.seperate_design_id:
                log_id_embeddings = self.word_embeddings(input_ids)
                design_id_embeddings = self.design_word_embeddings(design_ids)
                if self.is_gate_design_id:
                    design_id_gate = torch.sigmoid(
                        self.design_id_gate_w(log_id_embeddings))
                    design_id_embeddings = design_id_gate * design_id_embeddings

                inputs_embeds = log_id_embeddings + design_id_embeddings
            else:
                inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.use_sinusoidal:
            time_gap_embeddings = self.sinusoidal_time_embeddings[time_gaps].to(device)
        else:
            # batch fancy indexing
            time_gap_embeddings = self.time_gap_embeddings(time_gaps).to(device)  # time_gaps: batch_size x seq_length

        if self.seperate_design_id:
            if self.is_gate_timestamp:
                timestamp_gate = torch.sigmoid(self.timestamp_gate_w(log_id_embeddings))
                time_gap_embeddings = timestamp_gate * time_gap_embeddings

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + time_gap_embeddings
        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelTimeEmbed(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddingsTimeEmbed(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
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
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            time_gaps=time_gaps,
            design_ids=design_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLMTimeEmbed(BertPreTrainedModel):
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
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
