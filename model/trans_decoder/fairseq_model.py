
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.models import FairseqDecoder, FairseqEncoder

class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        seen = set()

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_') \
                    and module not in seen:
                seen.add(module)
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode=True):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_') \
                    and module not in seen:
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(cls, parser, *inputs, model_name_or_path, data_name_or_path, **kwargs):
        """
        Instantiate a FairseqModel from a pre-trained model file or pytorch state dict.
        Downloads and caches the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either
                - a str with the name of a pre-trained model to load
                - a path or url to a pretrained model state dict
        """
        from fairseq import checkpoint_utils, file_utils, options, tasks

        model_path = file_utils.load_archive_file(model_name_or_path)
        data_path = file_utils.load_archive_file(data_name_or_path)
        checkpoint_path = os.path.join(model_path, 'model.pt')

        # set data and parse
        model_args = options.parse_args_and_arch(parser, input_args=[data_path])

        # override any kwargs passed in
        if kwargs is not None:
            for arg_name, arg_val in kwargs.items():
                setattr(model_args, arg_name, arg_val)

        print(model_args)

        task = tasks.setup_task(model_args)
        print("loading model checkpoint from {}".format(checkpoint_path))

        model, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task)

        return model[0]


class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder, bertencoder, berttokenizer, mask_cls_sep, args=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bert_encoder = bertencoder
        self.berttokenizer = berttokenizer
        self.mask_cls_sep = mask_cls_sep
        self.bert_output_layer = getattr(args, 'bert_output_layer', -1)
        # outdim = self.encoder.layers[0].embed_dim
        # indim = self.bert_encoder.encoder.hidden_size
        # if not outdim == indim:
        #     self.trans_weight = Parameter(torch.Tensor(outdim, indim))
        #     bias = False
        #     if bias:
        #         self.trans_bias = Parameter(torch.Tensor(outdim))
        #     else:
        #         self.trans_bias = None
        # self.reset_parameters()
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trans_weight)
        if self.trans_bias is not None:
            nn.init.constant_(self.trans_bias, 0.)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, bert_input, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        bert_encoder_padding_mask = bert_input.eq(self.berttokenizer.pad())
        bert_encoder_out, _ =  self.bert_encoder(bert_input, output_all_encoded_layers=True, attention_mask= 1. - bert_encoder_padding_mask)
        bert_encoder_out = bert_encoder_out[self.bert_output_layer]
        if self.mask_cls_sep:
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.cls())
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.sep())
        bert_encoder_out = bert_encoder_out.permute(1,0,2).contiguous()
        # bert_encoder_out = F.linear(bert_encoder_out, self.trans_weight, self.trans_bias)
        bert_encoder_out = {
            'bert_encoder_out': bert_encoder_out,
            'bert_encoder_padding_mask': bert_encoder_padding_mask,
        }
        decoder_out = self.decoder(
            normalize=True, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()