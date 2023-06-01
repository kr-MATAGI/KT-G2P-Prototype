import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import  uniform
from fairseq import options, utils

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from model.trans_decoder.fairseq_model import FairseqEncoderDecoderModel

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from transformers import ElectraModel
from utils.kocharelectra_tokenization import KoCharElectraTokenizer

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

# 2023.04.10 - JAEHOON
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.iterative_refinement_generator import DecoderOut

from model.trans_decoder.transformer import ElectraFusedNATDecoder

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerDecoder,
)

import copy
import json
import itertools
from hangul_utils import split_syllables, join_jamos
from typing import List

class ElectraOnlyNART(nn.Module):
    def __init__(self, decoder, bertencoder, berttokenizer, mask_cls_sep=False, args=None):
        super().__init__()

        self.decoder = decoder
        self.bert_encoder = bertencoder
        self.berttokenizer = berttokenizer
        self.mask_cls_sep = mask_cls_sep
        self.args = args

        self.device = args.device
        self.bert_output_layer = -1

        #
        self.jaso_pair_dict = {}
        with open(self.args.syllable_constraint_vocab, mode="r", encoding="utf-8") as f:
            self.jaso_pair_dict = json.load(f)

    @classmethod
    def build_model(cls, args, source_dict, target_dict):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        # 2020.04.07 - JAEHOON
        # src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        src_dict, tgt_dict = source_dict, target_dict

        # 2020.04.07 - JAEHOON
        # if len(task.datasets) > 0:
        #     src_electratokenizer = next(iter(task.datasets.values())).berttokenizer
        # else:
        src_berttokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            # encoder_embed_tokens = build_embedding(
            #     src_dict, args.encoder_embed_dim, args.encoder_embed_path
            # )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        bert_encoder = ElectraModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)
        args.bert_out_dim = bert_encoder.config.hidden_size

        decoder = cls.build_nat_decoder(args, tgt_dict, decoder_embed_tokens)

        args.mask_cls_sep = False
        return ElectraOnlyDecModel(decoder, bert_encoder, src_berttokenizer, args.mask_cls_sep, args)

    @classmethod
    def build_nat_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = ElectraFusedNATDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

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

        bert_encoder_padding_mask = bert_input.eq(self.berttokenizer.pad_token_id).long()
        bert_encoder_out = self.bert_encoder(bert_input, attention_mask=1. - bert_encoder_padding_mask)
        bert_encoder_out = bert_encoder_out.hidden_states

        bert_encoder_out = bert_encoder_out[self.bert_output_layer]
        if self.mask_cls_sep:
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.cls_token_id).long()
            bert_encoder_padding_mask += bert_input.eq(self.berttokenizer.sep_token_id).long()
        bert_encoder_out = bert_encoder_out.permute(1, 0, 2).contiguous()

        ''' 이쪽 부분에서 Key가 원래 코드랑 다름 '''
        encoder_out = {
            'encoder_out': bert_encoder_out,
            'encoder_padding_mask': bert_encoder_padding_mask,
        }

        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, kwargs["tgt_tokens"]
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out
        )

        # 자소제한 후처리
        if self.args.do_post_method and "eval" == kwargs['mode']:
            decoded_batch_eumjeol_sent = self._decode_batch_sentences(src_tokens)
            for t in range(self.args.max_seq_len):
                mutable_pron_list = self._get_mutable_pron_list(time_step=t, batch_size=1,
                                                                origin_sent=decoded_batch_eumjeol_sent)
                mutable_pron_ids = self._get_score_handling_list(batch_size=1,
                                                                 out_vocab_size=len(self.berttokenizer),
                                                                 mutable_pron_list=mutable_pron_list)
                word_ins_out[0, t] *= mutable_pron_ids[0]

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": kwargs["tgt_tokens"],
                "mask": kwargs["tgt_tokens"].ne(self.berttokenizer.pad_token_id),
                "ls": 0.0,  # self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step
        ).max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )

        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def _decode_batch_sentences(self, input_ids):
        '''
            return은 아래와 같다
                [
                [CLS]', '런', '정', '페', '이', ' ', '화', '웨', '이', ' ', '회', '장', '은', '[SEP]',
                ...,
                ]
        '''
        decoded_batch_sent = []
        for input_item in input_ids.tolist():
            decoded_sent = self.berttokenizer.decode(input_item)

            # 분리를 위해 [CLS]/[SEP]는 뒤에 공백을 추가하고, [PAD]는 최대 길이 맞출려고 남겨둠
            decoded_sent = decoded_sent.replace("[CLS]", "[CLS] ").replace("[SEP]", " [SEP]").replace("[PAD]", " [PAD]")
            decoded_sent = decoded_sent.split(" ")

            conv_eumjeol = []
            for eojeol in decoded_sent:
                if eojeol in ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]:
                    if "[SEP]" == eojeol: # 문장의 맨 마지막 띄어쓰기 삭제
                        conv_eumjeol = conv_eumjeol[:-1]
                    conv_eumjeol.append(eojeol)
                    continue
                conv_eumjeol.extend(list(eojeol))
                conv_eumjeol.append(" ")
            decoded_batch_sent.append(conv_eumjeol)

        return decoded_batch_sent

    def _get_mutable_pron_list(self, time_step: int, batch_size: int, origin_sent: List[List[str]]):
        ret_mutable_pron = []

        for b_idx in range(batch_size):
            origin_char = origin_sent[b_idx][time_step]

            # TEST
            # origin_char = ' '

            if origin_char in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]:
                ret_mutable_pron.append([self.berttokenizer.encode(origin_char)[1]])
                continue
            elif " " == origin_char:
                ret_mutable_pron.append([self.berttokenizer.encode([" "])[1]])
                continue

            origin_jaso = split_syllables(origin_char)

            ''' candi_* 가 붙은건 올 수 있는 자소들 '''
            candi_initial = []
            candi_vowel = []
            candi_final = []
            candi_initial = self.jaso_pair_dict["initial"][origin_jaso[0]]
            candi_vowel = self.jaso_pair_dict["vowel"][origin_jaso[1]]

            all_combination = []
            if 3 == len(origin_jaso):
                candi_final = copy.deepcopy(self.jaso_pair_dict["final"][origin_jaso[2]])
                if " " in candi_final:
                    candi_final.remove(" ")
                    empty_handle_list = list(itertools.product(candi_initial, candi_vowel))
                    empty_handle_list = [join_jamos("".join(x)) for x in empty_handle_list]
                    empty_handle_list = [self.berttokenizer.encode(x)[1] for x in empty_handle_list]
                    all_combination.extend(empty_handle_list)

            if 0 == len(candi_final):
                candi_combination = list(itertools.product(candi_initial, candi_vowel))
            else:
                candi_combination = list(itertools.product(candi_initial, candi_vowel, candi_final))
            candi_combination = [join_jamos("".join(x)) for x in candi_combination]
            candi_combination = [self.berttokenizer.encode(x)[1] for x in candi_combination]

            all_combination.extend(candi_combination)

            ret_mutable_pron.append(all_combination)

        return ret_mutable_pron

    def _get_score_handling_list(self, batch_size: int, out_vocab_size: int,
                                 mutable_pron_list: List[List[int]]):
        ret_unmutable_tensor = torch.zeros(batch_size, out_vocab_size,
                                          device=self.device, dtype=torch.float32)
        # ret_unmutable_tensor.fill_(0.1)
        for b_idx, mutable_pron in enumerate(mutable_pron_list):
            for vocab_ids in mutable_pron:
                ret_unmutable_tensor[b_idx][vocab_ids] = 1.

        return ret_unmutable_tensor

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def base_architecture(args):
    # 2023.04.07 - JAEHOON
    args.encoder_ratio = 0.3
    args.bert_ratio = 0.1

    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', args.dec_layers)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

    # 2023.04.11 - JAEHOON
    args.decoder_layerdrop = 0
    args.no_scale_embedding = False
    args.quant_noise_pq = 0