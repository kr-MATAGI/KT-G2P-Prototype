import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nonauto_nmt.pos_nart_decoder import Decoder
from transformers import ElectraModel

import copy
import itertools
from typing import List
from hangul_utils import join_jamos, split_syllables

#========================================================
class ElectraNartPosDecModel(nn.Module):
#========================================================
    def __init__(
            self,
            args, tokenizer, decoder, electra_model,
            src_vocab, dec_vocab, post_proc_dict
    ):
        super().__init__()
        self.args = args

        self.tokenizer = tokenizer
        self.decoder = decoder
        self.electra = electra_model

        self.src_vocab = src_vocab
        self.dec_vocab = dec_vocab

        self.device = args.device
        self.post_proc_dict = post_proc_dict

        self.pad_ids = self.src_vocab.index('[PAD]') # 0
        self.unk_ids = self.src_vocab.index('[UNK]') # 1
        self.bos_ids = self.src_vocab.index('[CLS]') # 2
        self.eos_ids = self.src_vocab.index('[SEP]') # 3
        self.mask_ids = self.src_vocab.index('[MASK]') # 4
        self.gap_ids = self.src_vocab.index(' ') # 5

    @classmethod
    def build_decoder(cls, args, src_vocab, dec_vocab):
        base_decoder_architecture(args)
        return Decoder(args=args, src_vocab_size=len(src_vocab), dec_vocab=dec_vocab, positional=True)

    @classmethod
    def build_model(cls, args, tokenizer, src_vocab, dec_vocab, post_proc_dict):
        decoder = cls.build_decoder(args, src_vocab, dec_vocab)
        electra = ElectraModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)
        return ElectraNartPosDecModel(args, tokenizer, decoder, electra, src_vocab, dec_vocab, post_proc_dict)

    def prepare_masks(self, inputs):
        if inputs.ndimension() == 2:
            masks = (inputs.data != self.dec_vocab.index('[PAD]')).float()
        else:
            masks = (inputs.data[:, :, self.dec_vocab.index('[PAD]')] != 1).float()

        return masks

    def forward(
            self,
            src_tokens, **kwargs
    ):
        src_masks = self.prepare_masks(src_tokens)
        electra_out = self.electra(input_ids=src_tokens, attention_mask=src_masks)
        electra_out = electra_out.hidden_states
        '''
            electra_out.len: 13
            electra_out[0].size: [batch, seq_len, hidden]
        '''
        if 1 == self.args.dec_layers:
            electra_out = (electra_out[0], electra_out[-1])

        '''
            x = decoder_inputs,
            encoding = encodings
        '''

        decoder_out = self.decoder(x=src_tokens, encoding=electra_out, mask_src=src_masks)
        decoder_out = self.decoder.out(decoder_out) # [batch_size, seq_len, vocab_size]

        if self.args.do_post_method and 'eval' == kwargs['mode']:
            batch_size, seq_len = src_tokens.size()

            decoder_out = F.softmax(decoder_out, dim=-1)
            decoded_batch_eumjeol_sent = self._decode_batch_sentences(src_tokens)
            for t in range(seq_len):
                mutable_pron_list = self._get_mutable_pron_list(time_step=t, batch_size=batch_size,
                                                                origin_sent=decoded_batch_eumjeol_sent)
                # [ batch, vocab]
                mutable_pron_ids = self._get_score_handling_list(batch_size=batch_size,
                                                                 out_vocab_size=len(self.dec_vocab),
                                                                 mutable_pron_list=mutable_pron_list)
                decoder_out[:, t] *= mutable_pron_ids

                # for b in range(batch_size):
                #     decoder_out[b, t] *= mutable_pron_ids[b]

        return decoder_out


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
            decoded_sent = self.tokenizer.decode(input_item)

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
            # origin_char = '칙'
            if origin_char in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]:
                ret_mutable_pron.append([self.dec_vocab.index(origin_char)])
                continue
            elif " " == origin_char:
                ret_mutable_pron.append([self.gap_ids])
                continue

            origin_jaso = split_syllables(origin_char)

            ''' candi_* 가 붙은건 올 수 있는 자소들 '''
            candi_initial = []
            candi_vowel = []
            candi_final = []
            candi_initial = self.post_proc_dict["initial"][origin_jaso[0]]
            candi_vowel = self.post_proc_dict["vowel"][origin_jaso[1]]

            all_combination = []
            if 3 == len(origin_jaso):
                candi_final = copy.deepcopy(self.post_proc_dict["final"][origin_jaso[2]])
                if " " in candi_final:
                    empty_handle_list = list(itertools.product(candi_initial, candi_vowel))
                    empty_handle_list = [join_jamos("".join(x)) for x in empty_handle_list]
                    empty_handle_list = [self.dec_vocab.index(x) for x in empty_handle_list]
                    all_combination.extend(empty_handle_list)

            if 0 == len(candi_final):
                candi_combination = list(itertools.product(candi_initial, candi_vowel))
            else:
                candi_combination = list(itertools.product(candi_initial, candi_vowel, candi_final))
            candi_combination = [join_jamos("".join(x).strip()) for x in candi_combination]
            # print(candi_combination)
            # input()
            candi_combination = [self.dec_vocab.index(x) for x in candi_combination]
            all_combination.extend(candi_combination)

            ret_mutable_pron.append(all_combination)

        return ret_mutable_pron


    def _get_score_handling_list(self, batch_size: int, out_vocab_size: int,
                                 mutable_pron_list: List[List[int]]):
        ret_unmutable_tensor = torch.zeros(batch_size, out_vocab_size,
                                           device=self.args.device, dtype=torch.float32)

        for b_idx, mutable_pron in enumerate(mutable_pron_list):
            for vocab_ids in mutable_pron:
                ret_unmutable_tensor[b_idx][vocab_ids] = 1.

        return ret_unmutable_tensor


def base_decoder_architecture(args):
    # maybe decoder
    args.d_model = 768
    args.d_hidden = 768
    args.n_layers = args.dec_layers
    args.n_heads = 8
    args.drop_ratio = 0.1
    args.warmp = 16000
    args.input_orderless = True

    # decode
    args.length_ratio = 2
    args.decode_mode = 'argmax'
    args.beam_size = 1
    args.f_size = 1
    args.alpha = 1
    args.temperature = 1

    # need to know
    args.use_wo = True