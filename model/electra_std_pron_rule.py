import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import copy

from typing import List

from hangul_utils import split_syllables, join_jamos
from transformers import ElectraModel, ElectraPreTrainedModel

#==============================================================
class ElectraStdPronRules(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config, tokenizer, out_ids2tag, out_tag2ids, jaso_pair_dict):
        super(ElectraStdPronRules, self).__init__(config)

        # init
        self.tokenizer = tokenizer
        self.out_ids2tag = out_ids2tag
        self.out_tag2ids = out_tag2ids
        self.jaso_pair_dict = jaso_pair_dict

        self.config = config
        self.dropout_p = 0.1
        self.max_seq_len = config.max_seq_len

        self.pad_ids = config.pad_ids # 0
        self.unk_ids = config.unk_ids # 1
        self.start_ids = config.start_ids # 2
        self.end_ids = config.end_ids # 3
        self.mask_ids = config.mask_ids # 4
        self.special_tok_ids_list = [self.start_ids, self.end_ids, self.pad_ids, self.unk_ids, self.mask_ids]
        self.gap_ids = config.gap_ids # 5

        # PLM
        self.electra = ElectraModel.from_pretrained(config.model_name_or_path)
        self.dropout = nn.Dropout(self.dropout_p)

        # Encoder
        self.encoder = nn.LSTM(input_size=config.hidden_size, hidden_size=(config.hidden_size//2),
                               bidirectional=True, batch_first=True, num_layers=1)
        self.hx_dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Decoder
        self.decoder_cell = nn.LSTMCell(input_size=config.hidden_size, hidden_size=config.hidden_size,
                                        device=config.device)
        self.fc_layer = nn.Linear(config.hidden_size, config.out_vocab_size)


        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, mode: str="train"):
        '''
            mode
                1) train
                2) eval
        '''
        # KoCharELECTRA
        electra_out = self.electra(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        electra_out = electra_out.last_hidden_state
        electra_out = self.dropout(electra_out)

        # Encoder
        enc_out, enc_hn = self.encoder(electra_out)
        enc_hn = self._transform_decoder_init_state(enc_hn)

        # Decoder
        '''
            nn.LSTM을 이용할 경우 : 
                dec_out.size -> (batch, seq_len, hidden)
                h -> (1, batch, hidden)
                c -> (1, batch, hidden)
            
            nn.LSTMCell을 이용할 경우 :
                h_t, c_t : (batch, hidden)
        '''
        batch_size, seq_len = input_ids.size()
        h_t, c_t = enc_hn
        h_t = h_t.squeeze(0)
        c_t = c_t.squeeze(0)

        '''
            decoded_batch_sent.size: [batch_size, seq_len]
        '''
        if "eval" == mode:
            decoded_batch_eumjeol_sent = self._decode_batch_sentences(input_ids)

        output = []
        for t in range(seq_len):
            h_t, c_t = self.decoder_cell(enc_out[:, t], (h_t, c_t))
            out_t = self.fc_layer(h_t) # [batch, out_vocab_size]

            if "eval" == mode:
                '''
                    현재 시점에서 적용할 수 있는 발음열(음절)을 가져온다
                    mutable_pron_list : [batch, mutable_list_ids_size]
                    mutable_pron_ids : [batch, out_vocab_size] 
                '''
                mutable_pron_list = self._get_mutable_pron_list(t, batch_size, decoded_batch_eumjeol_sent)
                mutable_pron_ids = self._get_score_handling_list(batch_size,
                                                                 self.config.out_vocab_size, mutable_pron_list)
                out_t = F.softmax(out_t, dim=1)
                out_t = out_t * mutable_pron_ids

            output.append(out_t)
        # end loop, seq_len
        output = torch.stack(output, dim=1) # [batch, seq, out_vocab_size]

        return output

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

    def _transform_decoder_init_state(self, hn: torch.Tensor) -> torch.Tensor:
        hn, cn = hn
        cn = cn[-2:]  # take the last layer
        _, batch_size, hidden_size = cn.size()
        cn = cn.transpose(0, 1).contiguous()
        cn = cn.view(batch_size, 1, 2 * hidden_size).transpose(0, 1)
        cn = self.hx_dense(cn)

        hn = torch.tanh(cn)
        hn = (hn, cn)
        return hn

    def _get_mutable_pron_list(self, time_step: int, batch_size: int, origin_sent: List[List[str]]):
        ret_mutable_pron = []

        for b_idx in range(batch_size):
            origin_char = origin_sent[b_idx][time_step]
            
            # TEST
            # origin_char = '석'
            
            if origin_char in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]:
                ret_mutable_pron.append([self.out_tag2ids[origin_char]])
                continue
            elif " " == origin_char:
                ret_mutable_pron.append([self.gap_ids])
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
                    empty_handle_list = [self.out_tag2ids[x] for x in empty_handle_list]
                    all_combination.extend(empty_handle_list)

            if 0 == len(candi_final):
                candi_combination = list(itertools.product(candi_initial, candi_vowel))
            else:
                candi_combination = list(itertools.product(candi_initial, candi_vowel, candi_final))
            candi_combination = [join_jamos("".join(x)) for x in candi_combination]
            candi_combination = [self.out_tag2ids[x] for x in candi_combination]
            all_combination.extend(candi_combination)

            ret_mutable_pron.append(all_combination)

        return ret_mutable_pron

    def _get_score_handling_list(self, batch_size: int, out_vocab_size: int,
                                 mutable_pron_list: List[List[int]]):
        ret_unmutable_tensor = torch.zeros(batch_size, out_vocab_size,
                                          device=self.config.device, dtype=torch.float32)

        for b_idx, mutable_pron in enumerate(mutable_pron_list):
            for vocab_ids in mutable_pron:
                ret_unmutable_tensor[b_idx][vocab_ids] = 1.

        return ret_unmutable_tensor