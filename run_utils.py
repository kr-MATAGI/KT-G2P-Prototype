import os
import re
import torch
from torch.utils.data import Dataset
import numpy as np

import pickle
import logging
import copy
import random
import numpy as np

from typing import Dict, List
from definition.data_def import KT_TTS
from utils.error_fixer import ERR_SENT_ID_FIXED, ERR_SENT_CHANGED_FIXED

#===============================================================
def init_logger():
# ===============================================================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

#===============================================================
def set_seed(seed):
#===============================================================
    torch.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#===============================================================
def print_args(args, logger):
#===============================================================
    for key, val in args.items():
        logger.info(f"[print_args] {key}: {val}")

#===============================================================
def load_npy_file(src_path: str, mode: str):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    input_ids = np.load(root_path + "_input_ids.npy")
    attention_mask = np.load(root_path + "_attention_mask.npy")
    token_type_ids = np.load(root_path + "_token_type_ids.npy")
    labels = np.load(root_path + "_labels.npy")

    print(f"[run_utils][load_npy_file] {mode}.npy.shape:")
    print(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, "
          f"token_type_ids: {token_type_ids.shape}")

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask, "token_type_ids": token_type_ids
    }
    return inputs, labels

#===============================================================
class G2P_Dataset(Dataset):
#===============================================================
    def __init__(
            self,
            item_dict: Dict[str, np.ndarray],
            labels: np.ndarray,
    ):
        self.input_ids = torch.tensor(item_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(item_dict["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(item_dict["token_type_ids"], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx]
        }

        return items

#===============================================================
def make_inputs_from_batch(batch: torch.Tensor, device: str):
#===============================================================
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
        "labels": batch["labels"].to(device)
    }

    return inputs

#===============================================================
def make_eojeol_mecab_res(input_sent: str, mecab_res: List):
#===============================================================
    # 어절별 글자 수 체크해서 띄워쓰기 적기
    split_text = input_sent.split(" ")
    char_cnt_list = [len(st) for st in split_text]

    total_eojeol_morp = []
    eojeol_set = []
    use_check = [False for _ in range(len(mecab_res))]
    len_idx = 0
    curr_char_cnt = 0
    for ej_idx, eojeol in enumerate(mecab_res):
        if char_cnt_list[len_idx] == curr_char_cnt:
            total_eojeol_morp.append(copy.deepcopy(eojeol_set))
            eojeol_set = []
            len_idx += 1
            curr_char_cnt = 0
        if use_check[ej_idx]:
            continue

        eojeol_set.append((eojeol[0], eojeol[1].split("+")))
        curr_char_cnt += len(eojeol[0].strip())  # 에듀' '  <- Mecab 결과에 공백이 따라오는 경우 있음
        use_check[ej_idx] = True
    if 0 < len(eojeol_set):
        total_eojeol_morp.append(copy.deepcopy(eojeol_set))

    return total_eojeol_morp

#==================================================
def make_digits_ensemble_data(
        data_path: str, mode: str, num2kor,
        tokenizer, decode_vocab, max_seq_len: int=256
):
#==================================================
    print(f'[run_utils][make_digits_ensemble_data] mode: {mode}, data_path: {data_path}')

    if not os.path.exists(data_path):
        raise Exception(f'ERR - data_path: {data_path}')

    src_path = data_path + '/' + mode + '_src.pkl'
    tgt_path = data_path + '/' + mode + '_tgt.pkl'
    print(f'[run_utils][make_digits_ensemble_data] src_path: {src_path},\ntgt_path: {tgt_path}')

    all_src_data: List[KT_TTS] = []
    with open(src_path, mode='rb') as s_f:
        all_src_data = pickle.load(s_f)
        print(f'[run_utils][make_digits_ensemble_data] all_src_data.size: {len(all_src_data)}')
        print(f'{all_src_data[:10]}')

    all_tgt_data: List[KT_TTS] = []
    with open(tgt_path, mode='rb') as t_f:
        all_tgt_data = pickle.load(t_f)
        print(f'[run_utils][make_digits_ensemble_data] all_tgt_data.size: {len(all_tgt_data)}')
        print(f'{all_tgt_data[:10]}')

    assert len(all_src_data) == len(all_tgt_data), f'ERR - src_data.size != tgt_data.size'

    # Tokenization
    ret_dict = {
        'src_tokens': [],
        'src_lengths': [],
        'attention_mask': [],
        'prev_output_tokens': [],
        'target': []
    }

    for r_idx, (src_data, tgt_data) in enumerate(zip(all_src_data, all_tgt_data)):
        src_data.sent = src_data.sent.strip()
        tgt_data.sent = tgt_data.sent.strip()

        ''' Convert num2kor '''
        src_data.sent = num2kor.generate(src_data.sent)

        if 0 == (r_idx % 1000):
            print(f'[run_utils][make_digits_ensemble_data] {r_idx} is processing... {src_data.sent}')

        src_tokens = tokenizer(src_data.sent, padding='max_length', max_length=max_seq_len,
                               return_tensors='np', truncation=True)
        if tgt_data.id in ERR_SENT_ID_FIXED.keys():
            print(f'[run_utils][make_digits_ensemble_data] ERR DETECTED -> {tgt_data.id}, {tgt_data.sent}')
            print(f'[run_utils][make_digits_ensemble_data] {ERR_SENT_ID_FIXED[tgt_data.id]}')
            tgt_data.sent = tgt_data.sent.replace(ERR_SENT_ID_FIXED[tgt_data.id][0],
                                                  ERR_SENT_ID_FIXED[tgt_data.id][1])
            print(f'[run_utils][make_digits_ensemble_data] ERR FIXED -> {tgt_data.sent}')

        if src_data.id in ERR_SENT_CHANGED_FIXED.keys():
            print(f'[run_utils][make_digits_ensemble_data] ERR Sent\ninput:\n{src_data.sent}\nans:\n{tgt_data.sent}')
            print(f'[run_utils][make_digits_ensemble_data] \n{ERR_SENT_CHANGED_FIXED[src_data.id][0]} ->'
                  f'\n{ERR_SENT_CHANGED_FIXED[src_data.id][1]}')
            src_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][0]
            tgt_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][1]

        if re.search(r'[^가-힣\s]+', src_data.sent):
            continue

        tgt_tokens = [decode_vocab.index('[CLS]')] + [decode_vocab.index(x) for x in list(tgt_data.sent)] \
                     + [decode_vocab.index('[SEP]')]
        if max_seq_len <= len(tgt_tokens):
            tgt_tokens = tgt_tokens[:max_seq_len-1]
            tgt_tokens.append(decode_vocab.index('[SEP]'))
        else:
            diff_size = max_seq_len - len(tgt_tokens)
            tgt_tokens += [decode_vocab.index('[PAD]')] * diff_size
        assert max_seq_len == len(tgt_tokens), f'ERR - tgt_tokens.size: {len(tgt_tokens)}'

        cls_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[CLS]')[1])[0][0]
        sep_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[SEP]')[1])[0][0]
        src_lengths = len(src_tokens['input_ids'][0][cls_idx:sep_idx + 1])

        ret_dict['src_tokens'].append(src_tokens['input_ids'][0])
        ret_dict['src_lengths'].append(src_lengths)
        ret_dict['attention_mask'].append(src_tokens['attention_mask'][0])
        ret_dict['prev_output_tokens'].append(src_tokens['input_ids'][0])
        ret_dict['target'].append(tgt_tokens)
    # end loop

    # convert list to np
    for key, val in ret_dict.items():
        ret_dict[key] = torch.LongTensor(np.array(val))
        print(f'[run_utils][make_digits_ensemble_data] {mode}.{key}.size: {ret_dict[key].size()}')

    return ret_dict

