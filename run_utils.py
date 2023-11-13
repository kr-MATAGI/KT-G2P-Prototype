import os
import re
import sys

import torch
from torch.utils.data import Dataset
import glob
import itertools

import pickle
import logging
import copy
import random
import numpy as np
import pandas as pd

from typing import Dict, List
from definition.data_def import KT_TTS
from utils.error_fixer import ERR_SENT_ID_FIXED, ERR_SENT_CHANGED_FIXED
from utils.kt_tts_pkl_maker import KT_TTS_Maker
from utils.english_to_korean import Eng2Kor
from soynlp.hangle import jamo_levenshtein
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

def find_words_with_number_or_english(sentence):
    # 문장을 공백으로 분리하여 어절 리스트 생성
    words = sentence.split()

    # 숫자나 영어 문자가 들어간 어절의 인덱스를 저장할 리스트
    result_indices = []

    # 어절을 하나씩 확인하면서 숫자나 영어 문자가 있는지 검사
    for idx, word in enumerate(words):
        if re.search(r'[0-9a-zA-Z]', word):
            result_indices.append(idx)

    return result_indices

#==================================================
def make_digits_ensemble_data(
        args, data_path: str, num2kor,
        tokenizer, decode_vocab, max_seq_len: int=256
):
#==================================================
    print(f'[run_utils][make_digits_ensemble_data], data_path: {data_path}')

    if not os.path.exists(data_path):
        raise Exception(f'ERR - data_path: {data_path}')

    ''' 특수문자 처리하기 위해  '''
    sym2kor = KT_TTS_Maker()

    src_list = [f"{data_path}train_src.pkl", f"{data_path}dev_src.pkl", f"{data_path}test_src.pkl"]
    tgt_list = [f"{data_path}train_tgt.pkl", f"{data_path}dev_tgt.pkl", f"{data_path}test_tgt.pkl"]


    # src_list = [f"{data_path}test_src.pkl"]
    # tgt_list = [f"{data_path}test_tgt.pkl"]
    print(f'[run_utils][make_digits_ensemble_data] src_path: {src_list},\ntgt_path: {tgt_list}')

    all_src_data: List[KT_TTS] = []
    for src_path in src_list:
        with open(src_path, mode='rb') as s_f:
            all_src_data.extend(pickle.load(s_f))
    print(f'[run_utils][make_digits_ensemble_data] all_src_data.size: {len(all_src_data)}')
    print(f'{all_src_data[:10]}')

    all_tgt_data: List[KT_TTS] = []
    for tgt_path in tgt_list:
        with open(tgt_path, mode='rb') as t_f:
            all_tgt_data.extend(pickle.load(t_f))
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

    ''' eng to kor '''
    eng2kor = Eng2Kor()

    eng_count = []
    preprocess_data = {'id':[],
                        'source':[],
                       'pred':[],
                       'target':[]
                       }
    train_len_error = 0
    for r_idx, (src_data, tgt_data) in enumerate(zip(all_src_data, all_tgt_data)):
        original_source = src_data.sent.strip()
        indices = find_words_with_number_or_english(original_source)
        src_data.sent = src_data.sent.strip()
        tgt_data.sent = tgt_data.sent.strip()
        ''' Convert num2kor '''
        src_data.sent = num2kor.generate(src_data.sent)

        ''' Convert sym2kor '''
        src_data = sym2kor.get_converted_symbol_items(src_data)

        ''' Convert Eng2Kor '''
        src_data = eng2kor.convert_eng(src_data, use_ipa=args.use_ipa)

        ''' Check english word in src_data '''
        r_eng = r"[a-zA-Z]+"
        if re.search(r_eng, src_data.sent):
            eng_count.append(src_data.sent)
            continue

        ''' Convert Chinese2Kor '''
        src_data = sym2kor.convert_chinese(src_data)

        # 띄어쓰기 하나로
        src_data.sent = re.sub(r'\s{2,}', " ", src_data.sent)

        # # ''' compare num, eng to tgt '''
        # for i in indices:
        #     try:
        #         src_word = src_data.sent.split()[i]
        #         tgt_word = tgt_data.sent.split()[i]
        #         for j in range(len(src_word)):
        #             edit = jamo_levenshtein(src_word[j], tgt_word[j])
        #             if edit >= 1:
        #                 preprocess_data['id'].append(r_idx)
        #                 preprocess_data['source'].append(original_source)
        #                 preprocess_data['pred'].append(src_data.sent)
        #                 preprocess_data['target'].append(tgt_data.sent)
        #                 break
        #     except:
        #         preprocess_data['id'].append(r_idx)
        #         preprocess_data['source'].append(original_source)
        #         preprocess_data['pred'].append(src_data.sent)
        #         preprocess_data['target'].append(tgt_data.sent)
        #         break


        # if 0 == (r_idx % 1000):
        #     print(f'[run_utils][make_digits_ensemble_data] {r_idx} is processing... {src_data.sent}')

        src_tokens = tokenizer(src_data.sent, padding='max_length', max_length=max_seq_len,
                               return_tensors='np', truncation=True)
        if tgt_data.id in ERR_SENT_ID_FIXED.keys():
            # print(f'[run_utils][make_digits_ensemble_data] ERR DETECTED -> {tgt_data.id}, {tgt_data.sent}')
            # print(f'[run_utils][make_digits_ensemble_data] {ERR_SENT_ID_FIXED[tgt_data.id]}')
            tgt_data.sent = tgt_data.sent.replace(ERR_SENT_ID_FIXED[tgt_data.id][0],
                                                  ERR_SENT_ID_FIXED[tgt_data.id][1])
            # print(f'[run_utils][make_digits_ensemble_data] ERR FIXED -> {tgt_data.sent}')

        if src_data.id in ERR_SENT_CHANGED_FIXED.keys():
            # print(f'[run_utils][make_digits_ensemble_data] ERR Sent\ninput:\n{src_data.sent}\nans:\n{tgt_data.sent}')
            # print(f'[run_utils][make_digits_ensemble_data] \n{ERR_SENT_CHANGED_FIXED[src_data.id][0]} ->'
            #       f'\n{ERR_SENT_CHANGED_FIXED[src_data.id][1]}')
            src_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][0]
            tgt_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][1]

        if re.search(r'[^가-힣\s]+', src_data.sent):
            continue

        if len(src_data.sent) != len(tgt_data.sent):
            train_len_error += 1
            continue

        tgt_tokens = [decode_vocab.index('[CLS]')] + [decode_vocab.index(x) for x in list(tgt_data.sent)] \
                     + [decode_vocab.index('[SEP]')]
        if max_seq_len <= len(tgt_tokens):
            tgt_tokens = tgt_tokens[:max_seq_len - 1]
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
    # pre_df = pd.DataFrame(preprocess_data)
    # pre_df.to_excel('preprocess_data.xlsx', index=False, engine='openpyxl')

    # # convert list to np
    # for key, val in ret_dict.items():
    #     ret_dict[key] = torch.LongTensor(np.array(val))
    #     print(f'[run_utils][make_digits_ensemble_data] {key}.size: {ret_dict[key].size()}')

    total_size = len(ret_dict['src_tokens'])
    train_size = total_size * 0.8
    val_size = train_size + (total_size * 0.1)

    train_dict = {
        'src_tokens': [],
        'src_lengths': [],
        'attention_mask': [],
        'prev_output_tokens': [],
        'target': []
    }
    val_dict = {
    'src_tokens': [],
    'src_lengths': [],
    'attention_mask': [],
    'prev_output_tokens': [],
    'target': []
    }
    test_dict = {
    'src_tokens': [],
    'src_lengths': [],
    'attention_mask': [],
    'prev_output_tokens': [],
    'target': []
    }

    for i in range(len(ret_dict['src_tokens'])):
        # s = ret_dict['src_tokens'][i][ret_dict['src_tokens'][i] != 0]
        # t = [x for x in ret_dict['target'][i] if x != 0]
        # if len(s) != len(t):
        #     train_len_error += 1
        #     continue
        if i < train_size:
            train_dict['src_tokens'].append(ret_dict['src_tokens'][i])
            train_dict['src_lengths'].append(ret_dict['src_lengths'][i])
            train_dict['attention_mask'].append(ret_dict['attention_mask'][i])
            train_dict['prev_output_tokens'].append(ret_dict['prev_output_tokens'][i])
            train_dict['target'].append(ret_dict['target'][i])
        elif i < val_size:
            val_dict['src_tokens'].append(ret_dict['src_tokens'][i])
            val_dict['src_lengths'].append(ret_dict['src_lengths'][i])
            val_dict['attention_mask'].append(ret_dict['attention_mask'][i])
            val_dict['prev_output_tokens'].append(ret_dict['prev_output_tokens'][i])
            val_dict['target'].append(ret_dict['target'][i])
        else:
            test_dict['src_tokens'].append(ret_dict['src_tokens'][i])
            test_dict['src_lengths'].append(ret_dict['src_lengths'][i])
            test_dict['attention_mask'].append(ret_dict['attention_mask'][i])
            test_dict['prev_output_tokens'].append(ret_dict['prev_output_tokens'][i])
            test_dict['target'].append(ret_dict['target'][i])
    print("train_len_error:", train_len_error)

    print(f"[run_utils][make_digits_ensemble_data] train/val/test = {len(train_dict['src_tokens'])}/"
          f"{len(val_dict['src_tokens'])}/{len(test_dict['src_tokens'])}")
    print(f"[run_utils][make_digits_ensemble_data] pass english sentence = {len(eng_count)}")


    # convert list to np
    for key, val in train_dict.items():
        train_dict[key] = torch.LongTensor(np.array(val))

    for key, val in val_dict.items():
        val_dict[key] = torch.LongTensor(np.array(val))

    for key, val in test_dict.items():
        test_dict[key] = torch.LongTensor(np.array(val))

    # err_eng_df = pd.DataFrame(eng_count)
    # print(err_eng_df)
    # err_eng_df.to_csv('no_match_eng2.txt', index=False, encoding='utf-8')

    return train_dict, val_dict, test_dict


# ==================================================
def make_inference_data(
        args, data_path: str, num2kor,
        tokenizer, decode_vocab, max_seq_len: int = 256
):

    original_sentence = []
    # ==================================================
    print(f'[run_utils][make_digits_ensemble_data], data_path: {data_path}')

    if not os.path.exists(data_path):
        raise Exception(f'ERR - data_path: {data_path}')

    ''' 특수문자 처리하기 위해  '''
    sym2kor = KT_TTS_Maker()

    src_list = glob.glob(os.path.join(data_path, "*.pkl"))
    print(f'[run_utils][make_inference_data] src_path: {src_list}')

    all_src_data: List[KT_TTS] = []
    for src_path in src_list:
        with open(src_path, mode='rb') as s_f:
            all_src_data.extend(pickle.load(s_f))
    print(f'[run_utils][make_digits_ensemble_data] all_src_data.size: {len(all_src_data)}')
    print(f'{all_src_data[:10]}')

    all_tgt_data = all_src_data.copy()

    # Tokenization
    ret_dict = {
        'src_tokens': [],
        'src_lengths': [],
        'attention_mask': [],
        'prev_output_tokens': [],
        'target': []
    }

    eng_count = []
    for r_idx, (src_data, tgt_data) in enumerate(zip(all_src_data, all_tgt_data)):
        src_tmp = src_data.sent.strip()

        src_data.sent = src_data.sent.strip()
        tgt_data.sent = tgt_data.sent.strip()

        try:
            ''' Convert num2kor '''
            src_data.sent = num2kor.generate(src_data.sent)
        except:
            print(src_data.sent)
            sys.exit()

        ''' Convert sym2kor '''
        src_data = sym2kor.get_converted_symbol_items(src_data)

        ''' Convert Eng2Kor '''
        eng2kor = Eng2Kor()
        src_data = eng2kor.convert_eng(src_data)

        ''' Check english word in src_data '''
        r_eng = r"[a-zA-Z]+"
        if re.search(r_eng, src_data.sent):
            eng_count.append(src_data.sent)
            continue

        ''' Check special characters in src_data '''
        sp_pattern = r"[!@#$%^&*(),.?\":{}|<>]"
        if re.search(sp_pattern, src_data.sent):
            print("====================================")
            print("[run_utils][make_inference_data] symbol error", src_data)
            sys.exit()

        if 0 == (r_idx % 1000):
            print(f'[run_utils][make_inference_data] {r_idx} is processing... {src_data.sent}')

        src_tokens = tokenizer(src_data.sent, padding='max_length', max_length=max_seq_len,
                               return_tensors='np', truncation=True)

        if re.search(r'[^가-힣\s]+', src_data.sent):
            continue

        original_sentence.append(src_tmp)

        tgt_tokens = [decode_vocab.index('[CLS]')] + [decode_vocab.index(x) for x in list(tgt_data.sent)] \
                     + [decode_vocab.index('[SEP]')]
        if max_seq_len <= len(tgt_tokens):
            tgt_tokens = tgt_tokens[:max_seq_len - 1]
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
        print(f'[run_utils][make_inference_data] {key}.size: {ret_dict[key].size()}')

    print(f"[run_utils][make_inference_data] pass english sentence = {len(eng_count)}")

    err_eng_df = pd.DataFrame(eng_count)
    err_eng_df.to_csv('no_match_inference_eng.txt', index=False, encoding='utf-8')

    return ret_dict, original_sentence

