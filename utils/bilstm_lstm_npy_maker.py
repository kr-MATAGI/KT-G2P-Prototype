import os
import pickle
import json
import re
import numpy as np
import copy

import random
random.seed(42)

from typing import Dict, List

from kocharelectra_tokenization import KoCharElectraTokenizer
from definition.data_def import KT_TTS
from hangul_utils import split_syllables, join_jamos

import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux

#========================================================
class LstmEncDecNpyMaker:
#========================================================
    def __init__(self,
                 b_debug_mode: bool=False,
                 b_use_out_vocab: bool=False):
        print("[LstmEncDecNpyMaker] __init__ !")

        self.b_use_out_vocab = False
        self.b_debug_mode = b_debug_mode
        print(f"[LstmEncDecNpyMaker][__init__] b_use_out_vocab: {self.b_use_out_vocab}, "
              f"b_debug_mod: {self.b_debug_mode}")

    def make_bilstm_lstm_npy(
            self,
            raw_path: str, g2p_path: str,
            save_path: str, out_vocab_path: str,
            max_seq_len: int=256
    ):
        print(f"[LstmEncDecNpyMaker][make_bilstm_lstm_npy] raw_data: {raw_path},\ng2p_path: {g2p_path}")
        print(f"[LstmEncDecNpyMaker][make_bilstm_lstm_npy] out_vocab_path:{out_vocab_path}")

        if not os.path.exists(raw_path):
            raise Exception("Not Existed -", raw_path)
        if not os.path.exists(g2p_path):
            raise Exception("Not Existed -", g2p_path)

        if os.path.exists(out_vocab_path):
            self.b_use_out_vocab = True
        else:
            self.b_use_out_vocab = False
            print(f'[LstmEncDecNpyMaker][make_bilstm_lstm_npy] Not Existed - out_vocab_path !')

        # Load Raw Data
        all_raw_data = all_g2p_data = []
        with open(raw_path, mode="rb") as f:
            all_raw_data = pickle.load(f)
        with open(g2p_path, mode="rb") as f:
            all_g2p_data = pickle.load(f)
        print(f"[LstmEncDecNpyMaker][make_bilstm_lstm_npy] all_raw_data.size:: {len(all_raw_data)}, "
              f"all_g2p_data.size: {len(all_g2p_data)}")
        assert len(all_raw_data) == len(all_g2p_data), "ERR - diff size"

        # Tokenization
        npy_dict = self._tokenization(tokenizer_name="monologg/kocharelectra-base-discriminator",
                                      raw_data_list=all_raw_data, g2p_data_list=all_g2p_data, max_seq_len=max_seq_len,
                                      out_vocab_path=out_vocab_path)

        # Save
        self._save_npy(npy_dict=npy_dict, save_path=save_path)

    def _tokenization(self, tokenizer_name: str,
                      raw_data_list: List[KT_TTS], g2p_data_list: List[KT_TTS],
                      out_vocab_path: str, max_seq_len: int=256):
        print(f"[LstmEncDecNpyMaker][_tokenization] max_seq_len: {max_seq_len}")

        npy_dict = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "labels": []
        }
        except_output_pron_list = [] # (음절) 발음열을 따로 사용할 경우에만 데이터가 채워짐

        # init
        tokenizer = KoCharElectraTokenizer.from_pretrained(tokenizer_name)

        # 발음열 출력 사전
        if self.b_use_out_vocab:
            out_token_dict: Dict[str, int] = {}
            with open(out_vocab_path, mode="r", encoding="utf-8") as f:
                out_token_dict = json.load(f)
            print(f"[LstmEncDecNpyMaker][_tokenization] out_token_dict.size: {len(out_token_dict.keys())}")
            print(list(out_token_dict.items())[:10])

            out_token_ids2tag = {v: k for k, v in out_token_dict.items()}

        # Loop
        sent_max_len = 0
        total_sent_len = 0
        max_eojeol_size = 0
        for root_idx, (raw_data, g2p_data) in enumerate(zip(raw_data_list, g2p_data_list)):
            raw_data.sent = raw_data.sent.strip()
            g2p_data.sent = g2p_data.sent.strip()

            if not re.match(r"[가-힣]+", raw_data.sent):
                except_output_pron_list.append((raw_data.id, raw_data.sent, g2p_data.sent))
                continue
            if re.search(r"[ㄱ-ㅎ]+", raw_data.sent) or re.search(r'[ㅏ-ㅣ]+', raw_data.sent):
                except_output_pron_list.append((raw_data.id, raw_data.sent, g2p_data.sent))
                continue

            if raw_data.id != g2p_data.id:
                raise Exception(f"ERR - ID diff, raw_data: {raw_data.id}, g2p_data: {g2p_data.id}")

            # 2023.03.23에 추가 (반사,라고 -> 반사라고, 모델예측: 반사라고 답: 반사 라고)
            if len(raw_data.sent) != len(g2p_data.sent):
                except_output_pron_list.append((raw_data.id, raw_data.sent, g2p_data.sent))
                continue

            if 0 == (root_idx % 1000):
                print(f"[LstmEncDecNpyMaker][make_kt_tts_npy] {root_idx} is processing... {raw_data.sent}")

            total_sent_len += len(raw_data.sent)
            if sent_max_len < len(raw_data.sent):
                sent_max_len = len(raw_data.sent)
            if max_eojeol_size < len(raw_data.sent.split(" ")):
                max_eojeol_size = len(raw_data.sent.split(" "))

            # For raw_data
            raw_tokens = tokenizer(raw_data.sent, padding="max_length", max_length=max_seq_len, return_tensors="np",
                                   truncation=True)

            # For g2p_data
            ''' 발음열 문장에서 '였' '것' 같은 오류의 종성을 'ㄷ'으로 처리 '''
            for g_idx, g2p_item in enumerate(g2p_data.sent):
                g2p_jaso = list(split_syllables(g2p_item))
                if 3 == len(g2p_jaso) and ('ㅅ' == g2p_jaso[-1] or 'ㅆ' == g2p_jaso[-1]):
                    g2p_jaso[-1] = 'ㄷ'
                    g2p_data.sent = g2p_data.sent.replace(g2p_item, join_jamos("".join(g2p_jaso)))

            if self.b_use_out_vocab:
                g2p_tokens = {"input_ids": []}

                split_g2p = list(g2p_data.sent)
                split_g2p.insert(0, "[CLS]")

                if max_seq_len <= len(split_g2p):
                    split_g2p = split_g2p[:max_seq_len-1]
                    split_g2p.append("[SEP]")
                else:
                    split_g2p.append("[SEP]")
                    split_g2p += ["[PAD]"] * (max_seq_len - len(split_g2p))
                g2p_tokens["input_ids"].append([out_token_dict[x] for x in split_g2p])
            else:
                g2p_tokens = tokenizer(g2p_data.sent, padding="max_length", max_length=max_seq_len,
                                       return_tensors="np", truncation=True)

            # Check size
            assert max_seq_len == len(raw_tokens["input_ids"][0]), f"ERR - input_ids"
            assert max_seq_len == len(raw_tokens["attention_mask"][0]), "ERR - attention_mask"
            assert max_seq_len == len(raw_tokens["token_type_ids"][0]), "ERR - token_type_ids"
            assert max_seq_len == len(g2p_tokens["input_ids"][0]), "ERR - g2p_tokens"

            # Insert npy_dict
            npy_dict["input_ids"].append(raw_tokens["input_ids"][0])
            npy_dict["attention_mask"].append(raw_tokens["attention_mask"][0])
            npy_dict["token_type_ids"].append(raw_tokens["token_type_ids"][0])
            npy_dict["labels"].append(g2p_tokens["input_ids"][0])

        # (음절) 발음열 사전을 사용할 경우 예외에 대한 출력
        print("[LstmEncDecNpyMaker][make_kt_tts_npy] (음절) 발음열 사전 사용시 오류가 발생한 문장")
        for e_idx, except_item in enumerate(except_output_pron_list):
            print(e_idx, except_item)

        print(f"[LstmEncDecNpyMaker][make_kt_tts_npy] sent_max_len: {sent_max_len}, "
              f"mean_sent_len: {total_sent_len/len(raw_data_list)}")

        print(f"[LstmEncDecNpyMaker][make_kt_tts_npy] max_eojeol_size: {max_eojeol_size}")

        return npy_dict

    def _save_npy(self, npy_dict: Dict[str, List], save_path: str):
        total_size = len(npy_dict["input_ids"])
        print(f"[LstmEncDecNpyMaker][_save_npy] save_path: {save_path}, total_size: {total_size}")

        split_ratio = 0.1
        dev_s_idx = int(total_size * (split_ratio * 8))
        dev_e_idx = int(dev_s_idx + (total_size * split_ratio))
        print(f"[LstmEncDecNpyMaker][_save_npy] split_ratio: {split_ratio}, dev_s_idx: {dev_s_idx}, dev_e_idx: {dev_e_idx}")

        npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
        npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
        npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
        npy_dict["labels"] = np.array(npy_dict["labels"])

        # Train
        train_datasets = {
            'input_ids': None,
            'attention_mask': None,
            'token_type_ids': None,
            'labels': None
        }
        train_datasets['input_ids'] = npy_dict["input_ids"][:dev_s_idx]
        train_datasets['attention_mask'] = npy_dict["attention_mask"][:dev_s_idx]
        train_datasets['token_type_ids'] = npy_dict["token_type_ids"][:dev_s_idx]
        train_datasets['labels'] = npy_dict["labels"][:dev_s_idx]
        print(f"[KoCharNpyMaker][_save_npy] Train.shape\ninput_ids: {train_datasets['input_ids'].shape}, "
              f"attention_mask: {train_datasets['attention_mask'].shape}, "
              f"token_type_ids: {train_datasets['token_type_ids'].shape}, "
              f"labels.shape: {train_datasets['labels'].shape}")

        # Dev
        dev_datasets = {k: None for k in train_datasets.keys()}
        dev_datasets['input_ids'] = npy_dict["input_ids"][dev_s_idx:dev_e_idx]
        dev_datasets['attention_mask'] = npy_dict["attention_mask"][dev_s_idx:dev_e_idx]
        dev_datasets['token_type_ids'] = npy_dict["token_type_ids"][dev_s_idx:dev_e_idx]
        dev_datasets['labels'] = npy_dict["labels"][dev_s_idx:dev_e_idx]
        print(f"[KoCharNpyMaker][_save_npy] Dev.shape\ninput_ids: {dev_datasets['input_ids'].shape}, "
              f"attention_mask: {dev_datasets['attention_mask'].shape}, "
              f"token_type_ids: {dev_datasets['token_type_ids'].shape}, "
              f"labels.shape: {dev_datasets['labels'].shape}")

        # Test
        test_datasets = {k: None for k in dev_datasets.keys()}
        test_datasets['input_ids'] = npy_dict["input_ids"][dev_e_idx:]
        test_datasets['attention_mask'] = npy_dict["attention_mask"][dev_e_idx:]
        test_datasets['token_type_ids'] = npy_dict["token_type_ids"][dev_e_idx:]
        test_datasets['labels'] = npy_dict["labels"][dev_e_idx:]
        print(f"[KoCharNpyMaker][_save_npy] Test.shape\ninput_ids: {test_datasets['input_ids'].shape}, "
              f"attention_mask: {test_datasets['attention_mask'].shape}, "
              f"token_type_ids: {test_datasets['token_type_ids'].shape}, "
              f"labels.shape: {test_datasets['labels'].shape}")

        # Save
        for train_key, train_val in train_datasets.items():
            np.save(save_path + '/train_' + train_key, train_val)

        for dev_key, dev_val in dev_datasets.items():
            np.save(save_path + '/dev_' + dev_key, dev_val)

        for test_key, test_val in test_datasets.items():
            np.save(save_path + '/test_' + test_key, test_val)

    def _tokenize_using_output_vocab(self, g2p_sent: str, tag2ids: Dict[str, int], max_seq_len: int):
        split_sent = list(g2p_sent)
        split_sent.insert(0, "[CLS]")

        if max_seq_len <= len(split_sent):
            split_sent = split_sent[:max_seq_len-1]
            split_sent.append("[SEP]")
        else:
            split_sent.append("[SEP]")
            diff_len = max_seq_len - len(split_sent)
            split_sent += ["[PAD]"] * diff_len

        b_is_use = True
        try:
            g2p_tokens = [tag2ids[x] for x in split_sent]
            g2p_tokens = np.array(g2p_tokens)
        except:
            g2p_tokens = None
            b_is_use = False

        return g2p_tokens, b_is_use

    def _make_eojeol_mecab_res(self, input_sent: str, mecab_res: List):
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
            curr_char_cnt += len(eojeol[0].strip()) # 에듀' '  <- Mecab 결과에 공백이 따라오는 경우 있음
            use_check[ej_idx] = True
        if 0 < len(eojeol_set):
            total_eojeol_morp.append(copy.deepcopy(eojeol_set))

        return total_eojeol_morp

### MAIN ###
if "__main__" == __name__:
    lstm_npy_maker = LstmEncDecNpyMaker(b_use_out_vocab=True, b_debug_mode=False)

    lstm_npy_maker. make_bilstm_lstm_npy(raw_path="../data/kor/pkl/kor_source_filter.pkl",
                                        g2p_path="../data/kor/pkl/kor_target.pkl",
                                        save_path="../data/kor/npy/lstm",
                                        out_vocab_path="../data/vocab/pron_eumjeol_vocab.json")
