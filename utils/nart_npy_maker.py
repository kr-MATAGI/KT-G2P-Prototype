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
from utils.error_fixer import ERR_SENT_ID_FIXED, ERR_SENT_CHANGED_FIXED

import platform
if 'Windows' == platform.system():
    from eunjeon import MeCab
else:
    from konlpy.tag import Mecab


#========================================================
class NartNpyMaker:
#========================================================
    def __init__(
        self,
        b_debug_mode: bool=False,
        b_use_custom_vocab: bool=False
    ):
        print(f'[NartNpyMaker][__init__] b_debug_mode: {b_debug_mode}, b_use_custom_vocab: {b_use_custom_vocab}')

        self.b_debug_mode = b_debug_mode
        self.b_use_custom_vocab = b_use_custom_vocab

    def make_nart_npy(
            self,
            src_path: str, tgt_path: str, save_path: str,
            custom_vocab_path: str, max_seq_len: int=256
    ):
        print(f'[NartNpyMaker][make_nart_npy] src_path: {src_path}\ntgt_path:{tgt_path}'
              f'\nsave_path: {save_path}\ncustom_vocab_path: {custom_vocab_path}\nmax_seq_len: {max_seq_len}')

        if not os.path.exists(src_path):
            raise Exception('Not Existed -', src_path)
        if not os.path.exists(tgt_path):
            raise Exception('Not Existed -', tgt_path)
        if not os.path.exists(custom_vocab_path) and self.b_use_custom_vocab:
            raise Exception('Not Existed -', custom_vocab_path)

        # Load src/tgt_data
        all_src_data: List[KT_TTS] = []
        all_tgt_data: List[KT_TTS] = []
        with open(src_path, mode='rb') as s_f:
            all_src_data = pickle.load(s_f)
        with open(tgt_path, mode='rb') as t_f:
            all_tgt_data = pickle.load(t_f)
        print(f'[NartNpyMaker][make_nart_npy] all_src_data.size: {len(all_src_data)}, '
              f'all_tgt_data.size: {len(all_tgt_data)}')
        assert len(all_src_data) == len(all_tgt_data), 'ERR - Diff Size !'

        # Load Custom Vocab (for decoding)
        decoder_vocab: Dict[str, int] = None
        if self.b_use_custom_vocab:
            with open(custom_vocab_path, mode='r', encoding='utf-8') as d_f:
                decoder_vocab = json.load(d_f)
            print(f'[NartNpyMaker][make_nart_npy] decoder_vocab.size: {len(decoder_vocab.keys())}')
            print(list(decoder_vocab.items())[:10])

        # Tokenization
        npy_dict = self._tokenization(
            tokenizer_name='monologg/kocharelectra-base-discriminator',
            src_data_list=all_src_data, tgt_data_list=all_tgt_data,
            decoder_vocab=decoder_vocab, max_seq_len=max_seq_len
        )

        # Save to npy files
        self._save_npy(npy_dict=npy_dict, save_path=save_path)

    def _tokenization(
            self,
            tokenizer_name: str,
            src_data_list: List[KT_TTS], tgt_data_list: List[KT_TTS],
            decoder_vocab: Dict[str, int], max_seq_len: int
    ):
        # init
        npy_dict = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'src_lengths': [],
            'prev_output_tokens': [],
            'labels': []
        }
        except_output_pron_list = []  # (음절) 발음열을 따로 사용할 경우에만 데이터가 채워짐

        tokenizer = KoCharElectraTokenizer.from_pretrained(tokenizer_name)

        max_eojeol_size = 0
        max_sent_size = 0
        total_sent_size = 0

        for r_idx, (src_data, tgt_data) in enumerate(zip(src_data_list, tgt_data_list)):
            src_data.sent = src_data.sent.strip()
            tgt_data.sent = tgt_data.sent.strip()

            if src_data.id in ERR_SENT_CHANGED_FIXED.keys():
                print(f'[NartNpyMaker][_tokenization] ERR Sent\ninput:\n{src_data.sent}\nans:\n{tgt_data.sent}')
                print(f'[NartNpyMaker][_tokenization] \n{ERR_SENT_CHANGED_FIXED[src_data.id][0]} ->'
                      f'\n{ERR_SENT_CHANGED_FIXED[src_data.id][1]}')
                src_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][0]
                tgt_data.sent = ERR_SENT_CHANGED_FIXED[src_data.id][1]

            if not re.match(r"[가-힣]+", src_data.sent):
                except_output_pron_list.append((src_data.id, src_data.sent, tgt_data.sent))
                continue
            if re.search(r"[ㄱ-ㅎ]+", src_data.sent) or re.search(r'[ㅏ-ㅣ]+', src_data.sent):
                except_output_pron_list.append((src_data.id, src_data.sent, tgt_data.sent))
                continue

            if src_data.id != tgt_data.id:
                raise Exception(f"ERR - ID diff, raw_data: {src_data.id}, g2p_data: {tgt_data.id}")

            # 2023.03.23에 추가 (반사,라고 -> 반사라고, 모델예측: 반사라고 답: 반사 라고)
            if len(src_data.sent) != len(tgt_data.sent):
                except_output_pron_list.append((src_data.id, src_data.sent, tgt_data.sent))
                continue

            if 0 == (r_idx % 5000):
                print(f'[nart_npy_maker][_tokenization] {r_idx} is processing...')

            total_sent_size += len(src_data.sent)
            if max_sent_size < len(src_data.sent):
                sent_max_len = len(src_data.sent)
            if max_eojeol_size < len(src_data.sent.split(" ")):
                max_eojeol_size = len(src_data.sent.split(" "))

            # Tokenization src_data
            src_tokens = tokenizer(src_data.sent, padding='max_length', max_length=max_seq_len,
                                   return_tensors='np', truncation=True)

            # Tokenization tgt_data
            if tgt_data.id in ERR_SENT_ID_FIXED.keys():
                print(f'[NartNpyMaker][_tokenization] ERR DETECTED -> {tgt_data.id}, {tgt_data.sent}')
                print(f'[NartNpyMaker][_tokenization] {ERR_SENT_ID_FIXED[tgt_data.id]}')
                tgt_data.sent = tgt_data.sent.replace(ERR_SENT_ID_FIXED[tgt_data.id][0],
                                                      ERR_SENT_ID_FIXED[tgt_data.id][1])
                print(f'[NartNpyMaker][_tokenization] ERR FIXED -> {tgt_data.sent}')

            '''
            for t_idx, tgt_eumjeol in enumerate(tgt_data.sent):
                tgt_jaso = list(split_syllables(tgt_eumjeol))
                if 3 == len(tgt_jaso) and ('ㅅ' == tgt_jaso[-1] or 'ㅆ' == tgt_jaso[-1]):
                    print(f'[NartNpyMaker][_tokenization] Convert - {tgt_data.sent}')
                    print(f'[NartNpyMaker][_tokenization] {tgt_eumjeol} -> ')
                    tgt_jaso[-1] = 'ㄷ'
                    tgt_data.sent = tgt_data.sent.replace(tgt_eumjeol, join_jamos(''.join(tgt_jaso)))
                    print(f'[NartNpyMaker][_tokenization] {join_jamos("".join(tgt_jaso))}')
            '''

            if self.b_use_custom_vocab:
                split_tgt = list(tgt_data.sent)
                split_tgt.insert(0, '[CLS]')

                if max_seq_len <= len(split_tgt):
                    split_tgt = split_tgt[:max_seq_len-1]
                    split_tgt.append('[SEP]')
                else:
                    split_tgt.append('[SEP]')
                    split_tgt += ['[PAD]'] * (max_seq_len - len(split_tgt))
                convert_tgt_tokens = [decoder_vocab[x] for x in split_tgt]
                npy_dict['labels'].append(convert_tgt_tokens)
                assert max_seq_len == len(convert_tgt_tokens), 'ERR - tgt_tokens.size'
            else:
                tgt_tokens = tokenizer(tgt_data.sent, padding='max_length', max_length=max_seq_len,
                                       return_tensors='np', truncation=True)
                assert max_seq_len == len(tgt_tokens['input_ids'][0]), 'ERR - tgt_tokens.size'
                npy_dict['labels'].append(tgt_tokens['input_ids'][0])

            # Check Size
            assert max_seq_len == len(src_tokens['input_ids'][0]), 'ERR - input_ids.size'
            assert max_seq_len == len(src_tokens['attention_mask'][0]), 'ERR - attention_mask.size'
            assert max_seq_len == len(src_tokens['token_type_ids'][0]), 'ERR - token_type_ids.size'

            # Make src_lengths
            cls_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[CLS]')[1])[0][0]
            sep_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[SEP]')[1])[0][0]
            src_lengths = len(src_tokens['input_ids'][0][cls_idx:sep_idx + 1])

            # Insert to npy_dict
            npy_dict['input_ids'].append(src_tokens['input_ids'][0])
            npy_dict['attention_mask'].append(src_tokens['attention_mask'][0])
            npy_dict['token_type_ids'].append(src_tokens['token_type_ids'][0])
            npy_dict['src_lengths'].append(src_lengths)
            npy_dict['prev_output_tokens'].append(src_tokens['input_ids'][0])

        # (음절) 발음열 사전을 사용할 경우 예외에 대한 출력
        print("[NartNpyMaker][_tokenization] (음절) 발음열 사전 사용시 오류가 발생한 문장")
        for e_idx, except_item in enumerate(except_output_pron_list):
            print(e_idx, except_item)

        print(f"[NartNpyMaker][_tokenization] sent_max_len: {max_sent_size}, "
              f"mean_sent_len: {total_sent_size / len(src_data_list)}")

        print(f"[NartNpyMaker][_tokenization] max_eojeol_size: {max_eojeol_size}, sent_max_len: {sent_max_len}")

        return npy_dict

    def _save_npy(self, npy_dict: Dict[str, List], save_path: str):
        total_size = len(npy_dict['input_ids'])
        print(f'[NartNpyMaker][_save_npy] total_size: {total_size}, save_path: {save_path}')

        split_ratio = 0.1
        dev_s_idx = int(total_size * (split_ratio * 8))
        dev_e_idx = int(dev_s_idx + (total_size * split_ratio))
        print(f"[NartNpyMaker][_save_npy] split_ratio: {split_ratio}, "
              f"dev_s_idx: {dev_s_idx}, dev_e_idx: {dev_e_idx}")

        npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
        npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
        npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
        npy_dict["labels"] = np.array(npy_dict["labels"])
        npy_dict['src_lengths'] = np.array(npy_dict['src_lengths'])
        npy_dict['prev_output_tokens'] = np.array(npy_dict['prev_output_tokens'])

        # Train
        train_datasets = {
            'src_tokens': None,
            'attention_mask': None,
            'token_type_ids': None,
            'target': None,
            'src_lengths': None,
            'prev_output_tokens': None,
        }
        train_datasets['src_tokens'] = npy_dict["input_ids"][:dev_s_idx]
        train_datasets['attention_mask'] = npy_dict["attention_mask"][:dev_s_idx]
        train_datasets['token_type_ids'] = npy_dict["token_type_ids"][:dev_s_idx]
        train_datasets['target'] = npy_dict["labels"][:dev_s_idx]
        train_datasets['src_lengths'] = npy_dict['src_lengths'][:dev_s_idx]
        train_datasets['prev_output_tokens'] = npy_dict['prev_output_tokens'][:dev_s_idx]

        print(f"[NartNpyMaker][_save_npy] Train.shape\ninput_ids: {train_datasets['src_tokens'].shape}, "
              f"attention_mask: {train_datasets['attention_mask'].shape}, "
              f"token_type_ids: {train_datasets['token_type_ids'].shape}, "
              f"labels.shape: {train_datasets['target'].shape}, "
              f"src_lengths: {train_datasets['src_lengths'].shape},\n"
              f"prev_output_tokens: {train_datasets['prev_output_tokens'].shape}")

        # Dev
        dev_datasets = {k: [] for k in train_datasets.keys()}
        dev_datasets['src_tokens'] = npy_dict["input_ids"][dev_s_idx:dev_e_idx]
        dev_datasets['attention_mask'] = npy_dict["attention_mask"][dev_s_idx:dev_e_idx]
        dev_datasets['token_type_ids'] = npy_dict["token_type_ids"][dev_s_idx:dev_e_idx]
        dev_datasets['target'] = npy_dict["labels"][dev_s_idx:dev_e_idx]
        dev_datasets['src_lengths'] = npy_dict['src_lengths'][dev_s_idx:dev_e_idx]
        dev_datasets['prev_output_tokens'] = npy_dict['prev_output_tokens'][dev_s_idx:dev_e_idx]

        print(f"[NartNpyMaker][_save_npy] Dev.shape\ninput_ids: {dev_datasets['src_tokens'].shape}, "
              f"attention_mask: {dev_datasets['attention_mask'].shape}, "
              f"token_type_ids: {dev_datasets['token_type_ids'].shape}, "
              f"labels.shape: {dev_datasets['target'].shape}, src_lengths: {dev_datasets['src_lengths']},\n"
              f"prev_output_tokens: {dev_datasets['prev_output_tokens'].shape}")

        # Test
        test_datasets = {k: [] for k in dev_datasets.keys()}
        test_datasets['src_tokens'] = npy_dict["input_ids"][dev_e_idx:]
        test_datasets['attention_mask'] = npy_dict["attention_mask"][dev_e_idx:]
        test_datasets['token_type_ids'] = npy_dict["token_type_ids"][dev_e_idx:]
        test_datasets['target'] = npy_dict["labels"][dev_e_idx:]
        test_datasets['src_lengths'] = npy_dict['src_lengths'][dev_e_idx:]
        test_datasets['prev_output_tokens'] = npy_dict['prev_output_tokens'][dev_e_idx:]

        print(f"[NartNpyMaker][_save_npy] Test.shape\ninput_ids: {test_datasets['src_tokens'].shape}, "
              f"attention_mask: {test_datasets['attention_mask'].shape}, "
              f"token_type_ids: {test_datasets['token_type_ids'].shape}, "
              f"labels.shape: {test_datasets['target'].shape}, "
              f"src_lengths.shape: {test_datasets['src_lengths'].shape}, "
              f"prev_output_tokens: {test_datasets['prev_output_tokens'].shape}")

        # Save
        for train_key, train_val in train_datasets.items():
            np.save(save_path + '/train_' + train_key, train_val)

        for dev_key, dev_val in dev_datasets.items():
            np.save(save_path + '/dev_' + dev_key, dev_val)

        for test_key, test_val in test_datasets.items():
            np.save(save_path + '/test_' + test_key, test_val)

### MAIN ###
if '__main__' == __name__:
    print(f'[nart_npy_maker][__main__] MAIN !')

    nart_npy_maker = NartNpyMaker(b_debug_mode=False, b_use_custom_vocab=False)
    nart_npy_maker.make_nart_npy(
        src_path='../data/kor/pkl/kor_source_filter.pkl',
        tgt_path='../data/kor/pkl/kor_target.pkl',
        save_path='../data/susa_kor/npy/only_dec',
        custom_vocab_path='../data/vocab/pron_eumjeol_vocab.json'
    )