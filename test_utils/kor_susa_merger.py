import copy
import os
import re
import json
import numpy as np

from typing import List, Dict
from utils.kocharelectra_tokenization import KoCharElectraTokenizer

#=================================================================
class KorSusaDataMerger:
#=================================================================
    def __init__(
            self,
            tokenizer_name: str,
            max_seq_len: int
    ):
        print(f'[KorSusaDataMerger][__init__] tokenizer_name: {tokenizer_name}, max_seq_len: {max_seq_len}')
        self.tokenizer = KoCharElectraTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len

    def make_merged_kor_susa_npy_for_lstm(
        self,
        kor_npy_dir_path: str,
        susa_dir_path: str,
        b_use_custom_vocab: True,
        custom_vocab_path: str,
        npy_save_path: str
    ):
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] kor_npy_dir_path: {kor_npy_dir_path}\n'
              f'susa_dir_path: {susa_dir_path}\n'
              f'b_use_custom_vocab: {b_use_custom_vocab}\ncustom_vocab_path: {custom_vocab_path}')

        if not os.path.exists(kor_npy_dir_path):
            raise Exception('ERR - Not Existed kor_npy_dir_path')
        if not os.path.exists(susa_dir_path):
            raise Exception('ERR - Not Existed susa_dir_path')
        if b_use_custom_vocab and not os.path.exists(custom_vocab_path):
            raise Exception('ERR - Not Existed custom_vocab_path')

        train_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'train' in x]
        dev_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'dev' in x]
        test_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'test' in x]
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_kor_files:\n{train_kor_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_kor_files:\n{dev_kor_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_kor_files:\n{test_kor_files}')

        train_susa_files = [x for x in os.listdir(susa_dir_path) if 'train' in x]
        dev_susa_files = [x for x in os.listdir(susa_dir_path) if 'dev' in x]
        test_susa_files = [x for x in os.listdir(susa_dir_path) if 'test' in x]
        train_susa_files = sorted(train_susa_files) # 0: source, 1: target
        dev_susa_files = sorted(dev_susa_files)
        test_susa_files = sorted(test_susa_files)
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_susa_files:\n{train_susa_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_susa_files:\n{dev_susa_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_susa_files:\n{test_susa_files}')

        custom_vocab = None
        if b_use_custom_vocab:
            with open(custom_vocab_path, mode='r', encoding='utf-8') as cv_f:
                custom_vocab = json.load(cv_f)
                print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] custom_vocab.size: {len(custom_vocab)}')
                print(list(custom_vocab.items())[:10])

        # Load Kor *.npy Files
        train_kor_npy_dict = {
            'input_ids': None,
            'attention_mask': None,
            'token_type_ids': None,
            'labels': None
        }
        dev_kor_npy_dict = copy.deepcopy(train_kor_npy_dict)
        test_kor_npy_dict = copy.deepcopy(train_kor_npy_dict)

        for key in train_kor_npy_dict.keys():
            train_kor_npy_dict[key] = np.load(kor_npy_dir_path+'/train_'+key+'.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_{key}.shape: {train_kor_npy_dict[key].shape}')
        for key in dev_kor_npy_dict.keys():
            dev_kor_npy_dict[key] = np.load(kor_npy_dir_path+'/dev_'+key+'.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_{key}.shape: {dev_kor_npy_dict[key].shape}')
        for key in test_kor_npy_dict.keys():
            test_kor_npy_dict[key] = np.load(kor_npy_dir_path + '/test_' + key + '.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_{key}.shape: {test_kor_npy_dict[key].shape}')

        # Read susa train/dev/test txt files
        train_susa_src_data, train_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=train_susa_files, mode='train'
        )
        dev_susa_src_data, dev_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=dev_susa_files, mode='dev'
        )
        test_susa_src_data, test_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=test_susa_files, mode='test'
        )

        # Susa Tokenization
        train_susa_npy_dict = self._lstm_susa_tokenization(susa_src_data=train_susa_src_data,
                                                           susa_tgt_data=train_susa_tgt_data,
                                                           custom_vocab=custom_vocab, mode='train')
        dev_susa_npy_dict = self._lstm_susa_tokenization(susa_src_data=dev_susa_src_data,
                                                         susa_tgt_data=dev_susa_tgt_data,
                                                         custom_vocab=custom_vocab, mode='dev')
        test_susa_npy_dict = self._lstm_susa_tokenization(susa_src_data=test_susa_src_data,
                                                          susa_tgt_data=test_susa_tgt_data,
                                                          custom_vocab=custom_vocab, mode='test')

        # Merge kor + susa
        self._save_merged_npy(kor_npy_dict=train_kor_npy_dict,
                              susa_npy_dict=train_susa_npy_dict,
                              save_path=npy_save_path, mode='train')
        self._save_merged_npy(kor_npy_dict=dev_kor_npy_dict,
                              susa_npy_dict=dev_susa_npy_dict,
                              save_path=npy_save_path, mode='dev')
        self._save_merged_npy(kor_npy_dict=test_kor_npy_dict,
                              susa_npy_dict=test_susa_npy_dict,
                              save_path=npy_save_path, mode='test')

    def _load_susa_texts(
            self,
            susa_dir_path: str,
            target_files: List,
            mode: str
    ):
        susa_src_data = None
        susa_tgt_data = None
        with open(susa_dir_path + '/' + target_files[0], mode='r', encoding='utf-8') as f:
            susa_src_data = f.readlines()
            susa_src_data = [x.replace('\n', '').split('\t') for x in susa_src_data]
        with open(susa_dir_path + '/' + target_files[1], mode='r', encoding='utf-8') as f:
            susa_tgt_data = f.readlines()
            susa_tgt_data = [x.replace('\n', '').split('\t') for x in susa_tgt_data]
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] {mode}_susa.size - src: {len(susa_src_data)}, '
              f'tgt:{len(susa_tgt_data)}')
        assert len(susa_src_data) == len(susa_tgt_data), f'ERR - diff size susa {mode} files'
        print(susa_src_data[:10])
        print(susa_tgt_data[:10])

        susa_src_data = [x[1] for x in susa_src_data]
        susa_tgt_data = [x[1] for x in susa_tgt_data]
        return susa_src_data, susa_tgt_data

    def _lstm_susa_tokenization(
        self,
        susa_src_data: List, susa_tgt_data: List, mode: str,
        custom_vocab: Dict
    ):
        npy_dict = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }

        print(f'[KorSusaDataMerger][_susa_tokenization] mode: {mode}')
        for idx, (src_sent, tgt_sent) in enumerate(zip(susa_src_data, susa_tgt_data)):
            if 0 == (idx % 1000):
                print(f'[KorSusaDataMerger][_susa_tokenization] {idx} is processing... {src_sent}')

            src_sent = re.sub(r'[^가-힣\s]+', '', src_sent)
            src_tokens = self.tokenizer(src_sent, padding='max_length', max_length=self.max_seq_len,
                                        return_tensors='np', truncation=True)

            tgt_tokens = [custom_vocab['[CLS]']] + [custom_vocab[x] for x in list(tgt_sent)] + [custom_vocab['[SEP]']]
            if self.max_seq_len <= len(tgt_tokens):
                tgt_tokens = tgt_tokens[:self.max_seq_len-1]
                tgt_tokens.append(custom_vocab['[SEP]'])
            else:
                diff_size = self.max_seq_len - len(tgt_tokens)
                tgt_tokens += [custom_vocab['[PAD]']] * diff_size
            assert self.max_seq_len == len(tgt_tokens), f'ERR - tgt_tokens.size: {len(tgt_tokens)}'

            npy_dict['input_ids'].append(src_tokens['input_ids'][0])
            npy_dict['attention_mask'].append(src_tokens['attention_mask'][0])
            npy_dict['token_type_ids'].append(src_tokens['token_type_ids'][0])
            npy_dict['labels'].append(tgt_tokens)
        # end loop

        # convert list to np
        for key, val in npy_dict.items():
            npy_dict[key] = np.array(val)
            print(f'[KorSusaDataMerger][_susa_tokenization] {mode}.{key}.size: {npy_dict[key].shape}')

        return npy_dict

    def _save_merged_npy(
        self,
        kor_npy_dict: Dict,
        susa_npy_dict: Dict,
        save_path: str, mode: str
    ):
        merge_npy_dict = {}
        for key, val in susa_npy_dict.items():
            if 'length' in key:
                merge_npy_dict[key] = np.hstack((kor_npy_dict[key], susa_npy_dict[key]))
            else:
                merge_npy_dict[key] = np.vstack((kor_npy_dict[key], susa_npy_dict[key]))
            print(f'[KorSusaDataMerger][_save_merged_npy] {mode}.{key}.shape: {merge_npy_dict[key].shape}')

        for key, val in merge_npy_dict.items():
            np.save(save_path+'/'+mode+"_"+key, val)
        print(f'[KorSusaDataMerger][_save_merged_npy] mode: {mode}, save_path: {save_path}')

    def make_merged_kor_susa_npy_for_only_dec(
        self,
        kor_npy_dir_path: str,
        susa_dir_path: str,
        npy_save_path: str
    ):
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy_for_only_dec] kor_npy_dir_path: {kor_npy_dir_path}\n'
              f'susa_dir_path: {susa_dir_path}')

        if not os.path.exists(kor_npy_dir_path):
            raise Exception('ERR - Not Existed kor_npy_dir_path')
        if not os.path.exists(susa_dir_path):
            raise Exception('ERR - Not Existed susa_dir_path')

        train_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'train' in x]
        dev_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'dev' in x]
        test_kor_files = [x for x in os.listdir(kor_npy_dir_path) if 'test' in x]
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_kor_files:\n{train_kor_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_kor_files:\n{dev_kor_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_kor_files:\n{test_kor_files}')

        train_susa_files = [x for x in os.listdir(susa_dir_path) if 'train' in x]
        dev_susa_files = [x for x in os.listdir(susa_dir_path) if 'dev' in x]
        test_susa_files = [x for x in os.listdir(susa_dir_path) if 'test' in x]
        train_susa_files = sorted(train_susa_files)  # 0: source, 1: target
        dev_susa_files = sorted(dev_susa_files)
        test_susa_files = sorted(test_susa_files)
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_susa_files:\n{train_susa_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_susa_files:\n{dev_susa_files}')
        print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_susa_files:\n{test_susa_files}')

        # Load Kor *.npy Files
        train_kor_npy_dict = {
            'src_tokens': [],
            'src_lengths': [],
            'attention_mask': [],
            'token_type_ids': [],
            'prev_output_tokens': [],
            'target': []
        }
        dev_kor_npy_dict = copy.deepcopy(train_kor_npy_dict)
        test_kor_npy_dict = copy.deepcopy(train_kor_npy_dict)

        for key in train_kor_npy_dict.keys():
            train_kor_npy_dict[key] = np.load(kor_npy_dir_path + '/train_' + key + '.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] train_{key}.shape: {train_kor_npy_dict[key].shape}')
        for key in dev_kor_npy_dict.keys():
            dev_kor_npy_dict[key] = np.load(kor_npy_dir_path + '/dev_' + key + '.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] dev_{key}.shape: {dev_kor_npy_dict[key].shape}')
        for key in test_kor_npy_dict.keys():
            test_kor_npy_dict[key] = np.load(kor_npy_dir_path + '/test_' + key + '.npy')
            print(f'[KorSusaDataMerger][make_merged_kor_susa_npy] test_{key}.shape: {test_kor_npy_dict[key].shape}')

        # Read susa train/dev/test txt files
        train_susa_src_data, train_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=train_susa_files, mode='train'
        )
        dev_susa_src_data, dev_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=dev_susa_files, mode='dev'
        )
        test_susa_src_data, test_susa_tgt_data = self._load_susa_texts(
            susa_dir_path=susa_dir_path, target_files=test_susa_files, mode='test'
        )

        # Susa Tokenization
        train_susa_npy_dict = self._only_dec_susa_tokenization(susa_src_data=train_susa_src_data,
                                                               susa_tgt_data=train_susa_tgt_data,
                                                               mode='train')
        dev_susa_npy_dict = self._only_dec_susa_tokenization(susa_src_data=dev_susa_src_data,
                                                             susa_tgt_data=dev_susa_tgt_data,
                                                             mode='dev')
        test_susa_npy_dict = self._only_dec_susa_tokenization(susa_src_data=test_susa_src_data,
                                                              susa_tgt_data=test_susa_tgt_data,
                                                              mode='test')

        # Merge kor + susa
        self._save_merged_npy(kor_npy_dict=train_kor_npy_dict,
                              susa_npy_dict=train_susa_npy_dict,
                              save_path=npy_save_path, mode='train')
        self._save_merged_npy(kor_npy_dict=dev_kor_npy_dict,
                              susa_npy_dict=dev_susa_npy_dict,
                              save_path=npy_save_path, mode='dev')
        self._save_merged_npy(kor_npy_dict=test_kor_npy_dict,
                              susa_npy_dict=test_susa_npy_dict,
                              save_path=npy_save_path, mode='test')

    def _only_dec_susa_tokenization(
        self,
        susa_src_data: List, susa_tgt_data: List, mode: str
    ):
        npy_dict = {
            'src_tokens': [],
            'src_lengths': [],
            'attention_mask': [],
            'token_type_ids': [],
            'prev_output_tokens': [],
            'target': []
        }

        print(f'[KorSusaDataMerger][_only_dec_susa_tokenization] mode: {mode}')
        for idx, (src_sent, tgt_sent) in enumerate(zip(susa_src_data, susa_tgt_data)):
            if 0 == (idx % 1000):
                print(f'[KorSusaDataMerger][_only_dec_susa_tokenization] {idx} is processing... {src_sent}')

            src_sent = re.sub(r'[^가-힣\s]+', '', src_sent)
            src_tokens = self.tokenizer(src_sent, padding='max_length', max_length=self.max_seq_len,
                                        return_tensors='np', truncation=True)

            cls_idx = np.where(src_tokens['input_ids'][0] == self.tokenizer.encode('[CLS]')[1])[0][0]
            sep_idx = np.where(src_tokens['input_ids'][0] == self.tokenizer.encode('[SEP]')[1])[0][0]
            src_lengths = len(src_tokens['input_ids'][0][cls_idx:sep_idx + 1])

            tgt_tokens = self.tokenizer(tgt_sent, padding='max_length', max_length=self.max_seq_len,
                                        return_tensors='np', truncation=True)

            npy_dict['src_tokens'].append(src_tokens['input_ids'][0])
            npy_dict['src_lengths'].append(src_lengths)
            npy_dict['attention_mask'].append(src_tokens['attention_mask'][0])
            npy_dict['token_type_ids'].append(src_tokens['token_type_ids'][0])
            npy_dict['prev_output_tokens'].append(src_tokens['input_ids'][0])
            npy_dict['target'].append(tgt_tokens['input_ids'][0])
        # end loop

        # convert list to np
        for key, val in npy_dict.items():
            npy_dict[key] = np.array(val)
            print(f'[KorSusaDataMerger][_only_dec_susa_tokenization] {mode}.{key}.size: {npy_dict[key].shape}')

        return npy_dict

### MAIN ###
if '__main__' == __name__:
    print('[kor_susa_merger][__main__] MAIN !')

    kor_susa_merger = KorSusaDataMerger(tokenizer_name='monologg/kocharelectra-base-discriminator',
                                        max_seq_len=256)

    b_make_lstm = True
    b_make_only_dec = True
    if b_make_lstm:
        kor_susa_merger.make_merged_kor_susa_npy_for_lstm(
            kor_npy_dir_path='../data/kor/npy/lstm',
            susa_dir_path='../data/digits',
            b_use_custom_vocab=True,
            custom_vocab_path='../data/vocab/pron_eumjeol_vocab.json',
            npy_save_path='../data/susa_kor/npy/lstm'
        )
    if b_make_only_dec:
        kor_susa_merger.make_merged_kor_susa_npy_for_only_dec(
            kor_npy_dir_path='../data/kor/npy/only_dec',
            susa_dir_path='../data/digits',
            npy_save_path='../data/susa_kor/npy/only_dec'
        )