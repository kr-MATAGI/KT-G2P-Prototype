import os
import re
import pickle
import json
import numpy as np

from typing import List, Dict
from definition.data_def import KT_TTS, WordInfo
from utils.kocharelectra_tokenization import KoCharElectraTokenizer

#=================================================================
class KorEngDataMerger:
#=================================================================
    def __init__(self):
        print(f'[KorEngDataMerger][__init__] INIT !')

    def get_converted_eng_text(
            self,
            raw_src_path: str, raw_tgt_path: str,
            eng_dict_path: str
    ):
        ret_sent_list = [] # [ (id, raw_src_sent, conv_src_sent, tgt_sent), ... ]

        print(f'[KorEngDataMerger][get_converted_eng_text] raw_src_path: {raw_src_path}')
        print(f'[KorEngDataMerger][get_converted_eng_text] raw_tgt_path: {raw_tgt_path}')
        print(f'[KorEngDataMerger][get_converted_eng_text] eng_dict_path: {eng_dict_path}')

        if not os.path.exists(raw_src_path):
            raise Exception('ERR - raw_src_path NOT Existed !')
        if not os.path.exists(raw_tgt_path):
            raise Exception('ERR - raw_tgt_path NOT Existed !')
        if not os.path.exists(eng_dict_path):
            raise Exception('ERR - eng_dict_path NOT Existed !')

        raw_src_data: List[KT_TTS] = []
        raw_tgt_data: List[KT_TTS] = []
        eng_pkl: List[WordInfo] = []
        eng_dict: Dict[str, str] = {}
        with open(raw_src_path, mode='rb') as src_f:
            raw_src_data = pickle.load(src_f)
            print(f'[KorEngDataMerger][get_converted_eng_text] raw_src_data.size: {len(raw_src_data)}')
            print(raw_src_data[:10])
        with open(raw_tgt_path, mode='rb') as tgt_f:
            raw_tgt_data = pickle.load(tgt_f)
            print(f'[KorEngDataMerger][get_converted_eng_text] raw_tgt_data.size: {len(raw_tgt_data)}')
            print(raw_tgt_data[:10])
        with open(eng_dict_path, mode='rb') as eng_f:
            eng_pkl = pickle.load(eng_f)
            print(f'[KorEngDataMerger][get_converted_eng_text] eng_pkl.size: {len(eng_pkl)}')
            print(eng_pkl[:10])

            eng_dict = {}
            for eng_pkl_item in eng_pkl:
                key = eng_pkl_item.word
                val = eng_pkl_item.pronunciation
                if key in eng_dict.keys():
                    continue
                else:
                    eng_dict[key] = val
            print(f'[KorEngDataMerger][get_converted_eng_text] eng_dict.size: {len(eng_dict)}')
            print(list(eng_dict.items())[:10])

        skip_cnt = 0
        for raw_idx, (src_item, tgt_item) in enumerate(zip(raw_src_data, raw_tgt_data)):
            if src_item.id != tgt_item.id:
                skip_cnt += 1
                continue

            b_include_eng = False
            split_src = src_item.sent.split(' ')
            for sp_s_idx, src_eojeol in enumerate(split_src):
                eng_word_list = re.findall(r'[a-zA-Z]+', src_eojeol)
                if 0 < len(eng_word_list):
                    for src_eng in eng_word_list:
                        if src_eng in eng_dict.keys():
                            b_include_eng = True
                            conv_val = eng_dict[src_eng]
                            split_src[sp_s_idx] = split_src[sp_s_idx].replace(src_eng, conv_val)
                            '''
                                For Debug
                                e.g.
                                [('000020', '캐릭터는 게메뉴 홈 화면에서만 볼 수 있어요', '캐릭터는 지메뉴 홈 화며네서만 볼 쑤 이써요') 
                            '''
                            # if split_src[sp_s_idx] != tgt_item.sent.split(' ')[sp_s_idx]:
                            #     print(split_src[sp_s_idx], tgt_item.sent.split(' ')[sp_s_idx])
                            #     input()

            converted_src = " ".join(split_src)
            if b_include_eng:
                conv_src_eumjeol = list(converted_src)
                only_hangul_sent = ''
                for eumjeol in conv_src_eumjeol:
                    if ' ' == eumjeol:
                        only_hangul_sent += eumjeol
                    elif re.match(r'^[가-힣]', eumjeol):
                        only_hangul_sent += eumjeol
                # print(src_item.sent)
                # print(converted_src)
                # print(only_hangul_sent)
                # print(tgt_item.sent)
                # input()

                if len(only_hangul_sent) == len(tgt_item.sent):
                    ret_sent_list.append((src_item.id, src_item.sent, only_hangul_sent, tgt_item.sent))

        print(f'[KorEngDataMerger][get_converted_eng_text] skip_cnt: {skip_cnt}')
        print(f'[KorEngDataMerger][get_converted_eng_text] ret_sent_list.size: {len(ret_sent_list)}')
        print(ret_sent_list[:10])

        return ret_sent_list

    def make_nart_npy(
            self,
            data_path: str, save_path: str, ori_test_npy_path: str,
            b_use_custom_vocab: bool, custom_vocab_path: str,
            b_stack_ori_npy: bool,
            max_seq_len: int=256
    ):
        print(f'[KorEngDataMerger][make_nart_npy] data_path: {data_path}\nsave_path: {save_path}')
        print(f'[KorEngDataMerger][make_nart_npy] ori_test_npy_path: {ori_test_npy_path}')

        if not os.path.exists(data_path):
            raise Exception(f'ERR - Not Existed: {data_path}')
        if not os.path.exists(ori_test_npy_path):
            raise Exception(f'ERR - Not Existed: {ori_test_npy_path}')
        if b_use_custom_vocab and not os.path.exists(custom_vocab_path):
            raise Exception(f'ERR - Not Existed: {custom_vocab_path}')

        custom_vocab = None
        if b_use_custom_vocab:
            with open(custom_vocab_path, mode='r', encoding='utf-8') as cv_f:
                custom_vocab = json.load(cv_f)
                print(f'[KorEngDataMerger][make_nart_npy] custom_vocab.size: {len(custom_vocab)}')
                print(list(custom_vocab.items())[:10])

        eng_sent_data = []
        with open(data_path, mode='rb') as d_f:
            eng_sent_data = pickle.load(d_f)
        print(f'[KorEngDataMerger][make_nart_npy] eng_sent_data.size: {len(eng_sent_data)}')

        # Make *.npy files
        tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')

        # Load origin test npy files
        ori_src_tokens_np = []
        ori_src_lengths_np = []
        ori_attn_mask_np = []
        ori_token_type_ids_np = []
        ori_prev_output_tokens_np = []
        ori_target_np = []

        if b_stack_ori_npy:
            ori_src_tokens_np = np.load(ori_test_npy_path + '/test_src_tokens.npy')
            ori_src_lengths_np = np.load(ori_test_npy_path + '/test_src_lengths.npy')
            ori_attn_mask_np = np.load(ori_test_npy_path + '/test_attention_mask.npy')
            ori_token_type_ids_np = np.load(ori_test_npy_path + '/test_token_type_ids.npy')
            ori_prev_output_tokens_np = np.load(ori_test_npy_path + '/test_prev_output_tokens.npy')
            ori_target_np = np.load(ori_test_npy_path + '/test_target.npy')

            print(f'[KorEngDataMerger][make_nart_npy] ori_test_npy files shape:')
            print(f'src_tokens: {ori_src_tokens_np.shape}, src_lengths: {ori_src_lengths_np.shape}')
            print(f'ori_attn_mask_np: {ori_attn_mask_np.shape}, ori_token_type_ids_np: {ori_token_type_ids_np.shape}')
            print(f'ori_prev_output_tokens_np: {ori_prev_output_tokens_np.shape}, ori_target_np: {ori_target_np.shape}')

        # Tokenization
        for idx, (id, _, conv_src_sent, tgt_sent) in enumerate(eng_sent_data):
            if 0 == (idx % 100):
                print(f'[KorEngDataMerger][make_nart_npy] {idx} is processing...')

            conv_src_res = tokenizer(conv_src_sent,
                                        padding="max_length", max_length=max_seq_len,
                                        return_tensors='np', truncation=True)
            tgt_res = tokenizer(tgt_sent,
                                 padding='max_length', max_length=max_seq_len,
                                 return_tensors='np', truncation=True)

            src_token = conv_src_res['input_ids'][0]

            cls_idx = np.where(src_token == tokenizer.encode('[CLS]')[1])[0][0]
            sep_idx = np.where(src_token == tokenizer.encode('[SEP]')[1])[0][0]
            src_len = np.array([len(src_token[cls_idx:sep_idx+1])])

            attn_mask = conv_src_res['attention_mask'][0]
            token_type_ids = conv_src_res['token_type_ids'][0]
            target = tgt_res['input_ids'][0]

            if b_stack_ori_npy:
                ori_src_tokens_np = np.vstack((ori_src_tokens_np, src_token))
                ori_src_lengths_np = np.hstack((ori_src_lengths_np, src_len))
                ori_attn_mask_np = np.vstack((ori_attn_mask_np, attn_mask))
                ori_token_type_ids_np = np.vstack((ori_token_type_ids_np, token_type_ids))
                ori_prev_output_tokens_np = np.vstack((ori_prev_output_tokens_np, src_token))
                ori_target_np = np.vstack((ori_target_np, target))
            else:
                ori_src_tokens_np.append(src_token)
                ori_src_lengths_np.append(src_len)
                ori_attn_mask_np.append(attn_mask)
                ori_token_type_ids_np.append(token_type_ids)
                ori_prev_output_tokens_np.append(src_token)
                ori_target_np.append(target)
        # end loop, Tokenization

        if not b_stack_ori_npy:
            ori_src_tokens_np = np.array(ori_src_tokens_np)
            ori_src_lengths_np = np.array(ori_src_lengths_np)
            ori_attn_mask_np = np.array(ori_attn_mask_np)
            ori_token_type_ids_np = np.array(ori_token_type_ids_np)
            ori_prev_output_tokens_np = np.array(ori_prev_output_tokens_np)
            ori_target_np = np.array(ori_target_np)

        print(f'[KorEngDataMerger][make_nart_npy] Complete eng tokens stack processing...')
        print(f'src_tokens: {ori_src_tokens_np.shape}, src_lengths: {ori_src_lengths_np.shape}')
        print(f'ori_attn_mask_np: {ori_attn_mask_np.shape}, ori_token_type_ids_np: {ori_token_type_ids_np.shape}')
        print(f'ori_prev_output_tokens_np: {ori_prev_output_tokens_np.shape}, ori_target_np: {ori_target_np.shape}')

        # Save
        np.save(save_path + '/test_src_tokens', ori_src_tokens_np)
        np.save(save_path + '/test_src_lengths', ori_src_lengths_np)
        np.save(save_path + '/test_attention_mask', ori_attn_mask_np)
        np.save(save_path + '/test_token_type_ids', ori_token_type_ids_np)
        np.save(save_path + '/test_prev_output_tokens', ori_prev_output_tokens_np)
        np.save(save_path + '/test_target', ori_target_np)
        print(f'[KorEngDataMerger][make_nart_npy] Save Npy Files - {save_path}')

    def make_lstm_npy(
            self,
            data_path: str, save_path: str, ori_test_npy_path: str,
            b_use_custom_vocab: bool, custom_vocab_path: str, b_stack_ori_npy: bool,
            max_seq_len: int = 256
    ):
        print(f'[KorEngDataMerger][make_lstm_npy] data_path: {data_path}\nsave_path: {save_path}')
        print(f'[KorEngDataMerger][make_lstm_npy] ori_test_npy_path: {ori_test_npy_path}')

        if not os.path.exists(data_path):
            raise Exception(f'ERR - Not Existed: {data_path}')
        if not os.path.exists(ori_test_npy_path):
            raise Exception(f'ERR - Not Existed: {ori_test_npy_path}')
        if b_use_custom_vocab and not os.path.exists(custom_vocab_path):
            raise Exception(f'ERR - Not Existed: {custom_vocab_path}')

        custom_vocab = None
        if b_use_custom_vocab:
            with open(custom_vocab_path, mode='r', encoding='utf-8') as cv_f:
                custom_vocab = json.load(cv_f)
                print(f'[KorEngDataMerger][make_lstm_npy] custom_vocab.size: {len(custom_vocab)}')
                print(list(custom_vocab.items())[:10])

        eng_sent_data = []
        with open(data_path, mode='rb') as d_f:
            eng_sent_data = pickle.load(d_f)
        print(f'[KorEngDataMerger][make_lstm_npy] eng_sent_data.size: {len(eng_sent_data)}')

        # Make *.npy files
        tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')

        # Load origin test npy files
        ori_input_ids_np = []
        ori_attn_mask_np = []
        ori_token_type_ids_np = []
        ori_labels_np = []
        if b_stack_ori_npy:
            ori_input_ids_np = np.load(ori_test_npy_path + '/test_input_ids.npy')
            ori_attn_mask_np = np.load(ori_test_npy_path + '/test_attention_mask.npy')
            ori_token_type_ids_np = np.load(ori_test_npy_path + '/test_token_type_ids.npy')
            ori_labels_np = np.load(ori_test_npy_path + '/test_labels.npy')

            print(f'[KorEngDataMerger][make_lstm_npy] ori_test_npy files shape:')
            print(f'ori_input_ids_np: {ori_input_ids_np.shape}, ori_labels_np: {ori_labels_np.shape}')
            print(f'ori_attn_mask_np: {ori_attn_mask_np.shape}, ori_token_type_ids_np: {ori_token_type_ids_np.shape}')

        # Tokenization
        for idx, (id, _, conv_src_sent, tgt_sent) in enumerate(eng_sent_data):
            conv_src_res = tokenizer(conv_src_sent,
                                     padding="max_length", max_length=max_seq_len,
                                     return_tensors='np', truncation=True)
            input_ids = conv_src_res['input_ids'][0]
            attn_mask = conv_src_res['attention_mask'][0]
            token_type_ids = conv_src_res['token_type_ids'][0]

            labels = [custom_vocab[x] for x in list(tgt_sent)]
            labels.insert(0, custom_vocab['[CLS]'])
            if max_seq_len <= len(labels):
                labels = labels[:max_seq_len - 1]
                labels.append(custom_vocab['[SEP]'])
            else:
                labels.append(custom_vocab['[SEP]'])
                labels += [custom_vocab['[PAD]']] * (max_seq_len - len(labels))

            assert max_seq_len == len(labels), f'ERR - labels.size: {len(labels)}'

            if b_stack_ori_npy:
                ori_input_ids_np = np.vstack((ori_input_ids_np, input_ids))
                ori_attn_mask_np = np.vstack((ori_attn_mask_np, attn_mask))
                ori_token_type_ids_np = np.vstack((ori_token_type_ids_np, token_type_ids))
                ori_labels_np = np.vstack((ori_labels_np, labels))
            else:
                ori_input_ids_np.append(input_ids)
                ori_attn_mask_np.append(attn_mask)
                ori_token_type_ids_np.append(token_type_ids)
                ori_labels_np.append(labels)
        # end loop, Tokenization

        if not b_stack_ori_npy:
            ori_input_ids_np = np.array(ori_input_ids_np)
            ori_attn_mask_np = np.array(ori_attn_mask_np)
            ori_token_type_ids_np = np.array(ori_token_type_ids_np)
            ori_labels_np = np.array(ori_labels_np)

        print(f'[KorEngDataMerger][make_lstm_npy] Complete eng tokens stack processing...')
        print(f'ori_input_ids_np: {ori_input_ids_np.shape}, ori_labels_np: {ori_labels_np.shape}')
        print(f'ori_attn_mask_np: {ori_attn_mask_np.shape}, ori_token_type_ids_np: {ori_token_type_ids_np.shape}')

        # Save
        np.save(save_path + '/test_input_ids', ori_input_ids_np)
        np.save(save_path + '/test_attention_mask', ori_attn_mask_np)
        np.save(save_path + '/test_token_type_ids', ori_token_type_ids_np)
        np.save(save_path + '/test_labels', ori_labels_np)
        print(f'[KorEngDataMerger][make_lstm_npy] Save Npy files - {save_path}')

### MAIN ###
if '__main__' == __name__:
    print(f'[kor_eng_merger][__main__] MAIN !')

    kor_eng_data_merger = KorEngDataMerger()
    conv_eng2kor_sent_list = kor_eng_data_merger.get_converted_eng_text(
        raw_src_path='../data/tts_script_85ks_ms949_200407.pkl',
        raw_tgt_path='../data/tts_script_85ks_ms949_200506_g2p.pkl',
        eng_dict_path='../data/dictionary/dictionary.pkl'
    )

    # Save
    with open('../debug/eng2kor/debug_eng2kor_sent.txt', mode='w', encoding='utf-8') as w_f:
        for idx, (id, raw_src, conv_src, tgt) in enumerate(conv_eng2kor_sent_list):
            w_f.write(id + '\n' + raw_src + '\n' + conv_src + '\n' + tgt + '\n')
            w_f.write('====================================\n\n')

    save_path = '../data/eng_kor/pkl/eng2kor_sent.pkl'
    with open(save_path, mode='wb') as p_f:
        pickle.dump(conv_eng2kor_sent_list, p_f)
        print(f'[kor_eng_merger][__main__] Save Complete - {save_path}')

    # make to npy
    kor_eng_data_merger.make_nart_npy(
        data_path='../data/eng_kor/pkl/eng2kor_sent.pkl',
        save_path='../data/eng_kor/npy/only_dec',
        ori_test_npy_path='../data/kor/npy/only_dec',
        b_use_custom_vocab=False, custom_vocab_path='../data/vocab/pron_eumjeol_vocab.json',
        b_stack_ori_npy=False,
        max_seq_len=256
    )

    kor_eng_data_merger.make_lstm_npy(
        data_path='../data/eng_kor/pkl/eng2kor_sent.pkl',
        save_path='../data/eng_kor/npy/lstm',
        ori_test_npy_path='../data/kor/npy/lstm',
        b_use_custom_vocab=True, custom_vocab_path='../data/vocab/pron_eumjeol_vocab.json',
        b_stack_ori_npy=False,
        max_seq_len=256
    )