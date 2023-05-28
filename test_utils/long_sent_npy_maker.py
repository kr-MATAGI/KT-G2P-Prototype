import pickle
import os
import numpy as np

from utils.kocharelectra_tokenization import KoCharElectraTokenizer


#======================================================================
def get_longest_sent(src_path: str, tgt_path: str):
#======================================================================
    print(f'[get_longest_sent] src_path: {src_path}, tgt_path :{tgt_path}')

    ret_longest_src = ''
    ret_longest_tgt = ''

    if not os.path.exists(src_path):
        raise Exception('ERR - src_path is not existed !')
    if not os.path.exists(tgt_path):
        raise Exception('ERR - tgt_path is not existed !')

    # Read SRC / TGT
    src_data = tgt_data = None
    with open(src_path, mode='rb') as s_f:
        src_data = pickle.load(s_f)
    with open(tgt_path, mode='rb') as t_f:
        tgt_data = pickle.load(t_f)
    print(f'[get_longest_sent] src_data.size: {len(src_data)}, tgt_data.size: {len(tgt_data)}')

    max_len = 0
    for idx, (src_item, tgt_item) in enumerate(zip(src_data, tgt_data)):
        if src_item.id != tgt_item.id:
            continue
        if len(src_item.sent) != len(tgt_item.sent):
            continue

        if max_len < len(src_item.sent):
            max_len = len(src_item.sent)
            ret_longest_src = src_item.sent
            ret_longest_tgt = tgt_item.sent

    print(f'[get_longest_sent] max_len: {max_len}')
    print(f'[get_longest_sent] ret_longest_src: {ret_longest_src}')
    print(f'[get_longest_sent] ret_longest_tgt: {ret_longest_tgt}')

    return ret_longest_src, ret_longest_tgt

#======================================================================
def make_longest_npy(src_sent: str, tgt_sent: str, save_path: str,
                     max_seq_len: int=256, data_size: int=6861):
#======================================================================
    print(f'[make_longest_npy] src_sent: {src_sent}')
    print(f'[make_longest_npy] tgt_sent: {tgt_sent}')
    print(f'[make_longest_npy] max_seq_len: {max_seq_len}')

    tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')

    src_tokens = tokenizer(src_sent, padding="max_length", max_length=max_seq_len)
    tgt_tokens = tokenizer(tgt_sent, padding="max_length", max_length=max_seq_len)

    input_ids = []
    attention_mask = []
    target = []
    for _ in range(data_size):
        input_ids.append(src_tokens['input_ids'])
        attention_mask.append(src_tokens['attention_mask'])
        target.append(tgt_tokens['input_ids'])

    # convert npy
    input_ids = np.array(input_ids)
    attention_mask = np.array(attention_mask)
    target = np.array(target)

    print(f'[make_longest_npy] input_ids.size: {input_ids.shape}')
    print(f'[make_longest_npy] attention_mask.size: {attention_mask.shape}')
    print(f'[make_longest_npy] target.size: {target.shape}')

    np.save(save_path + "/test_src_tokens", input_ids)
    # np.save(save_path + "/" + "test_src_lengths", src_lengths)
    np.save(save_path + "/test_target", target)
    # np.save(save_path + "/" + "test_prev_output_tokens", prev_output_tokens)
    np.save(save_path + "/test_attention_mask", attention_mask)

    print(f'[make_longest_npy] save_path: {save_path}')

### MAIN ###
if '__main__' == __name__:
    print('[long_sent_npy_maker] MAIN !')

    src_path = '../data/kor/pkl/kor_source_filter.pkl'
    tgt_path = '../data/kor/pkl/kor_target.pkl'

    longest_src, longest_tgt = get_longest_sent(src_path, tgt_path)

    save_path = '../data/test_npy'
    make_longest_npy(src_sent=longest_src, tgt_sent=longest_tgt,
                     save_path=save_path)