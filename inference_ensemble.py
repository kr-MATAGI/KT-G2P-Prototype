import torch
import torch.nn as nn
import pickle

from torch.utils.data import DataLoader
import json
import argparse
from attrdict import AttrDict
from typing import Dict, List
import numpy as np
from definition.data_def import KT_TTS

from model.electra_std_pron_rule import ElectraStdPronRules
from utils.kocharelectra_tokenization import KoCharElectraTokenizer
from model.electra_nart_pos_dec_model import ElectraNartPosDecModel
from utils.electra_only_dec_utils import (
    get_vocab_type_dictionary, make_electra_only_dec_inputs, ElectraOnlyDecDataset)
from utils.post_method import (
    make_g2p_word_dictionary, save_our_sam_debug, apply_our_sam_word_item, re_evaluate_apply_dict
)

# Digits Converter
from KorDigits import Label2Num
from utils.kt_tts_pkl_maker import KT_TTS_Maker
from utils.english_to_korean import Eng2Kor
from tqdm import tqdm

### OurSam Dict
import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux


### MAIN ###
if "__main__" == __name__:
    print("[ar_test][__main__] MAIN !")

    parser = argparse.ArgumentParser(description="AR_TEST description")

    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--decoder_vocab_path", required=True)
    parser.add_argument("--jaso_dict_path", required=True)
    parser.add_argument("--our_sam_path", required=True)
    parser.add_argument("--max_seq_len", required=True)
    cli_args = parser.parse_args()

    # Init
    config_path = cli_args.config_path
    with open(config_path) as config_file:
        args = AttrDict(json.load(config_file))
    if 0 < len(args.device) and ("cuda" == args.device or "cpu" == args.device):
        print(f"---- Config.Device: {args.device}")
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

    # load src_vocab
    src_vocab = get_vocab_type_dictionary(tokenizer=tokenizer, is_kochar_electra=True)
    # load decoder_vocab
    decoder_vocab = get_vocab_type_dictionary(cli_args.decoder_vocab_path, is_kochar_electra=False)

    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(cli_args.jaso_dict_path, mode="r", encoding="utf-8") as f:
        post_proc_dict = json.load(f)

    ''' 우리말 샘 문자열-발음열 사전 '''
    our_sam_dict = {}
    with open(cli_args.our_sam_path, mode='rb') as f:
        our_sam_dict = pickle.load(f)
        our_sam_dict = make_g2p_word_dictionary(our_sam_word_items=our_sam_dict)

    # Do Test
    # tokenization
    target_sent = cli_args.input
    # output_ids2tok = {v: k for k, v in decoder_vocab.items()}

    # load model
    target_ckpt_path = cli_args.ckpt_path
    print(f"[ar_test][__main__] target_ckpt:\n{target_ckpt_path}")

    model = ElectraNartPosDecModel.build_model(args=args, tokenizer=tokenizer,
                                               src_vocab=src_vocab, dec_vocab=decoder_vocab,
                                               post_proc_dict=post_proc_dict)
    model.load_state_dict(torch.load(cli_args.ckpt_path + '/model.pt'))

    model.to(device)
    model.eval()

    # Tokenization
    ret_dict = {
        'src_tokens': [],
        'src_lengths': [],
        'attention_mask': [],
        'prev_output_tokens': [],
        'target': []
    }

    ''' Convert num2kor '''
    num2kor = Label2Num(mecab=Mecab())
    target_sent = num2kor.generate(target_sent)

    ''' Convert sym2kor '''
    sym2kor = KT_TTS_Maker()
    target = KT_TTS(id='0', sent=target_sent)
    target = sym2kor.get_converted_symbol_items(target)

    ''' Convert Eng2Kor '''
    eng2kor = Eng2Kor()
    target = eng2kor.convert_eng(target)

    ''' src token processing '''
    src_tokens = tokenizer(target.sent, padding='max_length', max_length=int(cli_args.max_seq_len),
                           return_tensors='np', truncation=True)
    cls_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[CLS]')[1])[0][0]
    sep_idx = np.where(src_tokens['input_ids'][0] == tokenizer.encode('[SEP]')[1])[0][0]
    src_lengths = len(src_tokens['input_ids'][0][cls_idx:sep_idx + 1])

    ''' tgt token processing '''
    tgt_tokens = [decoder_vocab.index('[CLS]')] + [decoder_vocab.index(x) for x in list(target.sent)] \
                 + [decoder_vocab.index('[SEP]')]
    if int(cli_args.max_seq_len) <= len(tgt_tokens):
        tgt_tokens = tgt_tokens[:int(cli_args.max_seq_len) - 1]
        tgt_tokens.append(decoder_vocab.index('[SEP]'))
    else:
        diff_size = int(cli_args.max_seq_len) - len(tgt_tokens)
        tgt_tokens += [decoder_vocab.index('[PAD]')] * diff_size

    ret_dict['src_tokens'].append(src_tokens['input_ids'][0])
    ret_dict['src_lengths'].append(src_lengths)
    ret_dict['attention_mask'].append(src_tokens['attention_mask'][0])
    ret_dict['prev_output_tokens'].append(src_tokens['input_ids'][0])
    ret_dict['target'].append(tgt_tokens)

    # convert list to np
    for key, val in ret_dict.items():
        ret_dict[key] = torch.LongTensor(np.array(val))

    target = ElectraOnlyDecDataset(args, ret_dict)
    target = DataLoader(target, batch_size=1)

    for batch in tqdm(target):
        inputs = make_electra_only_dec_inputs(batch)
        inputs['mode'] = 'eval'

        output = model(**inputs)

    candi_tok = torch.argmax(output, -1).detach().cpu()
    candi_str = "".join([decoder_vocab[x] for x in candi_tok.tolist()[0]]).strip()
    candi_str = candi_str.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

    print(f"[ar_test][__main__] raw:\n{target_sent}")
    print(f"[ar_test][__main__] candi:\n{candi_str}")


