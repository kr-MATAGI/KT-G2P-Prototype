import os
import json
import re
import glob
import copy
import pickle

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from utils.kocharelectra_tokenization import KoCharElectraTokenizer
from transformers import ElectraConfig, get_linear_schedule_with_warmup
from model.electra_std_pron_rule import ElectraStdPronRules

import time
from attrdict import AttrDict
from typing import Dict, List
from tqdm import tqdm
import evaluate as hug_eval

from run_utils import (
    load_npy_file, G2P_Dataset,
    init_logger, make_inputs_from_batch,
    make_eojeol_mecab_res
)

### OurSam Dict
import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux

# Digits Converter
from KorDigits import Label2Num

### GLOBAL
logger = init_logger()
numeral_model = Label2Num()


#===============================================================
def main(
        config_path: str, decode_vocab_path: str,
        jaso_post_path: str, our_sam_path: str
):
#===============================================================
    print(f'[run_digits_ensemble][main] config_path: {config_path}')
    print(f'[run_digits_ensemble][main] decode_vocab_path: {decode_vocab_path}')
    print(f'[run_digits_ensemble][main] jso_post_path: {jaso_post_path}')
    print(f'[run_digits_ensemble][main] our_sam_path: {our_sam_path}')

    if not os.path.exists(config_path):
        raise Exception(f'ERR - config_path: {config_path}')
    if not os.path.exists(decode_vocab_path):
        raise Exception(f'ERR - decode_vocab_path: {decode_vocab_path}')
    if not os.path.exists(jaso_post_path):
        raise Exception(f'ERR - jaso_post_path: {jaso_post_path}')
    if not os.path.exists(our_sam_path):
        raise Exception(f'ERR - our_sam_path: {our_sam_path}')

    # Read config
    with open(config_path) as f:
        args = AttrDict(json.load(f))
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    if 0 < len(args.device) and ('cuda' == args.device or 'cpu' == args.device):
        print(f'[run_digits_ensemble][main] Config.Device: {args.device}')
    else:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read decode vocab
    decode_vocab: Dict[str, int] = {}
    with open(decode_vocab_path, mode='r', encoding='utf-8') as f:
        decode_vocab = json.load(f)
        decode_ids2tag = {v: k for k, v in decode_vocab.items()}
    print(f'[run_digits_ensemble][main] decode_vocab.size: {len(decode_vocab.keys())}')

    # Read post_method_dict
    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(jaso_post_path, mode='r', encoding='utf-8') as f:
        post_proc_dict = json.load(f)
    print(f'[run_digits_ensemble][main] post_proc_dict.size: {len(post_proc_dict.keys())}')

    ''' 우리말 샘 문자열-발음열 사전 '''
    our_sam_dict: None
    with open(our_sam_path, mode='rb') as f:
        our_sam_dict = pickle.load(f)
    print(f'[run_digits_ensemble][main] our_sam_dict.size: {len(our_sam_dict.keys())}')

    # Load Model
    tokenizer = KoCharElectraTokenizer.from_pretrained(args.model_name_or_path)

    config = ElectraConfig.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.max_seq_len = args.max_seq_len

    config.pad_ids = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]  # 0
    config.unk_ids = tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0]  # 1
    config.start_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]  # 2
    config.end_ids = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]  # 3
    config.mask_ids = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]  # 4
    config.gap_ids = tokenizer.convert_tokens_to_ids([' '])[0]  # 5

    config.vocab_size = args.vocab_size = len(tokenizer)
    config.out_vocab_size = args.out_vocab_size = len(decode_vocab.keys())
    config.do_post_method = args.do_post_method

    model = ElectraStdPronRules.from_pretrained(
        args.model_name_or_path,
        config=config, tokenizer=tokenizer, out_tag2ids=decode_vocab,
        out_ids2tag=decode_ids2tag, jaso_pair_dict=post_proc_dict
    )
    model.to(args.device)

    # Do Train
    if args.do_train:
        pass

    # Do Eval
    if args.do_eval:
        pass

### MAIN ###
if '__main__' == __name__:
    logger.info(f'[run_digits_ensemble][__main__] START !')

    main(
        config_path='./config/digits_ensemble_config.json',
        decode_vocab_path='./data/vocab/pron_eumjeol_vocab.json',
        jaso_post_path='./data/post_method/jaso_filter.json',
        our_sam_path='./data/dictionary/our_sam_std_dict.pkl'
    )
