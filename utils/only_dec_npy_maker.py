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
from definition.data_def import MECAB_POS_TAG

#========================================================
class OnlyDecNpyMaker:
#========================================================
    def __init__(self,
                 b_debug_mode: bool=False):
        print(f'[OnlyDecNpyMaker][__init__] b_debug_mode: {b_debug_mode}')

        self.b_debug_mode = b_debug_mode

    def make_only_dec_npy_maker(
            self,
            src_path: str, tgt_path: str,
            save_path: str, dec_vocab_path: str,
            max_seq_len: int=256
    ):
        pass

    def _tokenization(
            self,
            tokenizer_name: str,
            raw_data_list: List[KT_TTS], tgt_data_list: List[KT_TTS],
            dec_vocab, max_seq_len: int=256
    ):
        pass

    def _save_npy(
            self,
            npy_dict: Dict[str, List], save_path: str
    ):
        pass

### MAIN ###
if '__main__' == __name__:
    print('[only_dec_npy_maker][__main__] MAIN !')
