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
from definition.data_def import OurSamDebug

# Digits Converter
from KorDigits import Label2Num

### GLOBAL
logger = init_logger()

### MAIN ###
if '__main__' == __name__:
    logger.info(f'[run_digits_ensemble][__main__] START !')

    numeral = Label2Num()
    sentence = "2013년 05월 18일"
    output = numeral.generate(sentence)

    print(output)