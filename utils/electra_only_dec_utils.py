import json
import numpy as np
import torch
from typing import List, Dict

from definition.dict_def import Dictionary
from torch.utils.data import Dataset

from collections import namedtuple

DecoderOut = namedtuple(
    "FastCorrectDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history", "to_be_edited_pred", "wer_dur_pred"],
)

#===============================================================
def get_vocab_type_dictionary(vocab_path: str="", is_kochar_electra: bool=True, tokenizer=None):
#===============================================================
    ret_dict = None

    if ('.json' in vocab_path) and (not is_kochar_electra):
        with open(vocab_path, mode='r', encoding='utf-8') as f:
            lines = json.load(f)
            lines = [k for k, v in lines.items()]
            lines.remove('[PAD]')
            lines.remove('[UNK]')
            lines.remove('[CLS]')
            lines.remove('[SEP]')
            ret_dict = Dictionary(extra_special_symbols=lines)
    else:
        electra_vocab = tokenizer.vocab
        electra_vocab = list(electra_vocab)
        electra_vocab.remove('[PAD]')
        electra_vocab.remove('[UNK]')
        electra_vocab.remove('[CLS]')
        electra_vocab.remove('[SEP]')

        ret_dict = Dictionary(extra_special_symbols=electra_vocab)

    return ret_dict

#===============================================================
def load_electra_transformer_decoder_npy(src_path, mode: str):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    src_tokens = np.load(root_path + "_src_tokens.npy")
    src_lengths = np.load(root_path + "_src_lengths.npy")
    attn_mask = np.load(root_path + "_attention_mask.npy")
    prev_output_tokens = np.load(root_path + "_prev_output_tokens.npy")
    target = np.load(root_path + "_target.npy")

    print(f"[run_utils][load_bert_fused_npy] {mode} npy shape:")
    print(f"src_tokens: {src_tokens.shape}, attn_mask: {attn_mask.shape}")
    print(f"target: {target.shape}")

    inputs = {
        "src_tokens": src_tokens,
        "src_lengths": src_lengths,
        "attention_mask": attn_mask,
        "prev_output_tokens": prev_output_tokens,
        "target": target
    }

    return inputs

#===============================================================
class ElectraOnlyDecDataset(Dataset):
#===============================================================
    def __init__(
            self,
            item_dict: Dict[str, np.ndarray]
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.src_tokens = torch.LongTensor(item_dict["src_tokens"]).to(device)
        self.src_len = torch.LongTensor(item_dict["src_lengths"]).to(device)
        self.attn_mask = torch.LongTensor(item_dict["attention_mask"]).to(device)
        self.prev_output_tokens = torch.LongTensor(item_dict["prev_output_tokens"]).to(device)
        self.target = torch.LongTensor(item_dict["target"]).to(device)

        print(f'[ElectraOnlyDecDataset] src_tokens.size: {self.src_tokens.size()}')
        print(f'[ElectraOnlyDecDataset] src_len.size: {self.src_len.size()}')
        print(f'[ElectraOnlyDecDataset] attn_mask.size: {self.attn_mask.size()}')
        print(f'[ElectraOnlyDecDataset] target.size: {self.target.size()}')
        print(f'[ElectraOnlyDecDataset] prev_output_tokens.size: {self.prev_output_tokens.size()}')

        assert len(self.src_tokens) == len(self.src_len), 'ERR - src_len'
        assert len(self.src_tokens) == len(self.attn_mask), 'ERR - attn_mask'
        assert len(self.src_tokens) == len(self.target), 'ERR - target'
        assert len(self.src_tokens) == len(self.prev_output_tokens), 'ERR - prev_output_tokens'


    def __len__(self):
        return len(self.src_tokens)

    def __getitem__(self, idx):
        items = {
            "src_tokens": self.src_tokens[idx],
            "src_lengths": self.src_len[idx],
            "attention_mask": self.attn_mask[idx],
            "prev_output_tokens": self.prev_output_tokens[idx],
            "tgt_tokens": self.target[idx]
        }

        return items

#===============================================================
def make_electra_only_dec_inputs(batch: torch.Tensor):
#===============================================================
    inputs = {
        "src_tokens": batch["src_tokens"],
        "attention_mask": batch["attention_mask"],
        "src_lengths": batch["src_lengths"],
        "tgt_tokens": batch["tgt_tokens"],
        "prev_output_tokens": batch["prev_output_tokens"],
        "bert_input": batch["src_tokens"]
    }

    return inputs