import torch
from torch.utils.data import Dataset

import logging

import random
import numpy as np

from typing import Dict, Optional, List

#===============================================================
def init_logger():
# ===============================================================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

#===============================================================
def set_seed(seed):
#===============================================================
    torch.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#===============================================================
def load_npy_file(src_path: str, mode: str):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    input_ids = np.load(root_path + "_input_ids.npy")
    attention_mask = np.load(root_path + "_attention_mask.npy")
    token_type_ids = np.load(root_path + "_token_type_ids.npy")
    labels = np.load(root_path + "_labels.npy")
    pos_ids = np.load(root_path + "_pos_ids.npy")

    print(f"[run_utils][load_npy_file] {mode}.npy.shape:")
    print(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, "
          f"token_type_ids: {token_type_ids.shape}, pos_ids: {pos_ids.shape}")

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask, "token_type_ids": token_type_ids,
        "pos_ids": pos_ids
    }
    return inputs, labels

#===============================================================
class G2P_Dataset(Dataset):
#===============================================================
    def __init__(
            self,
            item_dict: Dict[str, np.ndarray],
            labels: np.ndarray,
    ):
        self.input_ids = torch.tensor(item_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(item_dict["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(item_dict["token_type_ids"], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.pos_ids = torch.tensor(item_dict["pos_ids"], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
            "pos_ids": self.pos_ids[idx]
        }

        return items

#===============================================================
def make_inputs_from_batch(batch: torch.Tensor, device: str):
#===============================================================
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device),
        "labels": batch["labels"].to(device)
    }

    return inputs