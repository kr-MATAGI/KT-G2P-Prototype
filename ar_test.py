import torch
import torch.nn as nn

import json
import argparse
from attrdict import AttrDict
from typing import Dict, List

from model.electra_std_pron_rule import ElectraStdPronRules
from utils.kocharelectra_tokenization import KoCharElectraTokenizer

### MAIN ###
if "__main__" == __name__:
    print("[ar_test][__main__] MAIN !")

    parser = argparse.ArgumentParser(description="AR_TEST description")

    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--decoder_vocab_path", required=True)
    parser.add_argument("--jaso_dict_path", required=True)

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

    # load decoder_vocab
    decoder_vocab: Dict[str, int] = {}
    with open(cli_args.decoder_vocab_path, mode="r", encoding="utf-8") as f:
        decoder_vocab = json.load(f)
        out_ids2tag = {v: k for k, v in decoder_vocab.items()}
    print(f"[ar_test][__main__] output_vocab:\n{list(decoder_vocab.items())[:10]}")

    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(cli_args.jaso_dict_path, mode="r", encoding="utf-8") as f:
        post_proc_dict = json.load(f)

    # Do Test
    # tokenization
    target_sent = cli_args.input
    output_ids2tok = {v: k for k, v in decoder_vocab.items()}
    inputs = tokenizer(target_sent, padding="max_length", max_length=256, return_tensors="pt",
                       truncation=True)

    # load model
    target_ckpt_path = cli_args.ckpt_path
    print(f"[ar_test][__main__] target_ckpt:\n{target_ckpt_path}")

    model = ElectraStdPronRules.from_pretrained(target_ckpt_path, tokenizer=tokenizer, out_tag2ids=decoder_vocab,
                                                out_ids2tag=out_ids2tag, jaso_pair_dict=post_proc_dict)
    model.to(device)
    model.eval()

    output = model(inputs["input_ids"].to(device),
                   inputs["attention_mask"].to(device),
                   inputs["token_type_ids"].to(device), mode="eval")

    candi_str = "".join([output_ids2tok[x] for x in torch.argmax(output, dim=-1)[0].tolist()])
    candi_str = candi_str.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

    print(f"[ar_test][__main__] raw:\n{target_sent}")
    print(f"[ar_test][__main__] candi:\n{candi_str}")


