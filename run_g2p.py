import os
import json
import re
import glob

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
import pandas as pd

from run_utils import (
    load_npy_file, G2P_Dataset,
    init_logger, make_inputs_from_batch
)

### GLOBAL
logger = init_logger()

#========================================
def evaluate(args, model, tokenizer, eval_dataset, mode, output_vocab: Dict[str, int], global_step: str):
#========================================
    # init
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    output_ids2tok = {v: k for k, v in output_vocab.items()}

    # Eval
    logger.info("***** Running evaluation on {} dataset *****".format(mode))

    eval_loss = 0.0
    nb_eval_steps = 0

    references = []
    candidates = []
    total_correct = 0

    wrong_case = {
        "input_sent": [],
        "candi_sent": [],
        "ref_sent": []
    }

    # Test Batch가 모두 끝나고 Decoding 되도록
    inputs_list = []
    pred_list = []
    ans_list = []

    criterion = nn.CrossEntropyLoss()
    eval_pbar = tqdm(eval_dataloader)
    eval_start_time = time.time()

    for batch in eval_pbar:
        model.eval()

        with torch.no_grad():
            inputs = make_inputs_from_batch(batch, device=args.device)
            inputs["mode"] = "eval"

            logits = model(**inputs)  # predict [batch, seq_len] List
            loss = criterion(logits.view(-1, args.out_vocab_size), inputs["labels"].view(-1).to(args.device))
            eval_loss += loss.mean().item()

            inputs_list.append(inputs["input_ids"].detach().cpu())
            pred_list.append(torch.argmax(logits, dim=-1).detach().cpu())
            ans_list.append(inputs["labels"].detach().cpu())

        nb_eval_steps += 1
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))
    # end loop
    eval_end_time = time.time()

    for (input_item, pred_item, ans_item) in zip(inputs_list, pred_list, ans_list):
        for p_idx, (input_i, pred, lab) in enumerate(zip(input_item, pred_item, ans_item)):
            ref_str = "".join([output_ids2tok[x] for x in lab.tolist()])
            # ref_str = tokenizer.decode(lab.tolist())
            ref_str = ref_str.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

            candi_str = "".join([output_ids2tok[x] for x in pred.tolist()])
            # candi_str = tokenizer.decode(pred.tolist())
            candi_str = candi_str.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()

            raw_sent = tokenizer.decode(input_i)
            raw_sent = raw_sent.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
            print(f"{p_idx}:\nraw: \n{raw_sent}\ncandi: \n{candi_str}\nref: \n{ref_str}")

            references.append(ref_str)
            candidates.append(candi_str)
            if ref_str == candi_str:
                total_correct += 1
            else:
                wrong_case["input_sent"].append(raw_sent)
                wrong_case["candi_sent"].append(candi_str)
                wrong_case["ref_sent"].append(ref_str)

    wer_score = hug_eval.load("wer").compute(predictions=candidates, references=references)
    per_score = hug_eval.load("cer").compute(predictions=candidates, references=references)
    print(f"[run_g2p][evaluate] wer_score: {wer_score * 100}, size: {len(candidates)}")
    print(f"[run_g2p][evaluate] per_score: {per_score * 100}, size: {len(candidates)}")
    print(f"[run_g2p][evaluate] s_acc: {total_correct/len(eval_dataset) * 100}, size: {total_correct}, "
          f"total.size: {len(eval_dataset)}")
    print(f"[run_g2p][evaluate] Elapsed time: {eval_end_time - eval_start_time} seconds")

    eval_pbar.close()

#========================================
def train(args, model, tokenizer, train_dataset, dev_dataset, output_vocab: Dict[str, int]):
#========================================
    # init
    train_data_len = len(train_dataset)
    t_total = (train_data_len // args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # eps : 줄이기 전/후의 lr차이가 eps보다 작으면 무시한다.
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # @NOTE: optimizer에 설정된 learning_rate까지 선형으로 감소시킨다. (스케줄러)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    train_sampler = RandomSampler(train_dataset)

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        model.train()
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            inputs = make_inputs_from_batch(batch, device=args.device)
            inputs["mode"] = "train"

            logits = model(**inputs)
            loss = criterion(logits.view(-1, args.out_vocab_size), inputs["labels"].view(-1).to(args.device))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (len(train_dataloader) <= args.gradient_accumulation_steps and (step + 1) == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                pbar.set_description("Train Loss - %.04f" % (tr_loss / global_step))
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save samples checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving samples checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0) and \
                        args.evaluate_test_during_training:
                    evaluate(args, model, tokenizer, dev_dataset, "dev", output_vocab, global_step)

        logger.info("  Epoch Done= %d", epoch + 1)
        pbar.close()

    return global_step, tr_loss / global_step

#========================================
def main(config_path: str, decoder_vocab_path: str, jaso_post_proc_path: str):
#========================================
    # Check path
    print(f"[run_g2p][main] config_path: {config_path}\nout_vocab_path: {decoder_vocab_path}, "
          f"jaso_post_proc_path: {jaso_post_proc_path}")

    if not os.path.exists(config_path):
        raise Exception("ERR - Check config_path")
    if not os.path.exists(decoder_vocab_path):
        raise Exception("ERR - Check decoder_vocab_path")
    if not os.path.exists(jaso_post_proc_path):
        raise Exception("ERR - Check jaso_pos_proc_path")

    # Read config file
    with open(config_path) as f:
        args = AttrDict(json.load(f))

    if 0 < len(args.device) and ("cuda" == args.device or "cpu" == args.device):
        print(f"---- Config.Device: {args.device}")
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load decoder vocab
    decoder_vocab: Dict[str, int] = {}
    with open(decoder_vocab_path, mode="r", encoding="utf-8") as f:
        decoder_vocab = json.load(f)
        decoder_ids2tag = {v: k for k, v in decoder_vocab.items()}

    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(jaso_post_proc_path, mode="r", encoding="utf-8") as f:
        post_proc_dict = json.load(f)

    # Load model
    tokenizer = KoCharElectraTokenizer.from_pretrained(args.model_name_or_path)

    config = ElectraConfig.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.max_seq_len = args.max_seq_len

    config.pad_ids = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0] # 0
    config.unk_ids = tokenizer.convert_tokens_to_ids([tokenizer.unk_token])[0] # 1
    config.start_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0] # 2
    config.end_ids = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0] # 3
    config.mask_ids = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0] # 4
    config.gap_ids = tokenizer.convert_tokens_to_ids([' '])[0] # 5

    args.vocab_size = len(tokenizer)
    args.out_vocab_size = len(decoder_vocab.keys())
    config.vocab_size = args.vocab_size
    config.out_vocab_size = args.out_vocab_size

    model = ElectraStdPronRules.from_pretrained(args.model_name_or_path,
                                                config=config, tokenizer=tokenizer, out_tag2ids=decoder_vocab,
                                                out_ids2tag=decoder_ids2tag, jaso_pair_dict=post_proc_dict)
    model.to(args.device)

    # Do Train !
    if args.do_train:
        # Load npy
        train_inputs, train_labels = load_npy_file(args.train_npy, mode="train")
        dev_inputs, dev_labels = load_npy_file(args.dev_npy, mode="dev")

        # Make datasets
        train_datasets = G2P_Dataset(item_dict=train_inputs, labels=train_labels)
        dev_datasets = G2P_Dataset(item_dict=dev_inputs, labels=dev_labels)

        global_step, tr_loss = train(args, model, tokenizer, train_datasets, dev_datasets, output_vocab=decoder_vocab)
        logger.info(f'global_step = {global_step}, average loss = {tr_loss}')

    # Do Eval !
    if args.do_eval:
        # Load npy
        test_inputs, test_labels = load_npy_file(args.test_npy, mode="test")

        # Make datasets
        test_datasets = G2P_Dataset(item_dict=test_inputs, labels=test_labels)

        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logger.info("transformers.configuration_utils")
            logger.info("transformers.modeling_utils")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = ElectraStdPronRules.from_pretrained(checkpoint, tokenizer=tokenizer, out_tag2ids=decoder_vocab,
                                                        out_ids2tag=decoder_ids2tag, jaso_pair_dict=post_proc_dict)
            model.to(args.device)
            evaluate(args, model, tokenizer, test_datasets, mode="test",
                     output_vocab=decoder_vocab, global_step=global_step)

### MAIN ###
if "__main__" == __name__:
    print("[run_g2p][__main__] MAIN !")

    main(config_path="./config/kocharelectra_config.json",
         decoder_vocab_path="./data/vocab/decoder_vocab/pron_eumjeol_vocab.json",
         jaso_post_proc_path="./data/vocab/post_process/jaso_filter.json")