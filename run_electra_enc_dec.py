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
from definition.data_def import DictWordItem, OurSamItem
from utils.post_method import (
    apply_our_sam_word_item, make_g2p_word_dictionary,
    save_our_sam_debug, re_evaluate_apply_dict
)

import time
from attrdict import AttrDict
from typing import Dict, List
from tqdm import tqdm
import evaluate as hug_eval

from run_utils import (
    load_npy_file, G2P_Dataset,
    init_logger, make_inputs_from_batch
)

import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux


### GLOBAL
logger = init_logger()


#========================================
def evaluate(args, model, tokenizer, eval_dataset, mode,
             output_vocab: Dict[str, int], our_sam_dict: Dict[str, List[str]], global_steps: str):
#========================================
    # init
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    mecab = Mecab()

    output_ids2tok = {v: k for k, v in output_vocab.items()}

    # Eval
    logger.info("***** Running evaluation on {} dataset *****".format(mode))

    eval_loss = 0.0
    nb_eval_steps = 0

    references = []
    candidates = []
    total_correct = 0

    input_sent_list = []

    wrong_case = {
        "input_sent": [],
        "pred_sent": [],
        "ans_sent": []
    }

    # Test Batch가 모두 끝나고 Decoding 되도록
    inputs_list = []
    pred_list = []
    ans_list = []

    # 우리말샘 기분석 사전을 통해 바뀐 문장 갯수
    all_our_sam_debug_info: List[OurSamItem] = []
    total_change_cnt = 0

    criterion = nn.CrossEntropyLoss()
    eval_pbar = tqdm(eval_dataloader)

    eval_start_time = time.time()
    cuda_starter = torch.cuda.Event(enable_timing=True)
    cuda_ender = torch.cuda.Event(enable_timing=True)
    cuda_times = []
    for batch in eval_pbar:
        model.eval()

        with torch.no_grad():
            inputs = make_inputs_from_batch(batch, device=args.device)
            inputs["mode"] = "eval"

            cuda_starter.record()
            logits = model(**inputs)  # predict [batch, seq_len] List
            cuda_ender.record()
            torch.cuda.synchronize()

            cuda_times.append(cuda_starter.elapsed_time(cuda_ender) / 1000)

            loss = criterion(logits.view(-1, args.out_vocab_size), inputs["labels"].view(-1).to(args.device))
            eval_loss += loss.mean().item()

            inputs_list.append(inputs["input_ids"].detach().cpu())
            pred_list.append(torch.argmax(logits, dim=-1).detach().cpu())
            ans_list.append(inputs["labels"].detach().cpu())

        nb_eval_steps += 1
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / nb_eval_steps))
    # end loop
    eval_end_time = time.time()

    ''' Decode '''
    for (input_item, pred_item, ans_item) in zip(inputs_list, pred_list, ans_list):
        for p_idx, (input_i, pred, lab) in enumerate(zip(input_item, pred_item, ans_item)):
            input_sent = tokenizer.decode(input_i)
            pred_sent = "".join([output_ids2tok[x] for x in pred.tolist()])
            ans_sent = "".join([output_ids2tok[x] for x in lab.tolist()])

            input_sent = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', input_sent)
            pred_sent = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', pred_sent)
            ans_sent = re.sub(r'\[CLS\]|\[SEP\]|\[PAD\]', '', ans_sent)

            ''' 우리말 샘 문자열-발음열 대치 '''
            ''' debug '''
            if args.use_our_sam:
                our_sam_res, is_change = apply_our_sam_word_item(our_sam_g2p_dict=our_sam_dict,
                                                                 mecab=mecab,
                                                                 input_sent=input_sent,
                                                                 pred_sent=pred_sent,
                                                                 ans_sent=ans_sent)
                if is_change:
                    pred_sent = our_sam_res.conv_sent
                    total_change_cnt += 1
                    all_our_sam_debug_info.append(our_sam_res)

            print(f"{p_idx}:\n"
                  f"input_sent: \n{input_sent}\n"
                  f"pred_sent: \n{pred_sent}\n"
                  f"ans_sent: \n{ans_sent}\n")

            references.append(ans_sent)
            candidates.append(pred_sent)

            if ans_sent == pred_sent:
                total_correct += 1
            else:
                wrong_case["input_sent"].append(input_sent)
                wrong_case["pred_sent"].append(pred_sent)
                wrong_case["ans_sent"].append(ans_sent)

            input_sent_list.append(input_sent)

    wer_score = hug_eval.load("wer").compute(predictions=candidates, references=references)
    per_score = hug_eval.load("cer").compute(predictions=candidates, references=references)
    print(f"[run_electra_enc_dec][evaluate] global_steps: {global_steps}")
    print(f"[run_electra_enc_dec][evaluate] wer_score: {wer_score * 100}, size: {len(candidates)}")
    print(f"[run_electra_enc_dec][evaluate] per_score: {per_score * 100}, size: {len(candidates)}")
    print(f"[run_electra_enc_dec][evaluate] s_acc: {total_correct/len(eval_dataset) * 100}, size: {total_correct}, "
          f"total.size: {len(eval_dataset)}")
    print(f"[run_electra_enc_dec][evaluate] Elapsed time: {eval_end_time - eval_start_time} seconds")
    print(f'[run_electra_enc_dec][evaluate] GPU Time: {sum(cuda_times)} seconds')
    print(f"[run_electra_enc_dec][evaluate] our_sam - total_change_cnt: {total_change_cnt}")

    eval_pbar.close()

    ''' 결과 저장 '''
    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_steps) if global_steps else "{}.txt".format(mode))

    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))

        f_w.write("  wer = {}\n".format(wer_score))
        f_w.write("  per = {}\n".format(per_score))
        f_w.write("  acc = {}\n".format(total_correct / len(eval_dataset)))
        f_w.write("  Elapsed time: {} seconds\n".format(eval_end_time - eval_start_time))
        f_w.write("  GPU time: {} seconds".format(sum(cuda_times)))

    ''' 최종 결과에서 틀린 문장들 저장 '''
    with open('./results/bilstm_lstm/wrong_case.txt', mode='w', encoding='utf-8') as w_f:
        for w_idx, (w_inp_s, w_candi_s, w_ref_s) in enumerate(zip(wrong_case['input_sent'],
                                                                  wrong_case['pred_sent'],
                                                                  wrong_case['ans_sent'])):
            w_f.write(str(w_idx)+'\n')
            w_f.write(w_inp_s+'\n')
            w_f.write(w_candi_s+'\n')
            w_f.write(w_ref_s+'\n')
            w_f.write('==================\n\n')

    ''' 우리말 사전 적용 결과 저장 '''
    if args.use_our_sam and args.our_sam_debug:
        save_our_sam_debug(all_item_save_path='./results/bilstm_lstm/our_sam_all.txt',
                           wrong_item_save_path='./results/bilstm_lstm/our_sam_wrong.txt',
                           our_sam_debug_list=all_our_sam_debug_info)
        print(f'[run_electra_enc_dec][evaluate] OurSamDebug info Save Complete !')

    if args.use_our_sam:
        re_evaluate_apply_dict(target_items=all_our_sam_debug_info,
                               input_sent_list=input_sent_list,
                               pred_sent_list=candidates,
                               ans_sent_list=references)

#========================================
def train(args, model, tokenizer, train_dataset, dev_dataset,
          output_vocab: Dict[str, int], our_sam_dict: Dict[str, str]):
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
                    evaluate(args, model, tokenizer, dev_dataset, "dev",
                             output_vocab, our_sam_dict, global_steps=global_step)

        logger.info("  Epoch Done= %d", epoch + 1)
        pbar.close()

    return global_step, tr_loss / global_step

#========================================
def main(config_path: str,
         decoder_vocab_path: str,
         jaso_post_proc_path: str,
         our_sam_path: str
 ):
#========================================
    # Check path
    print(f"[run_g2p][main] config_path: {config_path}\nout_vocab_path: {decoder_vocab_path}\n"
          f"jaso_post_proc_path: {jaso_post_proc_path}\nour_sam_path: {our_sam_path}")

    if not os.path.exists(config_path):
        raise Exception("ERR - Check config_path")
    if not os.path.exists(decoder_vocab_path):
        raise Exception("ERR - Check decoder_vocab_path")
    if not os.path.exists(jaso_post_proc_path):
        raise Exception("ERR - Check jaso_pos_proc_path")
    if not os.path.exists(our_sam_path):
        raise Exception("ERR - Check our_sam_path")

    # Read config file
    with open(config_path) as f:
        args = AttrDict(json.load(f))
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

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

    ''' 우리말 샘 문자열-발음열 사전 '''
    our_sam_dict: List[DictWordItem] = None
    with open(our_sam_path, mode="rb") as f:
        our_sam_dict = pickle.load(f)
        our_sam_dict = make_g2p_word_dictionary(our_sam_word_items=our_sam_dict)
    print(f"[run_g2p][main] our_sam_dict.size: {len(our_sam_dict)}")

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
    config.do_post_method = args.do_post_method

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

        global_step, tr_loss = train(args, model, tokenizer, train_datasets, dev_datasets,
                                     output_vocab=decoder_vocab, our_sam_dict=our_sam_dict)
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
            ckpt_config = ElectraConfig.from_pretrained(checkpoint)
            ckpt_config.do_post_method = config.do_post_method
            model = ElectraStdPronRules.from_pretrained(checkpoint, tokenizer=tokenizer, out_tag2ids=decoder_vocab,
                                                        out_ids2tag=decoder_ids2tag, jaso_pair_dict=post_proc_dict,
                                                        config=ckpt_config)
            model.to(args.device)
            evaluate(args, model, tokenizer, test_datasets, mode="test",
                     output_vocab=decoder_vocab, our_sam_dict=our_sam_dict, global_steps=global_step)

### MAIN ###
if "__main__" == __name__:
    print("[run_electra_enc_dec][__main__] MAIN !")

    main(config_path="config/electra_bilstm_lstm.json",
         decoder_vocab_path="./data/vocab/pron_eumjeol_vocab.json",
         jaso_post_proc_path="./data/post_method/jaso_filter.json",
         our_sam_path="./data/dictionary/filtered_dict_word_item.pkl")