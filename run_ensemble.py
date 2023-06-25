import os
import json
import re
import glob
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from utils.kocharelectra_tokenization import KoCharElectraTokenizer
from transformers import ElectraConfig, get_linear_schedule_with_warmup
from model.electra_nart_pos_dec_model import ElectraNartPosDecModel
from definition.data_def import OurSamItem

import time
from attrdict import AttrDict
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import evaluate as hug_eval

from run_utils import (
    init_logger, print_args, set_seed, make_digits_ensemble_data
)

from utils.post_method import (
    make_g2p_word_dictionary, save_our_sam_debug, apply_our_sam_word_item
)
from utils.electra_only_dec_utils import (
    get_vocab_type_dictionary, load_electra_transformer_decoder_npy,
    ElectraOnlyDecDataset, make_electra_only_dec_inputs
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
mecab = Mecab()
numeral_model = Label2Num(mecab)
tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')

#===============================================================
def evaluate(args, model, eval_datasets, mode, src_vocab, dec_vocab, global_step, our_sam_vocab):
#===============================================================
    # init
    logger.info("***** Running evaluation on {} dataset *****".format(mode))

    eval_loss = 0.0
    eval_steps = 0

    references = []
    candidates = []
    total_correct = 0

    wrong_case = {
        "input_sent": [],
        "pred_sent": [],
        "ans_sent": []
    }

    batch_src_tok_list = []
    pred_tok_list = []
    ans_tok_list = []

    # 우리말샘 기분석 삿전을 통해 바뀐 문장 갯수
    all_our_sam_debug_info: List[OurSamItem] = []
    total_change_cnt = 0

    criterion = nn.NLLLoss()
    eval_sampler = SequentialSampler(eval_datasets)
    eval_dataloader = DataLoader(eval_datasets, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_pbar = tqdm(eval_dataloader)

    cuda_starter, cuda_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_times = []

    model.eval()
    start_time = time.time()
    for batch in eval_pbar:
        torch.cuda.synchronize()
        with torch.no_grad():
            inputs = make_electra_only_dec_inputs(batch)
            inputs["mode"] = "eval"

            cuda_starter.record()
            output = model(**inputs)
            cuda_ender.record()
            torch.cuda.synchronize()
            cuda_times.append(cuda_starter.elapsed_time(cuda_ender) / 1000)

            output = F.log_softmax(output, -1)
            loss = criterion(output.reshape(-1, len(dec_vocab)), batch["tgt_tokens"].view(-1).to(args.device))

            eval_loss += loss.mean().item()

            batch_src_tok_list.append(inputs["src_tokens"].detach().cpu())
            pred_tok_list.append(torch.argmax(output, -1).detach().cpu())
            ans_tok_list.append(batch["tgt_tokens"].detach().cpu())

        eval_steps += 1
        eval_pbar.set_description("Eval Loss - %.04f" % (eval_loss / eval_steps))
    # end loop
    end_time = time.time()

    # Decode
    for src_tok, pred_tok, ans_tok in zip(batch_src_tok_list, pred_tok_list, ans_tok_list):
        for d_idx, (input_i, pred, lab) in enumerate(zip(src_tok, pred_tok, ans_tok)):
            input_sent = "".join([src_vocab[x] for x in input_i.tolist()]).strip()
            pred_sent = "".join([dec_vocab[x] for x in pred.tolist()]).strip()
            ans_sent = "".join([dec_vocab[x] for x in lab.tolist()]).strip()

            input_sent = re.sub(r"\[CLS\]|\[SEP\]|\[PAD\]", "", input_sent)
            pred_sent = re.sub(r"\[CLS\]|\[SEP\]|\[PAD\]", "", pred_sent)
            ans_sent = re.sub(r"\[CLS\]|\[SEP\]|\[PAD\]", "", ans_sent)

            if args.use_our_sam:
                our_sam_res, is_change = apply_our_sam_word_item(our_sam_g2p_dict=our_sam_vocab,
                                                                 mecab=mecab,
                                                                 input_sent=input_sent,
                                                                 pred_sent=pred_sent,
                                                                 ans_sent=ans_sent)
                if is_change:
                    pred_sent = our_sam_res.conv_sent
                    total_change_cnt += 1
                    all_our_sam_debug_info.append(our_sam_res)

            print(f"{d_idx}\n"
                  f"input_sent:\n{input_sent}\n"
                  f"pred_sent:\n{pred_sent}\n"
                  f"ans_snet:\n{ans_sent}\n")

            candidates.append(pred_sent)
            references.append(ans_sent)

            if ans_sent == pred_sent:
                total_correct += 1
            else:
                wrong_case["input_sent"].append(input_sent)
                wrong_case["pred_sent"].append(pred_sent)
                wrong_case["ans_sent"].append(ans_sent)
    # end loop, decode

    wer_score = hug_eval.load("wer").compute(predictions=candidates, references=references)
    per_score = hug_eval.load("cer").compute(predictions=candidates, references=references)
    print(f"[run_ensemble][evaluate] wer_score: {wer_score * 100}, size: {len(candidates)}")
    print(f"[run_ensemble][evaluate] per_score: {per_score * 100}, size: {len(candidates)}")
    print(f"[run_ensemble][evaluate] s_acc: {total_correct / len(eval_datasets) * 100}, size: {total_correct}, "
          f"total.size: {len(eval_datasets)}")
    print(f"[run_ensemble][evaluate] Elapsed time: {end_time - start_time} seconds")
    print(f'[run_ensemble][evaluate] CUDA time: {sum(cuda_times)} seconds')

    logger.info("---Eval End !")
    eval_pbar.close()

    # Save score
    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))

    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))

        f_w.write("  wer = {}\n".format(wer_score))
        f_w.write("  per = {}\n".format(per_score))
        f_w.write("  acc = {}\n".format(total_correct / len(eval_datasets)))
        f_w.write("  Elapsed time: {} seconds\n".format(end_time - start_time))
        f_w.write("  GPU time: {} seconds".format(sum(cuda_times)))

    # wrong case
    wrong_df = pd.DataFrame(wrong_case)
    wrong_df.to_csv(f"./results/electra_nart_dec/{mode}_wrong_case.csv", index=False, header=True)

    ''' 우리말 사전 적용 결과 저장 '''
    if args.use_our_sam and args.our_sam_debug:
        save_our_sam_debug(all_item_save_path='./results/ensemble/our_sam_all.txt',
                           wrong_item_save_path='./results/ensemble/our_sam_wrong.txt',
                           our_sam_debug_list=all_our_sam_debug_info)
        print(f'[run_ensemble][evaluate] OurSamDebug info Save Complete !')

#===============================================================
def train(args, model, train_datasets, dev_datasets, src_vocab, dec_vocab, our_sam_vocab):
#===============================================================
    train_data_size = len(train_datasets)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (train_data_size // args.gradient_accumulation_steps) + 1
    else:
        t_total = (train_data_size // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"[train] t_toal: {t_total}")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = model.parameters()

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

    # Train !
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_datasets))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    criterion = nn.NLLLoss()
    train_sampler = RandomSampler(train_datasets)

    model.zero_grad()
    for epoch in range(args.num_train_epochs):
        model.train()

        train_dataloader = DataLoader(train_datasets, sampler=train_sampler, batch_size=args.train_batch_size)
        pbar = tqdm(train_dataloader)

        for step, batch in enumerate(pbar):
            inputs = make_electra_only_dec_inputs(batch)
            inputs["mode"] = "train"

            output = model(**inputs)
            output = F.log_softmax(output, -1)

            loss = criterion(output.reshape(-1, len(dec_vocab)), batch["tgt_tokens"].view(-1).to(args.device))
            loss.backward()
            optimizer.step()

            model.zero_grad()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    (len(train_datasets) <= args.gradient_accumulation_steps and (step + 1) == len(train_datasets)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                global_step += 1

                pbar.set_description("Train Loss - %.04f" % (tr_loss / global_step))
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save samples checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving samples checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0) and \
                        args.evaluate_test_during_training:
                    evaluate(args, model, dev_datasets, "dev", src_vocab, dec_vocab, global_step, our_sam_vocab)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info("   Epoch Done= %d", epoch + 1)
        pbar.close()

    return global_step, tr_loss / global_step

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

    print_args(args, logger)
    set_seed(args.seed)

    # Load Vocab
    '''
        [PAD] : 0
        [UNK] : 1
        [CLS] : 2
        [SEP] : 3
        [MASK] : 4
    '''
    src_vocab = get_vocab_type_dictionary(tokenizer=tokenizer, is_kochar_electra=True)
    if args.use_custom_vocab:
        dec_vocab = get_vocab_type_dictionary(decode_vocab_path, is_kochar_electra=False)
    else:
        dec_vocab = copy.deepcopy(src_vocab)
    args.src_vocab_size = len(src_vocab)
    args.decoder_vocab_size = len(dec_vocab)
    logger.info(f'src_vocab: {len(src_vocab)}')
    logger.info(f'dec_vocab: {len(dec_vocab)}')

    # Read post_method_dict
    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(jaso_post_path, mode='r', encoding='utf-8') as f:
        post_proc_dict = json.load(f)
    print(f'[run_digits_ensemble][main] post_proc_dict.size: {len(post_proc_dict.keys())}')

    ''' 우리말 샘 문자열-발음열 사전 '''
    our_sam_dict = {}
    with open(our_sam_path, mode='rb') as f:
        our_sam_dict = pickle.load(f)
        our_sam_dict = make_g2p_word_dictionary(our_sam_word_items=our_sam_dict)
    print(f'[run_digits_ensemble][main] our_sam_dict.size: {len(our_sam_dict.keys())}')

    # Load Model
    model = ElectraNartPosDecModel.build_model(args=args, tokenizer=tokenizer,
                                               src_vocab=src_vocab, dec_vocab=dec_vocab,
                                               post_proc_dict=post_proc_dict)
    model.to(args.device)

    # Do Train
    if args.do_train:
        train_datasets = make_digits_ensemble_data(data_path=args.data_pkl, mode='train',
                                                   tokenizer=tokenizer, decode_vocab=dec_vocab)
        dev_datasets = make_digits_ensemble_data(data_path=args.data_pkl, mode='dev',
                                                 tokenizer=tokenizer, decode_vocab=dec_vocab)
        train_datasets = ElectraOnlyDecDataset(item_dict=train_datasets)
        dev_datasets = ElectraOnlyDecDataset(item_dict=dev_datasets)

        global_step, tr_loss = train(args, model,
                                     train_datasets, dev_datasets,
                                     src_vocab, dec_vocab, our_sam_dict)
        logger.info(f'global_step = {global_step}, average loss = {tr_loss}')

    # Do Eval
    if args.do_eval:
        test_datasets = make_digits_ensemble_data(data_path=args.data_pkl, mode='test',
                                                  tokenizer=tokenizer, decode_vocab=dec_vocab)
        test_datasets = ElectraOnlyDecDataset(item_dict=test_datasets)
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(args.output_dir + "/**/" + "model.pt", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logger.info("transformers.configuration_utils")
            logger.info("transformers.modeling_utils")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = ElectraNartPosDecModel.build_model(args=args, tokenizer=tokenizer,
                                                       src_vocab=src_vocab, dec_vocab=dec_vocab,
                                                       post_proc_dict=post_proc_dict)
            model.load_state_dict(torch.load(checkpoint + '/model.pt'))
            model.to(args.device)
            evaluate(args, model, test_datasets, "test",
                     src_vocab, ç, global_step, our_sam_dict)

### MAIN ###
if '__main__' == __name__:
    logger.info(f'[run_digits_ensemble][__main__] START !')

    main(
        config_path='./config/digits_ensemble_config.json',
        decode_vocab_path='./data/vocab/pron_eumjeol_vocab.json',
        jaso_post_path='./data/post_method/jaso_filter.json',
        our_sam_path='./data/dictionary/filtered_dict_word_item.pkl'
    )
