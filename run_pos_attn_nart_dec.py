import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

import json
import copy
import evaluate as hug_eval
import pandas as pd
import pickle
import os
import re
import glob
from attrdict import AttrDict
from typing import List, Dict

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from model.electra_nart_pos_dec_model import ElectraNartPosDecModel

from utils.kocharelectra_tokenization import KoCharElectraTokenizer
from run_utils import (
    init_logger, set_seed, print_args
)
from utils.electra_only_dec_utils import (
    get_vocab_type_dictionary, load_electra_transformer_decoder_npy,
    ElectraOnlyDecDataset, make_electra_only_dec_inputs
)

import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux

import time

### GLOBAL
logger = init_logger()
tokenizer = KoCharElectraTokenizer.from_pretrained('monologg/kocharelectra-base-discriminator')


#==================================================================
def train(args, model, train_datasets, dev_datasets, src_vocab, dec_vocab, our_sam_vocab):
#==================================================================
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


#==================================================================
def evaluate(args, model, eval_datasets, mode, src_vocab, dec_vocab, global_step, our_sam_vocab):
#==================================================================
    logger.info("***** Running evaluation on {} dataset *****".format(mode))

    # init
    mecab = Mecab()

    eval_loss = 0.0
    eval_steps = 0

    references = []
    candidates = []
    total_correct = 0

    change_cnt = 0

    wrong_case = {
        "input_sent": [],
        "pred_sent": [],
        "ans_sent": []
    }

    batch_src_tok_list = []
    pred_tok_list = []
    ans_tok_list = []

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
                mecab_res = mecab.pos(input_sent)
                nn_pos_words = [x[0] for x in mecab_res if "NNG" == x[1] or "NNP" == x[1]]

                # Change candidate by vocab
                input_split = input_sent.split(" ")
                pred_split = pred_sent.split(" ")
                ans_split = ans_sent.split(" ")

                for i_idx, input_word in enumerate(input_split):
                    if len(input_split) != len(pred_split):
                        break
                    if input_word in our_sam_vocab.keys() and pred_split[i_idx] not in our_sam_vocab[
                        input_word] and input_word in nn_pos_words:
                        origin_pred_word = pred_split[i_idx]
                        conv_word = our_sam_vocab[input_word][0]
                        for proun in our_sam_vocab[input_word]:
                            if proun == input_split[i_idx]:
                                conv_word = proun
                        pred_split[i_idx] = conv_word

                        # change vocab_pronun to ref_pronun
                        if ans_split[i_idx] != conv_word:
                            print(ans_split[i_idx], conv_word)
                            ans_split[i_idx] = conv_word
                            print(" ".join(input_split))
                            print(" ".join(pred_split))
                            print(" ".join(ans_split))

                        change_cnt += 1

                input_sent = " ".join(input_split)
                pred_sent = " ".join(pred_split)
                ans_sent = " ".join(ans_split)

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
    print(f"[run_nart_pos_dec][evaluate] wer_score: {wer_score * 100}, size: {len(candidates)}")
    print(f"[run_nart_pos_dec][evaluate] per_score: {per_score * 100}, size: {len(candidates)}")
    print(f"[run_nart_pos_dec][evaluate] s_acc: {total_correct / len(eval_datasets) * 100}, size: {total_correct}, "
          f"total.size: {len(eval_datasets)}")
    print(f"[run_nart_pos_dec][evaluate] Elapsed time: {end_time - start_time} seconds")
    print(f'[run_nart_pos_dec][evaluate] CUDA time: {sum(cuda_times)} seconds')

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

#==================================================================
def main(
        config_path: str,
        custom_vocab_path: str,
        our_sam_path: str,
        jaso_post_proc_path: str
):
#==================================================================
    logger.info(f'config_path: {config_path}')
    logger.info(f'custom_vocab_path: {custom_vocab_path}')
    logger.info(f'our_sam_path: {our_sam_path}')
    logger.info(f'jaso_post_proc_path: {jaso_post_proc_path}')

    if not os.path.exists(config_path):
        raise Exception(f'ERR - config_path')
    if not os.path.exists(custom_vocab_path):
        raise Exception(f'ERR - custom_vocab_path')
    if not os.path.exists(our_sam_path):
        raise Exception(f'ERR - our_sam_path')
    if not os.path.exists(jaso_post_proc_path):
        raise Exception(f'ERR - jaso_post_proc_path')

    # Read Config
    config = None
    with open(config_path) as c_f:
        config = AttrDict(json.load(c_f))
    if "cuda" != config.device and "cpu" != config.device:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print_args(config, logger)
    set_seed(config.seed)
    config.output_dir = os.path.join(config.ckpt_dir, config.output_dir)

    ''' 초/중/종성 마다 올 수 있는 발음 자소를 가지고 있는 사전 '''
    post_proc_dict: Dict[str, Dict[str, List[str]]] = {}
    with open(jaso_post_proc_path, mode="r", encoding="utf-8") as f:
        post_proc_dict = json.load(f)

    # Load Vocab
    '''
        [PAD] : 0
        [UNK] : 1
        [CLS] : 2
        [SEP] : 3
        [MASK] : 4
    '''
    src_vocab = get_vocab_type_dictionary(tokenizer=tokenizer, is_kochar_electra=True)
    if config.use_custom_vocab:
        dec_vocab = get_vocab_type_dictionary(custom_vocab_path, is_kochar_electra=False)
    else:
        dec_vocab = copy.deepcopy(src_vocab)
    config.src_vocab_size = len(src_vocab)
    config.decoder_vocab_size = len(dec_vocab)
    logger.info(f'src_vocab: {len(src_vocab)}')
    logger.info(f'dec_vocab: {len(dec_vocab)}')

    # Load Our Sam Vocab
    our_sam_vocab = None
    with open(our_sam_path, mode='rb') as o_f:
        our_sam_vocab = pickle.load(o_f)
    logger.info(f'[__main__] our_sam_vocab.size: {len(our_sam_vocab)}')
    logger.info(list(our_sam_vocab.items())[:10])

    # Build Model
    model = ElectraNartPosDecModel.build_model(args=config, tokenizer=tokenizer,
                                               src_vocab=src_vocab, dec_vocab=dec_vocab,
                                               post_proc_dict=post_proc_dict)
    model.to(config.device)

    # Do Train
    if config.do_train:
        train_npy_dict = load_electra_transformer_decoder_npy(config.train_npy, mode='train')
        dev_npy_dict = load_electra_transformer_decoder_npy(config.dev_npy, mode='dev')

        train_datasets = ElectraOnlyDecDataset(item_dict=train_npy_dict)
        dev_datasets = ElectraOnlyDecDataset(item_dict=dev_npy_dict)

        global_step, tr_loss = train(config, model,
                                     train_datasets, dev_datasets,
                                     src_vocab, dec_vocab, our_sam_vocab)
        logger.info(f'global_step: {global_step}, average_loss: {tr_loss}')

    # Do Eval
    if config.do_eval:
        test_npy_dict = load_electra_transformer_decoder_npy(config.test_npy, mode='test')
        test_datasets = ElectraOnlyDecDataset(item_dict=test_npy_dict)
        checkpoints = list(os.path.dirname(c) for c in
                           sorted(glob.glob(config.output_dir + "/**/" + "model.pt", recursive=True),
                                  key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))

        if not config.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logger.info("transformers.configuration_utils")
            logger.info("transformers.modeling_utils")
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = ElectraNartPosDecModel.build_model(args=config, tokenizer=tokenizer,
                                                       src_vocab=src_vocab, dec_vocab=dec_vocab,
                                                       post_proc_dict=post_proc_dict)
            model.load_state_dict(torch.load(checkpoint + '/model.pt'))
            model.to(config.device)

            evaluate(config, model, test_datasets, "test", src_vocab, src_vocab, global_step, our_sam_vocab)

### MAIN ###
if '__main__' == __name__:
    logger.info(f'[run_nart_pos_dec][__main__] START !')

    config_path = './config/nart_pos_dec_config.json'
    custom_vocab_path = './data/vocab/pron_eumjeol_vocab.json'
    our_sam_path = './data/dictionary/our_sam_std_dict.pkl'
    jaso_post_proc_path = './data/post_method/jaso_filter.json'

    main(config_path=config_path,
         custom_vocab_path=custom_vocab_path,
         our_sam_path=our_sam_path,
         jaso_post_proc_path=jaso_post_proc_path
    )