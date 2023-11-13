import pickle
import evaluate as hug_eval
import time

from g2pk.g2pk import G2p
from typing import List

### Metrics
wer_metric = hug_eval.load('wer')
per_metric = hug_eval.load('cer')


#=====================================================================
def run_g2pk_test(src_data_list: List[str], tgt_data_list: List[str]):
#=====================================================================
    pred_data_list = []

    total_cnt = 0
    correct_cnt = 0
    except_cnt = 0

    g2pk = G2p()

    start_time = time.time()
    for idx, (src_item, tgt_item) in enumerate(zip(src_data_list, tgt_data_list)):
        if src_item.id != tgt_item.id:
            except_cnt += 1
        if len(src_item.sent) != len(tgt_item.sent):
            except_cnt += 1

        if 0 == (idx % 500):
            print(f'[run_g2pk_test] {idx} is processing...')

        total_cnt += 1
        res_g2pk = g2pk(src_item.sent)

        if res_g2pk == tgt_item:
            correct_cnt += 1

        pred_data_list.append(res_g2pk)
    end_time = time.time()

    ## Print Results
    wer_score = wer_metric.compute(predictions=pred_data_list, references=tgt_data_list)
    per_score = per_metric.compute(predictions=pred_data_list, references=tgt_data_list)

    print(f'[run_g2pk_test] WER: {wer_score * 100}')
    print(f'[run_g2pk_test] PER: {per_score * 100}')
    print(f'[run_g2pk_test] Sent Acc: {correct_cnt / total_cnt}')
    print(f'[run_g2pk_test] Elapsed time: {end_time - start_time}')

    print(f'[run_g2pk_test] -------END g2pk test !')

### MAIN ###
if "__main__" == __name__:
    src_data_path = "./data/kor/pkl/kor_source_filter.pkl"
    tgt_data_path = "./data/kor/pkl/kor_target.pkl"

    # Load test dataset
    src_data = []
    with open(src_data_path, mode='rb') as s_f:
        src_data = pickle.load(s_f)
    print(f'[run_g2pk][__main__] {len(src_data)}')
    print(f'{src_data[:10]}')

    tgt_data = []
    with open(tgt_data_path, mode='rb') as t_f:
        tgt_data = pickle.load(t_f)
    print(f'[run_g2pk][__main__] {len(tgt_data)}')
    print(f'{tgt_data[:10]}')

    run_g2pk_test(src_data, tgt_data)