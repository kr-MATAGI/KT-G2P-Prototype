import re
import copy

from definition.data_def import DictWordItem, OurSamItem
from dataclasses import dataclass, field
from typing import List, Dict

import evaluate as hug_eval

@dataclass
class PreDictItem:
    pronun_list: List[str] = field(default_factory=list)
    pos: str = ''

#========================================================
def make_g2p_word_dictionary(
    our_sam_word_items: List[DictWordItem]
):
#========================================================
    ret_dict = {} # { 'word': PreDictItem(), ... }

    print(f'[post_method][make_g2p_word_dictionary] raw_word_items.size: {len(our_sam_word_items)}')

    duplicated_cnt = 0
    conju_duplicated_cnt = 0
    for wi_idx, word_item in enumerate(our_sam_word_items):
        if word_item.word in ret_dict.keys():
            duplicated_cnt += 1
        else:
            ret_dict[word_item.word] = copy.deepcopy(PreDictItem(pronun_list=word_item.pronun_list, pos=word_item.pos))

        ''' 활용형에 대한 처리 '''
        for cj_idx, conju_item in enumerate(word_item.conju_list):
            # conju_item[0]: word, conju_item[1]: prounciation
            conju_item = list(conju_item)
            conju_item[1] = conju_item[1].replace('^', ' ').replace('-', '')
            conju_item[1] = re.sub(r'[^가-힣]+', '', conju_item[1])
            if conju_item[0] in ret_dict.keys():
                conju_duplicated_cnt += 1
            else:
                ret_dict[conju_item[0]] = copy.deepcopy(PreDictItem(pronun_list=[conju_item[1]], pos=word_item.pos))

    print(f'[post_method][make_g2p_word_dictionary] ret_dict.keys().size: {len(ret_dict.keys())}, '
          f'duplicated_cnt: {duplicated_cnt}')

    return ret_dict

#========================================================
def apply_our_sam_word_item(
    our_sam_g2p_dict: Dict[str, List[str]], mecab,
    input_sent: str, pred_sent: str, ans_sent: str
):
#========================================================
    debug_info = OurSamItem(
        input_sent=input_sent, pred_sent=pred_sent, ans_sent=ans_sent
    )
    is_change = False
    mecab_res = mecab.pos(input_sent)
    mecab_res = make_eojeol_mecab_res(input_sent, mecab_res)
    pos_list = []
    for res_item in mecab_res:
        eojeol_pos = []
        for morp_item in res_item:
            eojeol_pos.extend(morp_item[-1])
        pos_list.append(eojeol_pos)
    # [[('저', ['NP']), ('를', ['JKO'])], [('부르', ['VV']), ('셨', ['EP', 'EP']), ('나요', ['EC'])]]
    # [['NP', 'JKO'], ['VV', 'EP', 'EP', 'EC']]

    split_input_sent = input_sent.split(' ')
    split_pred_sent = pred_sent.split(' ')
    split_ans_sent = ans_sent.split(' ')

    if len(split_pred_sent) != len(split_ans_sent):
        is_change = False
        return None, is_change
    else:
        for inp_idx, inp_item in enumerate(split_input_sent):
            include_flag = True
            '''
                NNG: 일반 명사
                NNP: 고유 명사
                VV : 동사
                VA : 형용사
            '''
            b_include_nn = False
            b_include_vv_va = False
            for t_idx, tag in enumerate(pos_list[inp_idx]):
                if tag in ['NNG', 'NNP']: # 어절 안에 명사 종류가 포함되어 있는가
                    ''' 명사가 포함된경우 명사만 포함되어있는가 '''
                    b_include_nn = True
                else:
                    b_include_nn = False
                if tag in ['VV', 'VA'] and 0 == t_idx: # 어절 안에 동사, 형용사가 포함되어 있는가?
                    ''' 동사, 형용사가 포함된 경우 첫 품사가 동사 or 형용사인가 '''
                    b_include_vv_va = True
                    break

            if not b_include_nn and not b_include_vv_va:
                continue

            if (inp_item in our_sam_g2p_dict.keys()) and len(split_input_sent) == len(split_pred_sent) and \
                    (split_pred_sent[inp_idx] not in our_sam_g2p_dict[inp_item].pronun_list):
                '''
                    복수 표준 발음 처리 (임시)
                        - key-value 에서 value는 발음열들의 목록
                        - 이 안에 없다면 현재 예측된 발음열을 교체
                '''
                split_pred_sent[inp_idx] = our_sam_g2p_dict[inp_item].pronun_list[0]
                is_change = True

                # For Debug
                debug_info.input_word.append(inp_item)
                debug_info.pred_word.append(split_pred_sent[inp_idx])
                debug_info.our_sam_word.append(our_sam_g2p_dict[inp_item].pronun_list)
                debug_info.pos = our_sam_g2p_dict[inp_item].pos
                debug_info.ans_word.append(split_ans_sent[inp_idx])
        # end loop, inp_item

        debug_info.conv_sent = ' '.join(split_pred_sent).strip()

        return debug_info, is_change

#========================================================
def make_eojeol_mecab_res(input_sent: str, mecab_res: List):
#========================================================
    # 어절별 글자 수 체크해서 띄워쓰기 적기
    split_text = input_sent.split(" ")
    char_cnt_list = [len(st) for st in split_text]

    total_eojeol_morp = []
    eojeol_set = []
    use_check = [False for _ in range(len(mecab_res))]
    len_idx = 0
    curr_char_cnt = 0
    for ej_idx, eojeol in enumerate(mecab_res):
        if char_cnt_list[len_idx] == curr_char_cnt:
            total_eojeol_morp.append(copy.deepcopy(eojeol_set))
            eojeol_set = []
            len_idx += 1
            curr_char_cnt = 0
        if use_check[ej_idx]:
            continue

        eojeol_set.append((eojeol[0], eojeol[1].split("+")))
        curr_char_cnt += len(eojeol[0].strip())  # 에듀' '  <- Mecab 결과에 공백이 따라오는 경우 있음
        use_check[ej_idx] = True
    if 0 < len(eojeol_set):
        total_eojeol_morp.append(copy.deepcopy(eojeol_set))

    return total_eojeol_morp

#========================================================
def save_our_sam_debug(
        all_item_save_path: str, wrong_item_save_path: str,
        our_sam_debug_list: List[OurSamItem]
):
#========================================================
    print(f"[post_method][save_debug_txt] all_item_save_path: {all_item_save_path}\n"
          f"wrong_item_save_path: {wrong_item_save_path}")

    wrong_case_cnt = 0
    with open(all_item_save_path, mode="w", encoding="utf-8") as all_f, \
            open(wrong_item_save_path, mode='w', encoding='utf-8') as wrong_f:
        for d_idx, debug_item in enumerate(our_sam_debug_list):
            all_f.write(f"{str(d_idx)}\n\n")
            all_f.write(f"입력 문장:\n{debug_item.input_sent}\n\n")
            all_f.write(f"예측 문장:\n{debug_item.pred_sent}\n\n")
            all_f.write(f"정답 문장:\n{debug_item.ans_sent}\n\n")
            all_f.write(f"변경된 문장:\n{debug_item.conv_sent}\n\n")

            all_f.write("=========================\n")
            all_f.write(f"입력  예측  변경  품사  정답\n")
            for inp, pred, conv, ans in zip(debug_item.input_word, debug_item.pred_word,
                                            debug_item.our_sam_word, debug_item.ans_word):
                all_f.write(f"{inp}\t{pred}\t{conv}\t{ans}\n")
            all_f.write("=========================\n\n")

            # Wrong Case
            if (debug_item.pred_sent != debug_item.conv_sent) and (debug_item.pred_sent != debug_item.ans_sent):
                wrong_case_cnt += 1
                wrong_f.write(f"{str(d_idx)}\n\n")
                wrong_f.write(f"입력 문장:\n{debug_item.input_sent}\n\n")
                wrong_f.write(f"예측 문장:\n{debug_item.pred_sent}\n\n")
                wrong_f.write(f"정답 문장:\n{debug_item.ans_sent}\n\n")
                wrong_f.write(f"변경된 문장:\n{debug_item.conv_sent}\n\n")

                wrong_f.write("=========================\n")
                wrong_f.write(f"입력  예측  변경  품사  정답\n")
                for inp, pred, conv, ans in zip(debug_item.input_word, debug_item.pred_word,
                                                debug_item.our_sam_word, debug_item.ans_word):
                    wrong_f.write(f"{inp}\t{pred}\t{conv}\t{debug_item.pos}\t{ans}\n")
                wrong_f.write("=========================\n\n")

    print(f"[post_method][save_debug_txt] wrong_case_cnt: {wrong_case_cnt}")

#========================================================
def get_dict_items_info(target_dict: Dict[str, PreDictItem]):
#========================================================
    nn_cnt = 0
    vv_cnt = 0
    va_cnt = 0
    total_cnt = len(target_dict)
    for idx, (key, dict_item) in enumerate(target_dict.items()):
        if '명사' == dict_item.pos:
            nn_cnt += 1
        elif '동사' == dict_item.pos:
            vv_cnt += 1
        elif '형용사' == dict_item.pos:
            va_cnt += 1
    print(f'[post_method][get_dict_item_info] 갯수 - 총합: {total_cnt} 명사: {nn_cnt}, 동사: {vv_cnt}, 형용사: {va_cnt}')

#========================================================
def re_evaluate_apply_dict(
        target_items: List[OurSamItem],
        input_sent_list: List[str], pred_sent_list: List[str], ans_sent_list: List[str]):
#========================================================
    print(f'[post_method][re_evaluate_apply_dict] target_items.size: {len(target_items)}')
    print(f'[post_method][re_evaluate_apply_dict] list.size - input: {len(input_sent_list)}, '
          f'pred: {len(pred_sent_list)}, ans: {len(ans_sent_list)}')

    our_sam_correct_cnt = 0
    before_correct_cnt = 0
    pred_sent_change_cnt = 0
    for tgt_idx, target_items in enumerate(target_items):
        is_change = False
        for inp, pred, conv, ans in zip(target_items.input_word, target_items.pred_word,
                                         target_items.our_sam_word, target_items.ans_word):

            if pred == ans and conv[0] != ans:
                is_change = True
                before_correct_cnt += 1
                target_items.pred_sent = target_items.pred_sent.replace(conv[0], pred)
            elif pred != ans and ans not in conv:
                is_change = True
                our_sam_correct_cnt += 1
                target_items.pred_sent = target_items.pred_sent.replace(pred, conv[0])
                target_items.ans_sent = target_items.ans_sent.replace(ans, conv[0])

        if is_change:
            for p_idx, pred_sent in enumerate(pred_sent_list):
                if target_items.conv_sent == pred_sent:
                    pred_sent_list[p_idx] = target_items.pred_sent
                    pred_sent_change_cnt += 1
                    break

    re_correct_cnt = 0
    for pred, ans in zip(pred_sent_list, ans_sent_list):
        if pred == ans:
            re_correct_cnt += 1

    wer_score = hug_eval.load("wer").compute(predictions=pred_sent_list, references=ans_sent_list)
    per_score = hug_eval.load("cer").compute(predictions=pred_sent_list, references=ans_sent_list)
    sent_acc = re_correct_cnt / len(input_sent_list)
    print(f'[post_method][re_evaluate_apply_dict] our_sam_correct_cnt: {our_sam_correct_cnt}, '
          f'before_correct_cnt: {before_correct_cnt}, pred_sent_change_cnt: {pred_sent_change_cnt}, '
          f're_correct_cnt: {re_correct_cnt}')
    print(f'[post_method][re_evaluate_apply_dict] re_wer = {wer_score * 100}')
    print(f'[post_method][re_evaluate_apply_dict] re_per = {per_score * 100}')
    print(f'[post_method][re_evaluate_apply_dict] re_sent.acc = {sent_acc * 100}')


### MAIN ###
if '__main__' == __name__:
    print(f'[post_method][__main__] MAIN !')
