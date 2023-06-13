import copy

from definition.data_def import DictWordItem, OurSamDebug
from typing import List, Dict


#========================================================
def make_g2p_word_dictionary(
    our_sam_word_items: List[DictWordItem]
):
#========================================================
    ret_dict = {}

    print(f'[post_method][make_g2p_word_dictionary] raw_word_items.size: {len(our_sam_word_items)}')

    duplicated_cnt = 0
    for wi_idx, word_item in enumerate(our_sam_word_items):
        if word_item.word in ret_dict.keys():
            duplicated_cnt += 1
        else:
            ret_dict[word_item.word] = word_item.pronun_list

    print(f'[post_method][make_g2p_word_dictionary] ret_dict.keys().size: {len(ret_dict.keys())}, '
          f'duplicated_cnt: {duplicated_cnt}')

    return ret_dict

#========================================================
def apply_our_sam_word_item(
    our_sam_g2p_dict: Dict[str, List[str]], mecab,
    input_sent: str, pred_sent: str, ans_sent: str
):
#========================================================
    debug_info = OurSamDebug(
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

    split_pred_sent = pred_sent.split(' ')
    split_ans_sent = ans_sent.split(' ')
    for inp_idx, inp_item in enumerate(input_sent.split(' ')):
        include_flag = True
        for tag in pos_list[inp_idx]:
            if tag not in ['NNG', 'NNP']: # 1: NNP, 2: NNG
                include_flag = False
                break

        if not include_flag:
            continue

        if (inp_item in our_sam_g2p_dict.keys()) and (inp_item not in our_sam_g2p_dict[inp_item]):
            '''
                복수 표준 발음 처리 (임시)
                    - key-value 에서 value는 발음열들의 목록
                    - 이 안에 없다면 현재 예측된 발음열을 교체
            '''
            split_pred_sent[inp_idx] = our_sam_g2p_dict[inp_item]
            is_change = True

            # For Debug
            debug_info.input_word.append(inp_item)
            debug_info.pred_word.append(split_pred_sent[inp_idx])
            debug_info.our_sam_word.append(our_sam_g2p_dict[inp_item])
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

### MAIN ###
if '__main__' == __name__:
    print(f'[post_method][__main__] MAIN !')
