import os
import re
import sys
import copy
import pickle

from definition.data_def import KT_TTS
from definition import special_rules
from typing import List


#==========================================
class KT_TTS_Maker():
#==========================================
    def __init__(self):
        print(f'[KT_TTS_Maker][__init__] INIT !')

    def get_kt_tts_items(
            self,
            raw_src_path: str, raw_tgt_path: str,
            save_path: str
    ) -> (List[KT_TTS], List[KT_TTS]):
        '''
            Text 파일을 읽어서 KT_TTS로 만든다.
            Save path가 있다면 파일까지 *.pkl로 저장
        '''
        ''' 함수 테스트 필요! '''
        print(f'[KT_TTS_Maker][get_kt_tts_items] src_path: {raw_src_path}\ntgt_path: {raw_tgt_path}')

        if not os.path.exists(raw_src_path):
            raise Exception('[KT_TTS_Maker][get_kt_tts_items] Plz Check - raw_src_path')
        if not os.path.exists(raw_tgt_path):
            raise Exception('[KT_TTS_Maker][get_kt_tts_items] Plz Check - raw_tgt_path')

        ret_src_items: List[KT_TTS] = []
        ret_tgt_items: List[KT_TTS] = []

        # Read
        with open(raw_src_path, mode='r', encoding='utf-8') as s_f:
            src_lines = [x.replace('\n', '').split('\t') for x in s_f.readlines()]
        with open(raw_tgt_path, mode='r', encoding='utf-8') as t_f:
            tgt_lines = [x.replace('\n', '').split('\t') for x in t_f.readlines()]
        print(f'[KT_TTS_Maker][get_kt_tts_itmes] src_lines.size: {len(src_lines)}, tgt_lines.size: {len(tgt_lines)}')
        assert len(src_lines) == len(tgt_lines), 'ERR - src_line.size != tgt_lines.size'

        # insert items to list
        err_cnt = 0
        for r_idx, (s_line, t_line) in enumerate(zip(src_lines, tgt_lines)):
            s_id = s_line[0]
            s_sent = s_line[1]

            t_id = t_line[0]
            t_sent = t_line[1]

            if s_id != t_id:
                err_cnt += 1
                print(f'[KT_TTS_Maker][get_kt_tts_items] ERR ! - s_id: {s_id}, t_id: {t_id}')
                continue

            if 0 == (r_idx % 5000):
                print(f'[KT_TTS_Maker][get_kt_tts_items] {r_idx} is processing... {s_sent}')

            ret_src_items.append(copy.deepcopy(KT_TTS(id=s_id, sent=s_sent)))
            ret_tgt_items.append(copy.deepcopy(KT_TTS(id=t_id, sent=t_sent)))
        # end loop

        print(f'[KT_TTS_Maker][get_kt_tts_items] ret_src_items.size: {len(ret_src_items)}')
        print(f'[KT_TTS_Maker][get_kt_tts_items] ret_tgt_items.size: {len(ret_tgt_items)}')
        assert len(ret_src_items) == len(ret_tgt_items), 'ERR - ret_items.size is diff !'

        if 0 < len(save_path) and os.path.exists(save_path):
            ''' 저장 '''
            with open(save_path+'/kt_tts_src_items.pkl', mode='wb') as s_pf:
                pickle.dump(ret_src_items, s_pf)
            with open(save_path+'/kt_tts_tgt_items.pkl', mode='wb') as t_pf:
                pickle.dump(ret_tgt_items, t_pf)
            print(f'[KT_TTS_Maker][get_kt_tts_items] SAVE ! - src/tgt items, path: {save_path}')

        return ret_src_items, ret_tgt_items

    def get_converted_symbol_items(
            self,
            src_tts_items: [KT_TTS]
    ) -> [KT_TTS]:
        ''' 특수 기호를 발음열 변환된 문장으로 변환 '''

        id, src = src_tts_items.id, src_tts_items.sent
        sp_char = special_rules.SYMBOL_RULES
        # 플러스
        if re.search(sp_char['r_plus'], src):
            special_char = re.findall(sp_char['r_plus'], src)
            src = self._replace_word(src=src, special=special_char, symbol="+", replace_word=" 플러스 ")
        # 에
        if re.search(sp_char['r_eh'], src):
            special_char = re.findall(sp_char['r_eh'], src)
            src = self._replace_word(src=src, special=special_char, symbol="-", replace_word="에 ")
        # 묵음
        if re.search(sp_char['r_blank'], src):
            special_char = re.findall(sp_char['r_blank'], src)
            src = self._replace_word(src=src, special=special_char, symbol="-", replace_word=" ")
            src = self._replace_word(src=src, special=special_char, symbol="/", replace_word=" ")
        # 마이너스
        if re.search(sp_char['r_minus'], src):
            special_char = re.findall(sp_char['r_minus'], src)
            src = self._replace_word(src=src, special=special_char, symbol="-", replace_word=" 마이너스 ")
        # 곱하기
        if re.search(sp_char['r_multi'], src):
            special_char = re.findall(sp_char['r_multi'], src)
            src = self._replace_word(src=src, special=special_char, symbol="*", replace_word=" 곱하기 ")
        # 별
        if re.search(sp_char['r_star'], src):
            special_char = re.findall(sp_char['r_star'], src)
            src = self._replace_word(src=src, special=special_char, symbol="*", replace_word="별")
        # 나누기
        if re.search(sp_char['r_division'], src):
            special_char = re.findall(sp_char['r_division'], src)
            src = self._replace_word(src=src, special=special_char, symbol="/", replace_word=" 나누기 ")
        # 년 월 일
        if re.search(sp_char['r_date'], src):
            special_char = re.findall(sp_char['r_date'], src)
            src = self._replace_date(src=src, special=special_char)
        # 분에
        if re.search(sp_char['r_fraction'], src):
            special_char = re.findall(sp_char['r_fraction'], src)
            src = self._replace_fraction(src=src, special=special_char, replace_word=" 분에 ")
        # 은/는
        if re.search(sp_char['r_equal'], src):
            special_char = re.findall(sp_char['r_equal'], src)
            src = self._replace_equal(src=src, special=special_char)
        # 시간
        if re.search(sp_char['r_time'], src):
            special_char = re.findall(sp_char['r_time'], src)
            src = self._replace_time(src=src, special=special_char)
        # 샵
        if re.search(sp_char['r_hash'], src):
            special_char = re.findall(sp_char['r_hash'], src)
            src = self._replace_word(src=src, special=special_char, symbol="#", replace_word="샵 ")
        # 엔
        if re.search(sp_char['r_ampersand'], src):
            special_char = re.findall(sp_char['r_ampersand'], src)
            src = self._replace_word(src=src, special=special_char, symbol="&", replace_word=" 엔 ")
        # 이메일: 쩜, 골뱅이
        if re.search(sp_char['r_email'], src):
            special_char = re.findall(sp_char['r_email'], src)
            src = self._replace_word(src=src, special=special_char, symbol="@", replace_word=" 골뱅이 ")
            special_char = [re.findall("\.", s) for s in special_char][0]
            src = self._replace_word(src=src, special=special_char, symbol=".", replace_word=" 쩜 ")
        # 밑줄표
        if re.search(sp_char['r_underline'], src):
            special_char = re.findall(sp_char['r_underline'], src)
            src = self._replace_word(src=src, special=special_char, symbol="_", replace_word=" 밑줄표 ")

        # 띄어쓰기 하나로
        src = re.sub(r'\s{2,}', " ", src)
        src = src.strip()
        source = KT_TTS(id=id, sent=src)
        # print(ret_src_items)
        return source

    def _replace_word(self, src:str, special:List, symbol:str, replace_word:str) -> str:
        ''' 기본 단어 바꾸기 '''
        for s in special:
            sent = s.replace(symbol, replace_word)
            src = src.replace(s, sent)

        return src

    def _replace_date(self, src:str, special:List) -> str:
        ''' 날짜 형식으로 바꾸기 '''
        for s in special:
            if '.' in s and len(s.split(".")) == 3:
                sent = s.replace(".", "년 ", 1)
                sent = sent.replace(".", "월 ", 1)
                sent += "일"
            elif '/' in s and len(s.split("/")) == 3:
                sent = s.replace("/", "년 ", 1)
                sent = sent.replace("/", "월 ", 1)
                sent += "일"
            else:
                print("[date][error]", src, special)
                sys.exit()
            src = src.replace(s, sent)
        return src

    def _replace_fraction(self, src:str, special:List, replace_word:str) -> str:
        ''' 몇 분의 몇 '''
        for s in special:
            if len(s.split('/')) == 2:
                num1, num2 = s.split('/')
                sent = s.replace(s, num2 + replace_word + num1)
                src = src.replace(s, sent)
            else:
                print('[fraction][error]', src, special)
                sys.exit()
        return src

    def _replace_equal(self, src: str, special: List) -> str:
        ''' = -> 은/는 '''
        # 은 발음
        uen = ['0', '1', '3', '6', '7', '8', '9']

        for s in special:
            # 기호 = 앞의 숫자 '은' 발음
            if src.split("=")[0].strip()[-1] in uen:
                sent = s.replace(s, '은')
            # 그 외는 '는' 발음
            else:
                sent = s.replace(s, '는')
            src = src.replace(s, sent)
        return src

    def _replace_time(self, src: str, special: List) -> str:
        ''' 시간 형태로 바꾸기 '''
        for s in special:
            if ':' in s and len(s.split(":")) == 3:
                sent = s.replace(":", "시 ", 1)
                sent = sent.replace(":", "분 ", 1)
                sent += "초"
            else:
                print('[time][error]', src, special)
                sys.exit()
            src = src.replace(s, sent)
        return src

### MAIN ###
if '__main__' == __name__:
    print(f'[kt_tts_pkl_maker][__main__] MAIN !')

    tts_maker = KT_TTS_Maker()
    b_txt2pkl = False
    b_symbol_rules = False
    b_symbol_rules_per_sent = True

    '''
        *.txt -> *.pkl 변환 
    '''
    if b_txt2pkl:
        src_path = '../data/kt_tts/tts_script_85ks_ms949_200407.txt'
        tgt_path = '../data/kt_tts/tts_script_85ks_ms949_200506_g2p.txt'
        pkl_path = '../data/kt_tts/pkl/'
        tts_src_items, tts_tgt_items = tts_maker.get_kt_tts_items(
            raw_src_path=src_path, raw_tgt_path=tgt_path, save_path=pkl_path
        )

    '''
        특수문자 변환
        단위: List[KT_TTS]
        기준: KT 특수문자 처리.xlsx
    '''
    if b_symbol_rules:
        src = KT_TTS(id='0', sent="abc_def")
        tgt = KT_TTS(id='0', sent="십이 더하기 칠 은 십구")
        new_sources = tts_maker.get_converted_symbol_items(src)
        print(new_sources)

    '''
        특수문자 변환
        단위: str (1 문장씩)
        기준: KT 특수문자 처리.xlsx
    '''
    if b_symbol_rules_per_sent:
        ''' 여기서 테스트 :D '''
        src = KT_TTS(id='0', sent="abc_def")
        tgt = KT_TTS(id='0', sent="십이 더하기 칠 은 십구")
        new_sources = tts_maker.get_converted_symbol_items(src)
        print(new_sources)