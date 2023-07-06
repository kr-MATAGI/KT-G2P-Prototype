import os
import pickle
import re
import sys

from definition.data_def import KT_TTS
from typing import List

#==========================================
class KT_TTS_Maker():
#==========================================
    def __init__(self):
        print(f'[KT_TTS_Maker][__init__] INIT !')

    def get_kt_tts_items(
            self,
            raw_src_path: str, raw_tgt_path: str
    ):
        print(f'[KT_TTS_Maker][get_kt_tts_items]')

    def get_kt_tts_s2p(
            self,
            src_tts_items: List[KT_TTS], tgt_tts_items: List[KT_TTS]
   ) -> (List[KT_TTS], List[KT_TTS]):
        '''
        특수 기호를 발음열 변환된 문장으로 변환

        :return: List[KT_TTS]
        '''

        # 플러스
        plus = r"\++\s*\d+\+*"
        # 에
        eh = r"(\d+\-\d+\-\d+|\d+\-\d+)"
        # 마이너스
        minus = r"(\-+\s*)+\d+"
        # 묵음
        blank = r"[가-힣]+(?:\-[가-힣]*)+|[가-힣]+(?:\/[가-힣]*)+"
        # 곱하기
        multi = r"\d+\s?\*\s?\d+"
        # 별
        star = r"\*\d+"
        # 나누기
        division = r"\d+\s?/\s?\d+\s?="
        # 년 월 일
        date = r"\d{2,4}/\d{2}/\d{2}|\d{2,4}\.\d{2}.\d{2}"
        # 분에
        fraction = r"\d+\s?/\s?\d+\s?"
        # 은/는
        equal = r"="
        # 시간
        time = r"2[0-3]:[0-5][0-9]:[0-5][0-9]|[01][0-9]:[0-5][0-9]:[0-5][0-9]"
        # 샵
        hash = r"#"
        # 엔
        ampersand = r"[A-Za-z]&[A-Za-z]"
        # 이메일: 쩜, 골뱅이
        email = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        # 밑줄표
        underline = r"[A-Za-z]+(?:_[A-Za-z]*)+"

        ret_src_items: List[KT_TTS] = []
        for source in src_tts_items:
            id, src = source.id, source.sent
            # 플러스
            if re.search(plus, src):
                special_char = re.findall(plus, src)
                src = self._replace_word(src=src, special=special_char, symbol="+", replace_word=" 플러스 ")
            # 에
            if re.search(eh, src):
                special_char = re.findall(eh, src)
                src = self._replace_word(src=src, special=special_char, symbol="-", replace_word="에 ")
            # 묵음
            if re.search(blank, src):
                special_char = re.findall(blank, src)
                src = self._replace_word(src=src, special=special_char, symbol="-", replace_word=" ")
                src = self._replace_word(src=src, special=special_char, symbol="/", replace_word=" ")
            # 마이너스
            if re.search(minus, src):
                special_char = re.findall(minus, src)
                src = self._replace_word(src=src, special=special_char, symbol="-", replace_word=" 마이너스 ")
            # 곱하기
            if re.search(multi, src):
                special_char = re.findall(multi, src)
                src = self._replace_word(src=src, special=special_char, symbol="*", replace_word=" 곱하기 ")
            # 별
            if re.search(star, src):
                special_char = re.findall(star, src)
                src = self._replace_word(src=src, special=special_char, symbol="*", replace_word="별")
            # 나누기
            if re.search(division, src):
                special_char = re.findall(division, src)
                src = self._replace_word(src=src, special=special_char, symbol="/", replace_word=" 나누기 ")
            # 년 월 일
            if re.search(date, src):
                special_char = re.findall(date, src)
                src = self._replace_date(src=src, special=special_char)
            # 분에
            if re.search(fraction, src):
                special_char = re.findall(fraction, src)
                src = self._replace_fraction(src=src, special=special_char, replace_word=" 분에 ")
            # 은/는
            if re.search(equal, src):
                special_char = re.findall(equal, src)
                src = self._replace_equal(src=src, special=special_char)
            # 시간
            if re.search(time, src):
                special_char = re.findall(time, src)
                src = self._replace_time(src=src, special=special_char)
            # 샵
            if re.search(hash, src):
                special_char = re.findall(hash, src)
                src = self._replace_word(src=src, special=special_char, symbol="#", replace_word="샵 ")
            # 엔
            if re.search(ampersand, src):
                special_char = re.findall(ampersand, src)
                src = self._replace_word(src=src, special=special_char, symbol="&", replace_word=" 엔 ")
            # 이메일: 쩜, 골뱅이
            if re.search(email, src):
                special_char = re.findall(email, src)
                src = self._replace_word(src=src, special=special_char,symbol="@", replace_word=" 골뱅이 ")
                special_char = [re.findall("\.", s) for s in special_char][0]
                src = self._replace_word(src=src, special=special_char, symbol=".", replace_word=" 쩜 ")
            # 밑줄표
            if re.search(underline, src):
                special_char = re.findall(underline, src)
                src = self._replace_word(src=src, special=special_char, symbol="_", replace_word=" 밑줄표 ")

            # 띄어쓰기 하나로
            src = re.sub(r'\s{2,}', " ", src)
            src = src.strip()
            source = KT_TTS(id=id, sent=src)
            ret_src_items.append(source)
        print(ret_src_items)
        return ret_src_items, tgt_tts_items

#### 기본 단어 바꾸기############################################
    def _replace_word(self, src:str, special:List, symbol:str, replace_word:str) -> str:
        for s in special:
            sent = s.replace(symbol, replace_word)
            src = src.replace(s, sent)
        return src
### 날짜 형식으로 바꾸기 #########################################
    def _replace_date(self, src:str, special:List) -> str:
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
### 몇 분의 몇 ##########################################################
    def _replace_fraction(self, src:str, special:List, replace_word:str) -> str:
        for s in special:
            if len(s.split('/')) == 2:
                num1, num2 = s.split('/')
                sent = s.replace(s, num2 + replace_word + num1)
                src = src.replace(s, sent)
            else:
                print('[fraction][error]', src, special)
                sys.exit()
        return src
### = -> 은/는 ######################################################
    def _replace_equal(self, src:str, special:List) -> str:
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
### 시간 형태로 바꾸기 ###############################################
    def _replace_time(self, src:str, special:List) -> str:
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
    kt = KT_TTS_Maker()
    src = KT_TTS(id='0', sent="abc_def")
    tgt = KT_TTS(id='0', sent="십이 더하기 칠 은 십구")
    sources : List[KT_TTS] = []
    targets : List[KT_TTS] = []
    sources.append(src)
    targets.append(tgt)
    kt.get_kt_tts_s2p(sources, targets)
