from definition.data_def import DictWordItem, OurSamItem
from utils.db_utils import HeadWord_DB_Maker, IPA_DB_Maker
from definition.db_def import (
    HeadWordItems, ConjuWordItems, NnpWordItmes
)
import re
from definition.alphabet import ALPHABET
from definition.data_def import KT_TTS
from typing import List
from model.transformers.predict import Convert_Eng


class Eng2Kor:
    ''' English word to Korean word '''

    def __init__(self):
        self.headword_db_maker = HeadWord_DB_Maker(db_path='db/final_eng_database.db', table_name='words')
        self.ipa_db_maker = IPA_DB_Maker(db_path='db/ipa_database.db', table_name='words')
        self.headword_dic = self.headword_db_maker.select_all_items()
        self.ipa_db_dic = self.ipa_db_maker.select_all_items()
        self.transformer = Convert_Eng()
        self.our_num, self.ipa_num, self.transformer_num, self.alpha_num = 0, 0, 0, 0
        self.not_match = []

    def convert_eng(self, source: KT_TTS, use_ipa: bool) -> KT_TTS:
        r_eng_pattern = r"[a-zA-Z]+"

        eng_words = re.findall(r_eng_pattern, source.sent)
        for word in eng_words:
            try:
                results = [self.headword_dic[word]]
            except:
                results = []

            if len(results) < 1:
                is_lower = False
                # ignore upper lower case
                for k in self.headword_dic.keys():
                    if str(k).lower() == word.lower():
                        results = [self.headword_dic[k]]
                        is_lower = True
                        break

                if not is_lower:
                    results = []

            if len(results) > 0:
                self.our_num += 1
                source.sent = source.sent.replace(word, str(results[0]), 1)
                continue

            # convert ipa to kor
            if len(results) < 1 and use_ipa:
                try:
                    results = [self.ipa_db_dic[word.lower()]]
                except:
                    results = []
                if len(results) > 0:
                    self.ipa_num += 1
                    source.sent = source.sent.replace(word, results[0], 1)
                    continue

            ''' transformers '''
            if len(results) < 1:
                results = self.transformer.predict(word)
                if len(results) > 0:
                    source.sent = source.sent.replace(word, results[0], 1)
                    self.transformer_num += 1
                    continue

            '''if no results and word is uppercase then read in the alphabet'''
            # if (len(results) < 1 and len(word) < 6 and word.isupper()) or (len(results) < 1 and len(word) == 1):
            if len(results) < 1:
                source.sent = source.sent.replace(word, word.upper())
                word = word.upper()
                alpha_pronun = ""
                alphabet = ALPHABET
                for w in word:
                    alpha_pronun += alphabet[w]
                results = [alpha_pronun]
                source.sent = source.sent.replace(word, results[0], 1)
                self.alpha_num += 1

            source.sent = source.sent.replace(word, results[0], 1)
        return source


if __name__ == '__main__':
    source = '18일 S B S funE 보도에 따르면 피해 여성 A씨는 단톡방에 유포된 음성파일과 사진 이들이 나눈 대화 등을 통해 자신이 이들에게 성폭행을 당한 사실을 뒤늦게 확인했다.'
    tts = KT_TTS(id=1, sent=source)
    eng2kor = Eng2Kor()
    result = eng2kor.convert_eng(tts)
    print(result)