from definition.data_def import DictWordItem, OurSamItem
from utils.db_utils import HeadWord_DB_Maker
from definition.db_def import (
    HeadWordItems, ConjuWordItems, NnpWordItmes
)
import re
from definition.alphabet import ALPHABET

from typing import List


class Eng2Kor:
    ''' English word to Korean word '''

    def __init__(self):

        self.headword_db_maker = HeadWord_DB_Maker(db_path='db/eng_database.db', table_name='words')

    def convert_eng(self, source: str) -> str:
        r_eng_pattern = r"[a-zA-Z]+"

        eng_words = re.findall(r_eng_pattern, source.sent)

        for word in eng_words:
            eng = HeadWordItems(word=word)
            results = self.headword_db_maker.search_table_items(eng, lower=False)
            if len(results) < 1:
                results = self.headword_db_maker.search_table_items(eng, lower=True)

            # if no results and word is uppercase then read in the alphabet
            if len(results) < 1 and word.isupper():

                alpha_pronun = ""
                alphabet = ALPHABET
                for w in word:
                    alpha_pronun += alphabet[w]
                results = [(alpha_pronun,)]

            # if english word in db
            if len(results) > 0:
                source.sent = source.sent.replace(word, results[0][0])

        return source

if __name__ == '__main__':
    source = 'ABCDEF를 하시는 모습이 꽤나 즐거워 보이세요, 하이!'
    eng2kor = Eng2Kor()
    result = eng2kor.convert_eng(source)
    print(result)