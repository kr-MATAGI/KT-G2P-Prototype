import os
import sqlite3

from abc import *
from db.sql_query import (
    CREATE_TABLE_SQL, INSERT_ITEMS_SQL, SELECT_ALL_ITEMS
)
from typing import Dict, List
from definition.db_def import (
    HeadWordItems, ConjuWordItems, NnpWordItmes, ChnWordItems
)

DB_PATH = '../db/dict.db'

#============================================
class DB_Maker(metaclass=ABCMeta):
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        if not os.path.exists(db_path):
            raise Exception(f'[DB_Maker] ERR - DB_Path: {db_path}')
        self.db_path = db_path
        self.table_name = table_name

    def create_db(
            self, db_path: str
    ):
        pass

    @abstractmethod
    def create_table(self):
        pass

    @abstractmethod
    def insert_table_items(
            self,
            src_items: List
    ):
        pass

    def select_all_items(self):
        pass

#============================================
class HeadWord_DB_Maker:
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        '''
            표제어 기분석 사전
            적용순위 : 3
        '''
        super(HeadWord_DB_Maker, self).__init__()
#         print(f'[db_utils][HeadWord_DB_Maker] db_path: {db_path}, table_name: {table_name}')
        self.db_path = db_path
        self.table_name = table_name

    def create_table(self):
        print(f'[db_utils][HeadWord_DB_Maker]: table_name: {self.table_name}')

        conn = sqlite3.connect(self.table_name)
        cursor = conn.cursor()

        cursor.execute(CREATE_TABLE_SQL['CREATE_TABLE_SQL'])
        conn.close()

        print(f'[db_utils][HeadWord_DB_Maker] Complete !')

    def insert_table_items(
            self,
            src_items: List[HeadWordItems]
    ):
        print(f'[db_utils][HeadWord_DB_Maker] table_name: {self.table_name}')

        conn = sqlite3.connect(self.table_name)
        cursor = conn.cursor()

        insert_data_items = []
        for idx, item in enumerate(src_items):
            if 0 == (idx % 50000):
                print(f'[db_utils][insert_table_items] table_name: {self.table_name}, {item.word}')

            insert_data_items.append(('여기는 테이블 내의 column', '[여기에 db에 들어갈 데이터]'))
        print(f'[db_utils][insert_table_items] all_items.size: {len(insert_data_items)}')
        cursor.executemany(INSERT_ITEMS_SQL['HeadWord'], insert_data_items)
        conn.commit()
        cursor.execute(SELECT_ALL_ITEMS.format(self.table_name))

        ''' db에 들어갔는지 확인 '''
        temp_item_list = cursor.fetchall()
        print(f'[db_utils][insert_table_items] CHECK - all_items.size: {len(temp_item_list)}')
        print(temp_item_list[:5])

        conn.close()
        print(f'[db_utils][insert_table_items] Complete !')

    def search_table_items(self, word_item: HeadWordItems, lower=False):
        '''
            search foreign word
        '''

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if lower:
            query = f"SELECT pronun_list FROM {self.table_name} WHERE word COLLATE NOCASE = '{word_item.word}'"

        else:
            query = f"SELECT pronun_list FROM {self.table_name} WHERE word = '{word_item.word}'"

        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except:
            return None

        cursor.close()
        conn.close()

        return rows

    def select_all_items(self):
        ''' search all items '''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT pronun_list, word FROM {}".format(self.table_name)
        cursor.execute(query)
        rows = cursor.fetchall()
        ret = {}
        for row in rows:
            if row[1] not in ret:
                ret[row[1]] = row[0]

        cursor.close()
        conn.close()

        return ret

#============================================
class IPA_DB_Maker:
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        '''
            표제어 기분석 사전
            적용순위 : 3
        '''
        super(IPA_DB_Maker, self).__init__()
#         print(f'[db_utils][HeadWord_DB_Maker] db_path: {db_path}, table_name: {table_name}')
        self.db_path = db_path
        self.table_name = table_name

    def create_table(self):
        print(f'[db_utils][HeadWord_DB_Maker]: table_name: {self.table_name}')

        conn = sqlite3.connect(self.table_name)
        cursor = conn.cursor()

        cursor.execute(CREATE_TABLE_SQL['CREATE_TABLE_SQL'])
        conn.close()

        print(f'[db_utils][HeadWord_DB_Maker] Complete !')

    def insert_table_items(
            self,
            src_items: List[HeadWordItems]
    ):
        print(f'[db_utils][HeadWord_DB_Maker] table_name: {self.table_name}')

        conn = sqlite3.connect(self.table_name)
        cursor = conn.cursor()

        insert_data_items = []
        for idx, item in enumerate(src_items):
            if 0 == (idx % 50000):
                print(f'[db_utils][insert_table_items] table_name: {self.table_name}, {item.word}')

            insert_data_items.append(('여기는 테이블 내의 column', '[여기에 db에 들어갈 데이터]'))
        print(f'[db_utils][insert_table_items] all_items.size: {len(insert_data_items)}')
        cursor.executemany(INSERT_ITEMS_SQL['HeadWord'], insert_data_items)
        conn.commit()
        cursor.execute(SELECT_ALL_ITEMS.format(self.table_name))

        ''' db에 들어갔는지 확인 '''
        temp_item_list = cursor.fetchall()
        print(f'[db_utils][insert_table_items] CHECK - all_items.size: {len(temp_item_list)}')
        print(temp_item_list[:5])

        conn.close()
        print(f'[db_utils][insert_table_items] Complete !')

    def search_table_items(self, word_item: HeadWordItems):
        '''
            search foreign word
        '''

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = f"SELECT pronun_list FROM {self.table_name} WHERE origin_lang COLLATE NOCASE = '{word_item.word}'"

        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except:
            return None

        cursor.close()
        conn.close()

        return rows

    def select_all_items(self):
        ''' search all items '''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT pronun_list, origin_lang FROM {}".format(self.table_name)
        cursor.execute(query)
        rows = cursor.fetchall()
        ret = {}

        for row in rows:
            if row[1] not in ret:
                ret[row[1]] = row[0]

        cursor.close()
        conn.close()

        return ret


#============================================
class ConjuWord_DB_Maker:
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        '''
            활용어 기분석 사전
            적용순위 : 2
        '''
        super(ConjuWord_DB_Maker, self).__init__()

        print(f'[db_utils][ConjuWord_DB_Maker] db_path: {db_path}, table_name: {table_name}')
        if not os.path.exists(db_path):
            raise Exception('ERR - DB_PATH')

        self.db_path = db_path
        self.table_name = table_name

    def create_table(self):
        pass

    def insert_table_items(
            self,
            src_items: List
    ):
        pass
    def search_table_items(self, word_item: ConjuWordItems):
        '''
            search foreign word
        '''

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"SELECT pronun_list FROM {self.table_name} WHERE word = '{word_item.word}'"
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except:
            return None

        cursor.close()
        conn.close()

        return rows

#============================================
class NNPWord_DB_Maker:
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        '''
            고유어 기분석 사전
            적용순위 : 1
        '''
        super(NNPWord_DB_Maker, self).__init__()
        self.db_path = db_path
        self.table_name = table_name



    def create_table(self):
        pass

    def insert_table_items(
            self,
            src_items: List
    ):
        pass

    def search_table_items(self, word_item: NnpWordItmes):
        '''
            search foreign word
        '''
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"SELECT pronun_list FROM {self.table_name} WHERE word = '{word_item.word}'"
        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows
    def select_all_items(self):
        ''' search all items '''
        print(f'[db_utils][HeadWord_DB_Maker] table_name: {self.table_name} search all')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT pronun_list, word FROM {}".format(self.table_name)
        cursor.execute(query)
        rows = cursor.fetchall()

        ret = {}
        for row in rows:
            ret[row[1]] = row[0]

        cursor.close()
        conn.close()

        return ret


#============================================
class ChnWord_DB_Maker:
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        '''
            Chinese 기분석 사전

        '''
        super(ChnWord_DB_Maker, self).__init__()

        print(f'[db_utils][ConjuWord_DB_Maker] db_path: {db_path}, table_name: {table_name}')
        if not os.path.exists(db_path):
            raise Exception('ERR - DB_PATH')

        self.db_path = db_path
        self.table_name = table_name

    def create_table(self):
        pass

    def insert_table_items(
            self,
            src_items: List
    ):
        pass
    def search_table_items(self, word_item: ChnWordItems):
        '''
            search foreign word
        '''

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = f"SELECT pronun_list FROM {self.table_name} WHERE word = '{word_item.word}'"
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except:
            return None

        cursor.close()
        conn.close()

        return rows
    def select_all_items(self):
        ''' search all items '''
        print(f'[db_utils][HeadWord_DB_Maker] table_name: {self.table_name} search all')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = "SELECT * FROM {}".format(self.table_name)
        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            print(row)

        cursor.close()
        conn.close()

        return rows

### MAIN ###aa
if '__main__' == __name__:
    print('[db_utils][__main__] MAIN')

    '''
        어떤 기분석 사전 DB를 만들지 설정
    '''

    # db_maker = DB_Maker(db_path=DB_PATH)
    # headword_db_maker = HeadWord_DB_Maker(db_path=DB_PATH)
    ipa_db_maker = ChnWord_DB_Maker(db_path='../db/chn_database.db', table_name='words')
    # tmp = NNPWord_DB_Maker(word="G")
    # chn_db_maker = ChnWord_DB_Maker(db_path='../db/chn_database.db', table_name='words')
    tmp = ChnWordItems(word='小')
    results = ipa_db_maker.search_table_items(tmp)
    print(results)
    print(ipa_db_maker.select_all_items())