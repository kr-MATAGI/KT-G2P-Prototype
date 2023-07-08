import os
import sqlite3

from abc import *
from db.sql_query import *
from utils.post_method import PreDictItem
from typing import Dict, List

DB_PATH = '../db/dict.db'

#============================================
class DB_Maker(metaclass=ABCMeta):
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        if not os.path.exists(db_path):
            pass
        print(f'{__class__} Set DB path: {db_path}')
        self.db_path = db_path

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
class HeadWord_DB_Maker(DB_Maker):
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        super(HeadWord_DB_Maker, self).__init__(db_path)

#============================================
class ConjuWord_DB_Maker(DB_Maker):
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        super(ConjuWord_DB_Maker, self).__init__(db_path=db_path, table_name=table_name)

    def create_table(self):
        pass

    def insert_table_items(
            self,
            src_items: List
    ):
        pass

#============================================
class NNPWord_DB_Maker(DB_Maker):
#============================================
    def __init__(
            self,
            db_path: str, table_name: str
    ):
        super(NNPWord_DB_Maker, self).__init__(db_path=db_path, table_name=table_name)

    def create_table(self):
        pass

    def insert_table_items(
            self,
            src_items: List
    ):
        pass

#============================================
def create_db(db_path: str):
#============================================
    print(f'[db_utils][create_db] db_path: {db_path}')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(CREATE_SQL)

    conn.close()

#============================================
def insert_items_to_db(db_path: str, dict_items: Dict[str, PreDictItem]):
#============================================
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    insert_data_list = []
    for idx, (key, dict_item) in enumerate(dict_items.items()):
        if 0 == (idx % 50000):
            print(f'[db_utils][insert_items_to_db] {idx} is processing... {key}')

        concat_pronun_list = ','.join(dict_item.pronun_list)
        insert_data_list.append((key, concat_pronun_list, dict_item.pos))
    print(f'[db_utils][insert_items_to_db] all_items.size: {len(insert_data_list)}')
    cursor.executemany(INSERT_SQL, insert_data_list)
    conn.commit()

    cursor.execute(SELECT_ALL_ITEMS)
    item_list = cursor.fetchall()
    print(f'[db_utils][insert_items_to_db] check all_items.size: {len(item_list)}')
    print(item_list[:5])

    conn.close()


### MAIN ###
if '__main__' == __name__:
    print('[db_utils][__main__] MAIN')

    db_maker = DB_Maker(db_path=DB_PATH)
    headword_db_maker = HeadWord_DB_Maker(db_path=DB_PATH)