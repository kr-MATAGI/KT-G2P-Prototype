import sqlite3
import os
from db.sql_query import *
from utils.post_method import PreDictItem
from typing import List, Dict

DB_PATH = '../db/dict.db'

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
<<<<<<< HEAD
        if 0 == (idx % 50000):
=======
        if 0 == (idx % 1000):
>>>>>>> ca717a8b9243084d1d06c5d36a3cf83684799cae
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
    print('[db_utils][__main__]')

    create_db(db_path=DB_PATH)