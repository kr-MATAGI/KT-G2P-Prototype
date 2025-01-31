import os
import pickle

from typing import List
from definition.data_def import KT_TTS

#============================================================
class DataSpliter:
#============================================================
    def __init__(self):
        print(f'[DataSpliter][__init__] Get instance !')

    def make_split_data(
            self,
            src_data_path: str, tgt_data_path: str, save_root_path: str,
            train_ratio: int=8, dev_ratio: int=1, test_ratio: int=1
    ):
        print(f'[DataSpliter][make_split_data] src_data_path: {src_data_path}\ntgt_data_path: {tgt_data_path}\n'
              f'save_root_path: {save_root_path}')
        print(f'[DataSpliter][make_split_data] ratio - train: {train_ratio}, dev: {dev_ratio}, test: {test_ratio}')

        if not os.path.exists(src_data_path):
            raise Exception(f'ERR - src_data_path: {src_data_path}')
        if not os.path.exists(tgt_data_path):
            raise Exception(f'ERR - tgt_data_path: {tgt_data_path}')
        if not os.path.exists(save_root_path):
            raise Exception(f'ERR - save_root_path: {save_root_path}')

        all_src_data: List[KT_TTS] = []
        with open(src_data_path, mode='rb') as f:
            all_src_data = pickle.load(f)
            print(f'[DataSpliter][make_split_data] all_src_data.size: {len(all_src_data)}')
            print(f'{all_src_data[:10]}')

        all_tgt_data: List[KT_TTS] = []
        with open(tgt_data_path, mode='rb') as f:
            all_tgt_data = pickle.load(f)
            print(f'[DataSpliter][make_split_data] all_tgt_data.size: {len(all_tgt_data)}')
            print(f'{all_tgt_data[:10]}')

        assert len(all_src_data) == len(all_tgt_data), 'ERR - src_data.size != tgt_data.size'

        # Split
        save_items = {
            'train_src': [], 'train_tgt': [],
            'dev_src': [], 'dev_tgt': [],
            'test_src': [], 'test_tgt': []
        }

        total_size = len(all_src_data)
        base_ratio = 0.1
        train_e_idx = int(total_size * (base_ratio * train_ratio))
        dev_e_idx = train_e_idx + int(total_size * (base_ratio * dev_ratio))
        print(f'[DataSpliter][make_split_data] total_size: {total_size}, base_ratio: {base_ratio}')
        print(f'[DataSpliter][make_split_data] train_e_idx: {train_e_idx}, dev_e_idx: {dev_e_idx}')

        save_items['train_src'] = all_src_data[:train_e_idx]
        save_items['train_tgt'] = all_tgt_data[:train_e_idx]
        save_items['dev_src'] = all_src_data[train_e_idx:dev_e_idx]
        save_items['dev_tgt'] = all_tgt_data[train_e_idx:dev_e_idx]
        save_items['test_src'] = all_src_data[dev_e_idx:]
        save_items['test_tgt'] = all_tgt_data[dev_e_idx:]

        # Save
        for k, v in save_items.items():
            with open(save_root_path + '/' + k + '.pkl', mode='wb') as f:
                pickle.dump(v, f)
            print(f'[DataSpliter][make_split_data] {k}.size: {len(v)}')


### MAIN ###
if '__main__' == __name__:
    data_spliter = DataSpliter()

    data_spliter.make_split_data(
        src_data_path='../data/tts_script_85ks_ms949_200407.pkl',
        tgt_data_path='../data/tts_script_85ks_ms949_200506_g2p.pkl',
        save_root_path='../data/raw_split'
    )

