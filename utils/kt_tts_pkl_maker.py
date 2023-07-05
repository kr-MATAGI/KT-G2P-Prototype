import os
import pickle

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
   ) -> List[KT_TTS]:
        '''
        특수 기호를 발음열 변환된 문장으로 변환

        :return: List[KT_TTS]
        '''

### MAIN ###
if '__main__' == __name__:
    print(f'[kt_tts_pkl_maker][__main__] MAIN !')

