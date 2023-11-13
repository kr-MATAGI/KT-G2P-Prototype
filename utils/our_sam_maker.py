import json
import os
import pickle
import copy
import re

from definition.data_def import KrStdDict, OurSamDict, ConjuInfo, WordInfo, DictWordItem

import xml.etree.ElementTree as ET
from typing import List, Dict
from collections import deque

''' XML 버전 '''
#===================================
class OurSamMaker:
#===================================
    def __init__(self):
        print("[OurSamMaker][__init__]")

    def parse_xml_files(self, dir_path: str, save_path: str):
        print(f"[OutSamMaker][parse_xml_files] dir_path: {dir_path}")

        ret_all_data: List[OurSamDict] = []

        xml_files = os.listdir(dir_path)
        for f_idx, file_name in enumerate(xml_files):
            if ".xml" not in file_name:
                continue
            print(file_name)
            if 0 == (f_idx % 10):
                print(f"[OutSamMaker][parse_xml_files] f_idx: {f_idx} is processing...")
            full_path = dir_path + "/" + file_name
            root = self._load_xml_file(full_path)

            for child in root:
                if "item" != child.tag:
                    continue
                target_code = child.find("target_code").text
                word_info_obj = child.find("wordInfo")

                # word info
                word, pronun = self._parse_word_info(word_info_obj)
                if not word or 0 >= len(pronun):
                    continue

                # sense info
                sense_info_obj = child.find("senseInfo")
                sense_no, pos, cat = self._parse_sense_info(sense_info_obj)
                if not pos:
                    pos = ""

                # conju_info - 활용과 준말을 포함하는 것
                conju_info_obj = word_info_obj.findall("conju_info")
                conjugation, abbreviation = self._parse_conju_info(all_conju_info=conju_info_obj)

                ret_all_data.append(OurSamDict(
                    target_code=target_code, word=word, pronun=pronun,
                    sense_id=sense_no, pos=pos, cat=cat,
                    conjugation=conjugation, abbreviation=abbreviation
                ))
            # end loop, root
        # end loop, xml_files
        print(f"[OutSamMaker][parse_xml_files] ret_all_data.size: {len(ret_all_data)}")

        # save our sam
        with open(save_path, mode="wb") as f:
            pickle.dump(ret_all_data, f)
        print(f"[OurSamMaker][parse_xml_files] save path: {save_path}")

        return ret_all_data

    def make_our_sam_dict(self, pkl_path: str):
        print(f"[OutSamMaker][make_our_sam_dict] pkl_path: {pkl_path}")

        all_our_sam_data: List[OurSamDict] = []
        with open(pkl_path, mode="rb") as f:
            all_our_sam_data = pickle.load(f)
        print(f"[OutSamMaker][make_our_sam_dict] all_our_sam_data.size: {len(all_our_sam_data)}")
        print(all_our_sam_data[:10])

        filtered_our_sam_data: List[OurSamDict] = []
        word_sense_id_dict = {} # key: word, val : List[sense_id]
        for our_sam_data in all_our_sam_data:
            # 우리말 샘 사전에서 word_info 부분에 담겨있는 것.
            our_sam_data.word = our_sam_data.word.replace("-", "").replace("^", "")
            our_sam_data.pronun = [x.replace("ː", "").replace(":", "") for x in our_sam_data.pronun]
            if not re.match("[가-힣]+", our_sam_data.word):
                continue
            is_skip = False
            for pr in our_sam_data.pronun:
                if not re.match("[가-힣]+", pr):
                    is_skip = True
                    break
            if is_skip:
                continue
            filtered_our_sam_data.append(our_sam_data)

            # 활용형에 대한 처리
            for conju_item in our_sam_data.conjugation:
                conju_data = OurSamDict(target_code=our_sam_data.target_code,
                                        word=conju_item.word.replace("-", "").replace("^", ""),
                                        sense_id=our_sam_data.sense_id,
                                        pronun=[x.replace("ː", "").replace(":", "") for x in conju_item.pronun],
                                        pos=our_sam_data.pos, cat=our_sam_data.cat)
                filtered_our_sam_data.append(conju_data)
                if conju_data.word not in word_sense_id_dict.keys():
                    word_sense_id_dict[conju_data.word] = [conju_data.sense_id]
                else:
                    word_sense_id_dict[conju_data.word].append(conju_data.sense_id)
            for abbrev_item in our_sam_data.abbreviation:
                conju_data = OurSamDict(target_code=our_sam_data.target_code,
                                        word=abbrev_item.word.replace("-", "").replace("^", ""),
                                        sense_id=our_sam_data.sense_id,
                                        pronun=[x.replace("ː", "").replace(":", "") for x in abbrev_item.pronun],
                                        pos=our_sam_data.pos, cat=our_sam_data.cat)
                filtered_our_sam_data.append(conju_data)
                if conju_data.word not in word_sense_id_dict.keys():
                    word_sense_id_dict[conju_data.word] = [conju_data.sense_id]
                else:
                    word_sense_id_dict[conju_data.word].append(conju_data.sense_id)

            # sense_id 개수를 통해 동음이의어 구분
            if our_sam_data.word not in word_sense_id_dict.keys():
                word_sense_id_dict[our_sam_data.word] = [our_sam_data.sense_id]
            else:
                word_sense_id_dict[our_sam_data.word].append(our_sam_data.sense_id)
        # end loop, all_our_sam_data
        print(f"[OutSamMaker][make_our_sam_dict] filtered_our_sam_data.size: {len(filtered_our_sam_data)}")
        print(f"[OutSamMaker][make_our_sam_dict] word_sense_id_dict.size: {len(word_sense_id_dict.keys())}")

        ret_dict = {}
        pos_set = []
        for filter_data in filtered_our_sam_data:
            ''' 1. 동음이의어가 있으면 사용하지 않는다 '''
            sense_id_size = len(word_sense_id_dict[filter_data.word])
            if 1 < sense_id_size:
                continue

            ''' 2. 지명 제외 (KT-TTS에서는 발음열이 연음 반영 안되있는 걸로 보임) '''
            if "지명" == filter_data.cat:
                continue

            ''' 3. 품사 필터링 '''
            if filter_data.pos not in ['명사', '대명사']:
                continue

            ''' 4. word가 없는 것 제외 '''
            if 0 >= len(filter_data.word.strip()):
                continue

            pos_set.append(filter_data.pos)
            ret_dict[filter_data.word.replace("\t", "").strip()] = filter_data.pronun[0]
        pos_set = list(set(pos_set))

        print(f"[OutSamMaker][make_our_sam_dict] pos_set: {pos_set}")
        print(f"[OutSamMaker][make_our_sam_dict] ret_dict.keys.size: {len(list(ret_dict.keys()))}")
        print([(k, v) for k, v in ret_dict.items()][:5])
        return ret_dict

    def _parse_sense_info(self, sense_info_obj: ET.Element):
        sense_id = sense_info_obj.findtext("sense_no")
        pos = sense_info_obj.findtext("pos")
        cat_info = sense_info_obj.find("cat_info")
        cat = ""
        if None != cat_info:
            cat = cat_info.findtext("cat")

        return sense_id, pos, cat

    def _parse_word_info(self, word_info_obj: ET.Element):
        word = word_info_obj.findtext("word")
        pronun_list = []
        all_pronun_info = word_info_obj.findall("pronunciation_info")
        for pronun_info in all_pronun_info:
            pronun_list.append(pronun_info.findtext("pronunciation"))

        return word, pronun_list

    def _parse_conju_info(self, all_conju_info: List[ET.Element]):
        ret_conjugation_list: List[ConjuInfo] = []
        ret_abbreviation_list: List[ConjuInfo] = []

        for conju_info in all_conju_info:
            all_conjugation_info = conju_info.findall("conjugation_info")
            all_abbreviation_info = conju_info.findall("abbreviation_info")

            # conjugation
            for conjugation_info in all_conjugation_info:
                conju_data = ConjuInfo()

                conju_text = conjugation_info.findtext("conjugation")
                conju_data.word = conju_text

                all_pronunc = conjugation_info.find("pronunciation_info")
                if None == all_pronunc:
                    break
                for pronunc in all_pronunc:
                    conju_data.pronun.append(pronunc.text)

                ret_conjugation_list.append(conju_data)

            # abbreviation
            for abbrev_info in all_abbreviation_info:
                conju_data = ConjuInfo()

                abbrev_text = abbrev_info.findtext("abbreviation")
                conju_data.word = abbrev_text

                all_pronunc = abbrev_info.find("pronunciation_info")
                if None == all_pronunc:
                    break
                for pronunc in all_pronunc:
                    conju_data.pronun.append(pronunc.text)

                ret_abbreviation_list.append(conju_data)

        return ret_conjugation_list, ret_abbreviation_list

    def _load_xml_file(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return root


#===================================
class StdDictMaker:
#===================================
    def __init__(self):
        print("[StdDictMaker][__init__]")

    def make_kr_std_dict(self, src_path: str):
        print(f"[StdDictMaker][make_kr_std_dict] src_path: {src_path}")
        dict_data = self._load_tsv(src_path)
        print(f"[StdDictMaker][make_kr_std_dict] raw_data.size: {len(dict_data)}")
        print(dict_data[:5])

        return dict_data

    def filter_kr_std_dict(self, origin_dict: List[KrStdDict]):
        print(f"[StdDictMaker][filter_kr_std_dict] origin_dict.size: {len(origin_dict)}")

        filtered_dict = []
        for ori_idx, ori_data in enumerate(origin_dict):
            ''' 1. 발음열 데이터 0이면 사용하지 않는다. '''
            if '0' == ori_data.pronun:
                continue

            ''' 2. 발음열이 '/'로 나누어지며 2개 이상이면 첫 번째 발음열을 사용한다. '''
            if '/' in ori_data.pronun and 2 <= len(ori_data.pronun.split('/')):
                ori_data.pronun = ori_data.pronun.split('/')[0]

            ''' 3. 외래어는 발음의 차이가 클 것으로 보여 일단 사용 X '''
            if re.search(r'[a-z][A-Z]+', ori_data.origin):
                # 현재 filter 되는 단어 없음
                continue

            ''' 4. ':' 기호 삭제 '''
            if ':' in ori_data.pronun:
                ori_data.pronun = ori_data.pronun.replace(':', '')

            ''' 5.1 발음열에 한글만 포함된 데이터만 사용 '''
            if not re.match("[가-힣]+", ori_data.pronun):
                continue
            ''' 5.2 utf-8로 제대로 변환되지 않은게 있다면 사용하지 않음  '''
            if 0 != (str(ori_data.pronun.encode()).count("\\x") % 3):
                continue

            ''' 6. 동음이의어가 있으면 사용하지 않음 '''
            if re.match("[가-힣]+[0-9]+", ori_data.lexi_super):
                continue

            filtered_dict.append(ori_data)

        print(f"[StdDictMaker][filter_kr_std_dict] filtered_dict.size: {len(filtered_dict)}")
        print(filtered_dict[:5])
        return filtered_dict

    def _load_tsv(self, src_path: str):
        dict_data: List[KrStdDict] = []
        with open(src_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [x.replace("\n", "").split("\t") for x in lines]
            dict_data.extend(
                [KrStdDict(id=int(x[0]), word=x[1], lexi_super=x[2], origin=x[3], pronun=x[4]) for x in lines if
                 5 < len(x)])
        return dict_data

#======================================================
class OurSamMerger:
#======================================================
    def __init__(self):
        print(f'[OurSamMerger][__init__] INIT !')

    def merge_kor_eng_dict(
            self,
            kor_path: str, eng_path: str
    ):
        # init
        print(f'[OurSamMerger][merge_kor_eng_dict] kor_path: {kor_path}\neng_path: {eng_path}')

        ret_merged_dict = {}

        # Load *.pkl
        kor_dict: Dict[str, List] = None
        eng_dict: List[WordInfo] = None
        with open(kor_path, mode='rb') as k_f:
            kor_dict = pickle.load(k_f)
            print(f'[OurSamMerger][merge_kor_eng_dict] kor_dict.size: {len(kor_dict)}')
            print(list(kor_dict.items())[:10])
        with open(eng_path, mode='rb') as e_f:
            eng_dict = pickle.load(e_f)
            print(f'[OurSamMerger][merge_kor_eng_dict] eng_dict.size: {len(eng_dict)}')
            print(f'{eng_dict[:10]}')

        # Add - kor word
        for k_key, k_val in kor_dict.items():
            ret_merged_dict[k_key] = k_val
        print(f'[OurSamMerger][merge_kor_eng_dict] Add kor, ret_merge_dict.size: {len(ret_merged_dict.keys())}')

        # Add - etc word
        duplicated_cnt = 0
        for eng_item in eng_dict:
            if eng_item.word in ret_merged_dict.keys():
                duplicated_cnt += 1
                # print(eng_item)
            elif 0 < len(eng_item.pronunciation) and 0 < len(eng_item.word):
                ret_merged_dict[eng_item.word] = eng_item.pronunciation
        print(f'[OurSamMerger][merge_kor_eng_dict] Add eng, ret_merge_dict.size: {len(ret_merged_dict.keys())}')
        print(f'[OurSamMerger][merge_kor_eng_dict] duplicated_cnt: {duplicated_cnt}')

        return ret_merged_dict

''' JSON 버전 '''
#======================================================
class OurSamMakerByJson:
#======================================================
    def __init__(self):
        print(f'[OurSamMakerByJson][__init__] __init__ !')
        self.TARGET_INFO = {
            'word_type': ['고유어', '한자어', '외래어'],
            'pos': ['명사', '동사', '형용사'],
            'sense_no': ['001'],
            'sense_type': ['일반어'],
            'origin_lang_type': ['영어', '고유어', '한자어']
        }
        print(f'[OurSamMakerByJson][__init__] Target_info:\n{self.TARGET_INFO}')

    def make_dict_word_item_list(
            self,
            raw_json_dir_path: str
    ) -> List[DictWordItem]:
        print(f'[OurSamMakerByJson][make_dict_word_item_list] raw_json_dir_path: {raw_json_dir_path}')
        if not os.path.exists(raw_json_dir_path):
            raise Exception(f'ERR - raw_json_dir_path: {raw_json_dir_path}')

        raw_json_files = [x for x in os.listdir(raw_json_dir_path) if '.json' in x]
        print(f'[OurSamMakerByJson][make_dict_word_item_list] raw_json_files.size: {len(raw_json_files)}')

        raw_word_item_list: List[DictWordItem] = []
        for f_idx, file_name in enumerate(raw_json_files):
            print(f'[OurSamMakerByJson][make_dict_word_item_list] f_idx: {f_idx}, file_name: {file_name}')
            root_data = None
            with open(raw_json_dir_path + '/' + file_name, mode='r') as f:
                root_data = json.load(f)

            item_arr = root_data['channel']['item']
            print(f'[OurSamMakerByJson][make_dict_word_item_list] item_arr.size: {len(item_arr)}')

            for item_idx, item_obj in enumerate(item_arr):
                dic_word_item = DictWordItem(
                    word=item_obj['wordinfo']['word'],
                    word_type=item_obj['wordinfo']['word_type'],
                    word_unit=item_obj['wordinfo']['word_unit']
                )

                if 'conju_info' in item_obj['wordinfo'].keys():
                    for conju_item in item_obj['wordinfo']['conju_info']:
                        if ('abbreviation_info' in conju_item.keys()) \
                                and ('pronunciation_info' in conju_item['abbreviation_info'].keys()):
                            abbreviation_info_obj = conju_item['abbreviation_info']
                            dic_word_item.conju_list.append((abbreviation_info_obj['abbreviation'],
                                                             abbreviation_info_obj['pronunciation_info']['pronunciation']))
                        if ('conjugation_info' in conju_item.keys()) \
                                and ('pronunciation_info' in conju_item['conjugation_info'].keys()):
                            conjugation_info_obj = conju_item['conjugation_info']
                            dic_word_item.conju_list.append((conjugation_info_obj['conjugation'],
                                                             conjugation_info_obj['pronunciation_info']['pronunciation']))

                if 'pronunciation_info' in item_obj['wordinfo'].keys():
                    for p_info in item_obj['wordinfo']['pronunciation_info']:
                        dic_word_item.pronun_list.append(p_info['pronunciation'])
                else:
                    dic_word_item.pronun_list = dic_word_item.word

                if 'original_language_info' in item_obj['wordinfo'].keys():
                    dic_word_item.origin_lang = item_obj['wordinfo']['original_language_info'][0]['original_language']
                    dic_word_item.origin_lang_type = item_obj['wordinfo']['original_language_info'][0]['language_type']

                if 'senseinfo' in item_obj.keys():
                    dic_word_item.sense_no = item_obj['senseinfo']['sense_no']
                    if 'pos' in item_obj['senseinfo'].keys():
                        dic_word_item.pos = item_obj['senseinfo']['pos']

                    if 'type' in item_obj['senseinfo'].keys():
                        dic_word_item.sense_type = item_obj['senseinfo']['type']

                raw_word_item_list.append(copy.deepcopy(dic_word_item))
            # end loop, item
        # end loop, file
        print(f'[OurSamMakerByJson][make_dict_word_item_list] raw_word_item_list.size: {len(raw_word_item_list)}')

        return raw_word_item_list

    def get_filtered_word_item(
            self,
            dict_word_item_list: List[DictWordItem]
    ):
        print(f'[OurSamMakerByJson][get_filtered_word_item] dict_word_item_list.size: {len(dict_word_item_list)}')

        deq_word_item_list = deque(dict_word_item_list)
        deque_size = len(deq_word_item_list)
        for _ in range(deque_size):
            curr_item = deq_word_item_list.popleft()

            ''' 혼종어를 제외한 갯수 ''' # 개별로 했을 때 1,164,952 -> 897,693
            if curr_item.word_type not in self.TARGET_INFO['word_type']:
                continue

            ''' 명사만을 추출 ''' # 개별로 했을 때 1,164,952 -> 563,694
            if curr_item.pos not in self.TARGET_INFO['pos']:
                continue
            # 혼종어 제외, 명사만 추출 개수: 1,164,952 -> 500,296

            ''' sense id == 001 ''' # 위에꺼까지 종합: 1,164,952 -> 360,705
            if (curr_item.sense_no not in self.TARGET_INFO['sense_no']) and ('' != curr_item.sense_no):
                continue

            ''' 일반어 ''' # 1,164,952 -> 268,343
            if curr_item.sense_type not in self.TARGET_INFO['sense_type']:
                continue

            deq_word_item_list.append(curr_item)
        print(f'[OurSamMakerByJson][get_filtered_word_item] deq_word_item_list.size: {len(deq_word_item_list)}')

        ''' word나 pron_list에서 특수 기호 제거 '''
        for word_item in deq_word_item_list:
            word_item.word = word_item.word.replace('^', ' ').replace('-', '')
            if not isinstance(word_item.pronun_list, list):
                # str -> list
                word_item.pronun_list = [word_item.pronun_list.replace('^', ' ').replace('-', '')]
            else:
                for p_idx, proun_item in enumerate(word_item.pronun_list):
                    word_item.pronun_list[p_idx] = re.sub(r'[^가-힣]+', '', proun_item)

        return deq_word_item_list

    def get_splited_kor_eng_dict(
            self,
            src_dict_path: str
    ):
        print(f'[OurSamMakerByJson][get_splited_kor_eng_dict] src_dict_path: {src_dict_path}')

        # init
        kor_dict: List[DictWordItem] = []
        eng_dict: List[DictWordItem] = []

        raw_dict: List[DictWordItem] = []
        with open(src_dict_path, mode='rb') as r_f:
            raw_dict = pickle.load(r_f)
        print(f'[OurSamMakerByJson][get_splited_kor_eng_dict] raw_dict.size: {len(raw_dict)}')

        # split
        for raw_idx, raw_item in enumerate(raw_dict):
            if '' == raw_item.origin_lang_type:
                kor_dict.append(raw_item)
            elif '영어' == raw_item.origin_lang_type:
                eng_dict.append(raw_item)

        print(f'[OurSamMakerByJson][get_splited_kor_eng_dict] kor_dict.size: {len(kor_dict)}')
        print(f'[OurSamMakerByJson][get_splited_kor_eng_dict] eng_dict.size: {len(eng_dict)}')

        return kor_dict, eng_dict

    def make_lang_item_info_json(
            self,
            target_dict: List[DictWordItem],
            lang: str
    ):
        print(f'[OurSamMakerByJson][make_lang_item_info_json] lang: {lang}, target_dict.size: {len(target_dict)}')

        json_format = {
            'root': []
        }

        target_dict.sort(key=lambda x: x.word)
        for item in target_dict:
            json_format['root'].append(item.to_json(ensure_ascii=False))
        print(f'[OurSamMakerByJson][make_lang_item_info_json] json_format.root.size: {len(json_format["root"])}')

        return json_format

    def get_word_type_cnt_info(
            self,
            target_dict: List[DictWordItem],
            target_word_type: str
    ):
        print(f'[OurSamMakerByJson][get_word_type_cnt_info] target_dict.size: {len(target_dict)}, '
              f'target_word_type: {target_word_type}')

        ret_cnt = 0
        for idx, dict_item in enumerate(target_dict):
            if target_word_type == dict_item.word_type:
                ret_cnt += 1
        print(f'[OurSamMakerByJson][get_word_type_cnt_info] target_lang_type: {target_word_type}, count: {ret_cnt}')

        return ret_cnt

    def get_conju_items_count(
            self,
            target_dict: List[DictWordItem],
    ):
        print(f'[OurSamMakerByJson][get_conju_items_count] target_dict.size: {len(target_dict)}')

        ret_cnt = 0
        for idx, dict_items in enumerate(target_dict):
            ret_cnt += len(dict_items.conju_list)
        print(f'[OurSamMakerByJson][get_conju_items_count] conju_list.size: {ret_cnt}')

        return ret_cnt

### MAIN ###
if "__main__" == __name__:
    print("[dict_maker] __main__")

    b_use_xml_version_maker = False
    b_use_json_version_maker = True

    if b_use_xml_version_maker:
        is_make_our_sam_dict = False
        is_merge_kor_eng = True

        if is_make_our_sam_dict:
            our_sam_maker = OurSamMaker()
            # all_sam_data = our_sam_maker.parse_xml_files(dir_path="../data/our_sam/xml",
            #                                              save_path="../data/our_sam/pkl/our_sam_data.pkl")
            our_sam_dict = our_sam_maker.make_our_sam_dict(pkl_path="../data/our_sam/pkl/our_sam_data.pkl")

            # Save
            dict_save_path = "../data/our_sam_filter_dict.json"
            with open(dict_save_path, mode="w", encoding="utf-8") as f:
                json.dump(our_sam_dict, f, ensure_ascii=False, indent=4)

        if is_merge_kor_eng:
            our_sam_merger = OurSamMerger()
            merged_our_sam_dict = our_sam_merger.merge_kor_eng_dict(
                kor_path='../data/dictionary/our_sam_std_dict.pkl',
                eng_path='../data/dictionary/dictionary.pkl'
            )

            # save merge_dict
            save_path = '../data/dictionary/merged_kor_eng_info.pkl'
            with open(save_path, mode="wb") as merge_f:
                pickle.dump(merged_our_sam_dict, merge_f)
                print(f'[our_sam_maker][__main__] Complete svae - {save_path}')
                print(f'[our_sam_maker][__main__] merged_dict.size: {len(merged_our_sam_dict)}')

            # to txt file
            with open('../data/dictionary/merged_kor_eng_info.txt', mode='w', encoding='utf-8') as f:
                for key, val in merged_our_sam_dict.items():
                    f.write(key + ':' + val[0] + '\n')

    if b_use_json_version_maker:
        print(f'[our_sam_maker][__main__] maker - json_version')
        dict_maker_json_ver = OurSamMakerByJson()

        raw_word_item_pkl_path = '../data/dictionary/raw_dict_word_item.pkl'
        filtered_word_item_pkl_path = '../data/dictionary/filtered_dict_word_item.pkl'
        b_make_raw_word_item_list = False

        if b_make_raw_word_item_list:
            word_item_list = dict_maker_json_ver.make_dict_word_item_list(raw_json_dir_path='../data/our_sam')
            print(f'[our_sam_maker][__main__] word_item_list.size: {len(word_item_list)}')

            ''' Save '''
            with open(raw_word_item_pkl_path, mode='wb') as f:
                pickle.dump(word_item_list, f)

        raw_word_item_list = []
        with open(raw_word_item_pkl_path, mode='rb') as f:
            raw_word_item_list = pickle.load(f)
        print(f'[our_sam_maker][__main__] raw_word_item_list.size: {len(raw_word_item_list)}')

        dict_word_item_list = dict_maker_json_ver.get_filtered_word_item(dict_word_item_list=raw_word_item_list)
        print(f'[our_sam_maker][__main__] dict_word_item_list.size: {len(dict_word_item_list)}')

        ''' Save '''
        with open(filtered_word_item_pkl_path, mode='wb') as f:
            ''' 정렬 ! '''
            list(dict_word_item_list).sort(key=lambda x: x.word)
            pickle.dump(dict_word_item_list, f)

        #============================================================
        ''' 만들어진 기분석 사전에서 한글과 영어 분리하기'''
        kor_dict, eng_dict = dict_maker_json_ver.get_splited_kor_eng_dict(src_dict_path=filtered_word_item_pkl_path)

        ''' Save '''
        with open('../data/dictionary/kor_dict.json', mode='w', encoding='utf-8') as k_f:
            kor_save_json = dict_maker_json_ver.make_lang_item_info_json(target_dict=kor_dict, lang='kor')
            json.dump(kor_save_json, k_f, indent=4, ensure_ascii=False)
            print(f'[our_sam_maker][__main__] Save Complete ! - Kor Dict')

        with open('../data/dictionary/eng_dict.json', mode='w', encoding='utf-8') as e_f:
            eng_save_json = dict_maker_json_ver.make_lang_item_info_json(target_dict=eng_dict, lang='eng')
            json.dump(eng_save_json, e_f, indent=4, ensure_ascii=False)
            print(f'[our_Sam_maker][__main__] Save Complete ! - Eng Dict')

        dict_maker_json_ver.get_word_type_cnt_info(target_dict=list(dict_word_item_list), target_word_type='고유어')
        dict_maker_json_ver.get_word_type_cnt_info(target_dict=list(dict_word_item_list), target_word_type='한자어')
        dict_maker_json_ver.get_word_type_cnt_info(target_dict=list(dict_word_item_list), target_word_type='외래어')

        dict_maker_json_ver.get_conju_items_count(target_dict=list(dict_word_item_list))