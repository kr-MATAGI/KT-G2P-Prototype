import json
import os
import pickle
import re

from typing import List
from definition.data_def import KrStdDict, OurSamDict, ConjuInfo

import xml.etree.ElementTree as ET


# ===================================
class OurSamMaker:
    # ===================================
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


# ===================================
class StdDictMaker:
    # ===================================
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


### MAIN ###
if "__main__" == __name__:
    print("[dict_maker] __main__")

    is_make_our_sam_dict = True

    if is_make_our_sam_dict:
        our_sam_maker = OurSamMaker()
        # all_sam_data = our_sam_maker.parse_xml_files(dir_path="../data/our_sam/xml",
        #                                              save_path="../data/our_sam/pkl/our_sam_data.pkl")
        our_sam_dict = our_sam_maker.make_our_sam_dict(pkl_path="../data/our_sam/pkl/our_sam_data.pkl")

        # Save
        dict_save_path = "../data/our_sam_filter_dict.json"
        with open(dict_save_path, mode="w", encoding="utf-8") as f:
            json.dump(our_sam_dict, f, ensure_ascii=False, indent=4)