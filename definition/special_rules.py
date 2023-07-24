SYMBOL_RULES = {
    # 플러스
    'r_plus' : r"\++\s*\d+\+*",
    # 에
    'r_eh' : r"(\d+\-\d+\-\d+|\d+\-\d+)",
    # 마이너스
    'r_minus' : r"(\-+\s*)+\d+",
    # 묵음
    'r_blank' : r"[가-힣]+(?:\-[가-힣]*)+|[가-힣]+(?:\/[가-힣]*)+",
    # 곱하기
    'r_multi' : r"\d+\s?\*\s?\d+",
    # 별
    'r_star' : r"\*\d+",
    # 나누기
    'r_division' : r"\d+\s?/\s?\d+\s?=",
    # 년 월 일
    'r_date' : r"\d{2,4}/\d{2}/\d{2}|\d{2,4}\.\d{2}.\d{2}",
    # 분에
    'r_fraction' : r"\d+\s?/\s?\d+\s?",
    # 은/는
    'r_equal' : r"=",
    # 시간
    'r_time' : r"2[0-3]:[0-5][0-9]:[0-5][0-9]|[01][0-9]:[0-5][0-9]:[0-5][0-9]",
    # 샵
    'r_hash' : r"#",
    # 엔
    'r_ampersand' : r"[A-Za-z]&[A-Za-z]",
    # 이메일: 쩜, 골뱅이
    'r_email' : r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    # 밑줄표
    'r_underline' : r"[A-Za-z]+(?:_[A-Za-z]*)+",
    # space
    'r_space' : r"[ㄱ-ㅎ가-힣a-zA-Z]+:[ㄱ-ㅎ가-힣a-zA-Z]+|[ㄱ-ㅎ가-힣a-zA-Z]+,[ㄱ-ㅎ가-힣a-zA-Z]+|[ㄱ-ㅎ가-힣a-zA-Z]+>[ㄱ-ㅎ가-힣a-zA-Z]+",
    # micro 3m
    'r_micro_meter' : r"㎍/㎥",
    # 3m
    'r_meter' : r"㎥",
    # microgram
    'r_micro' : r"㎍",
    # nanogram
    'r_nanogram' : r"ng/㎖",
    # miriliter
    'r_miri' : r"㎖"
}