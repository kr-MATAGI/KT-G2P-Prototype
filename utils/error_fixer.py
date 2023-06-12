ERR_SENT_ID_FIXED = {
    '014228': ('애지중지하였었다', '애지중지하여썯따'),
    '030492': ('하는것이', '하는거시'),
    '036172': ('가라안아있네요', '가라안자인네요'),
    '037984': ('가라안았다가', '가라안잗따가'),
    '046654': ('사라지겠군요', '사라지겓꾸뇨'),
    '047445': ('사라지겠네요', '사라지겐네요'),
}

ERR_SENT_CHANGED_FIXED = {
    # id: (input_sent, ans_sent)
    '010752': ('외국인이 티읕 피읖을 발음하기란 참 어렵다', '외구기니 티읃 피읍을 바름하기란 참 어렵따'), # '외국인이 을 발음하기란 참 어렵다', '외구기니 티읃 피읍 을 바름하기란 참 어렵따'
    '017352': ('중 랴오닝 잉커우 우수기업 잇단 유치', '중 랴오닝 잉커우 우수기업 읻딴 유치'), # '랴오닝 잉커우 우수기업 잇단 유치', '중 랴오닝 잉커우 우수기업 읻딴 유치'
    '024001': ('기역 니은 디귿 리을', '기역 니은 디귿 리을'), # '', '기역 니은 디귿 리을'
    '024002': ('미음 비읍 시옷 이응', '미음 비읍 시옫 이응'), # '', '미음 비읍 시옫 이응'
    '024003': ('지읒 치읓 키읔 티읕 피읖 히읗', '지읃 치읃 키윽 티읃 피읍 히읃'), # '', '지읃 치읃 키윽 티읃 피읍 히읃'
    '024004': ('쌍기역 쌍디귿 쌍비읍 쌍시옷 쌍지읒', '쌍기역 쌍디귿 쌍비읍 쌍시옫 쌍지읃'), # '', '쌍기역 쌍디귿 쌍비읍 쌍시옫 쌍지읃'
    '024005': ('아 애 야 얘', '아 애 야 얘'), # '', '아 애 야 얘'
    '024006': ('어 에 여 예', '어 에 여 예'), # '', '어 에 여 예'
    '024007': ('오 와 왜 외 요', '오 와 왜 외 요'), # '', '오 와 왜 외 요'
    '024008': ('우 워 웨 위 유', '우 워 웨 위 유'), # '', '우 워 웨 위 유'
    '024009': ('으 의 이', '으 의 이'), # '', '으 의 이'
    '028512': ('이 대표는 이날 오전 인천 남동구 주식회사 서울화장품에서 열린 현장 최고위원회의에서',
               '이 대표는 이날 오전 인천 남동구 주시쾨사 서울화장푸메서 열린 현장 최고위원회이에서'),
    # '이 대표는 이날 오전 인천 남동구 서울화장품에서 열린 현장 최고위원회의에서', '이 대표는 이날 오전 인천 남동구 주시쾨사 서울화장푸메서 열린 현장 최고위원회이에서'
    '035254': ('이응 가루에 우유 넣어 거품기로 섞어주기', '이응 가루에 우유 너어 거품기로 써꺼주기'),
    # '가루에 우유 넣어 거품기로 섞어주기', '이응 가루에 우유 너어 거품기로 써꺼주기',
    '035736': ('바락바락 여러번 위에서 아래로 뒤적여주세요', '바락빠락 여러번 위에서 아래로 뒤저겨주세요'),
    # '바락바락 여러번 위에서 아래로 뒤적여주세요', '바락빠락 여러번 위에서 아래로 뒤저겨주세요이'
    '037209': ('후라이팬에 오일을 살짝만 두르고 고등어의 등쪽부터 구워주기 시작하세요',
               '후라이패네 오이를 살짱만 두르고 고등어에 등쪽뿌터 구워주기 시자카세요'),
    # '후라이팬에 오일을 살짝만 두르고 고등어의 등쪽부터 구워주기 시작하세요', '후라이패네 오이를 살짱만 두르고 고등어에 등쪽뿌터 구워주기 시자카세요피읍'
    '060001': ('지니 노래 키읔 키읔 밴드 가 버린 당신 가가라이브 가거라 사랑아 가고 싶은 곳이 있어 가고 있어 들려줘',
               '지니 노래 키윽 키윽 밴드 가 버린 당신 가가라이브 가거라 사랑아 가고 시픈 고시 이써 가고 이써 들려줘'),
    # '지니 노래 밴드 가 버린 당신 가가라이브 가거라 사랑아 가고 싶은 곳이 있어 가고 있어 들려줘', '지니 노래 키윽 키윽 밴드 가 버린 당신 가가라이브 가거라 사랑아 가고 시픈 고시 이써 가고 이써 들려줘'
    '061180': ('지니 노래 내 마음 별과 같이 내 마음 별이 되어 내 마음속 이야기 내 마음속 전부를 내 마음 안에 발자국 내 마음대로 안되는 건 너뿐이야 들려줘',
               '지니 노래 내 마음 별과 가치 내 마음 벼리 되어 내 마음쏙 이야기 내 마음쏙 전부를 내 마음 아네 발짜국 내 마음대로 안되는 건 너뿌니야 들려줘'),
    # '지니 노래 내 마음 별과 같이 내 마음 별이 되어 내 마음속 이야기 내 마음속 전부를 내 마음 안에 발자국 내 마음대로 안되는 건 너뿐이야 들려줘', '지니 노래 내 마음 별과 가치 내 마음 벼리 되어 내 마음쏙 이야기 내 마음쏙 전부를 내 마음 아네 발짜국 내 마음대로 안되는 건 너뿌니야유 유 들려줘'
    '061737': ('지니 노래 다음 사람에게는 다음 사람에게는 이진성 다음날 다음날 아침 다음에 또 만나요 다음에 또 봐 들려줘',
               '지니 노래 다음 사라메게는 다음 사라메게는 이진성 다음날 다음날 아침 다으메 또 만나요 다으메 또 봐 들려줘'),
    # '지니 노래 다음 사람에게는 다음 사람에게는 이진성 다음날 다음날 아침 다음에 또 만나요 다음에 또 봐 들려줘', '지니 노래 다음 사라메게는 다음 사라메게는 이진성 다음날 다음날 아침 다으메 또 만나요 다으메 또 봐이응 들려줘'
    '065678': ('지니 노래 어학연수 어허라 사랑 어헤드 어브 마이셀프 어화 둥 둥 억지로웃지않 위치 언 아메리컨 인 패리스 들려줘',
               '지니 노래 어하견수 어허라 사랑 어헤드 어브 마이셀프 어화 둥 둥 억찌로욷찌안 위치 언 아메리컨 인 패리쓰 들려줘'),
    # '지니 노래 어학연수 어허라 사랑 어헤드 어브 마이셀프 어화 둥 둥 억지로웃지않 위치 언 아메리컨 인 패리스 들려줘', '지니 노래 어하견수 어허라 사랑 어헤드 어브 마이셀프 어화 둥 둥 억찌로욷찌안니을 위치 리을 언 아메리컨 인 패리쓰 들려줘'
    '066479': ('지니 노래 웜홀 웨 웨더 웨딩 더 브라이트 웨딩 데이 웨딩드레스 들려줘',
               '지니 노래 웜홀 웨 웨더 웨딩 더 브라이트 웨딩 데이 웨딩드레쓰 들려줘'),
    # '지니 노래 웜홀 웨 웨더 웨딩 더 브라이트 웨딩 데이 웨딩드레스 들려줘', '지니 노래 웜홀 웨에 에 에 에 웨더 웨딩 더 브라이트 웨딩 데이 웨딩드레쓰 들려줘'
    '068249': ('지니 노래 턴잇업 턴코트 털 업해야해 털어 텀블키즈 텁텁 들려줘',
               '지니 노래 터니덥 턴코트 털 어패야해 터러 텀블키즈 텁텁 들려줘'),
    # '지니 노래 턴잇업 턴코트 털 업해야해 털어 텀블키즈 텁텁 들려줘', '지니 노래 터니덥 턴코트 털리은 어패야해 터러 텀블키즈 텁텁 들려줘'
    '081330': ('한국기독교총연합회 한기총', '한국끼도꾜총년하푀 한기총'), # '한국기독교총연합회 한기총', '한국끼도꾜총년하푀한기총'
    '082586': ('아까 면허증 준 사람이에요', '아까 면허쯩 준 사라미에요'), # '아까 면허증 준 사람이에요', '아까 면허쯩 준 사라미에요히읃 히읃'
}