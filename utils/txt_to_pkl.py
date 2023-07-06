import pickle
from definition.data_def import KT_TTS

### MAIN ###
if '__main__' == __name__:
    # source txt read
    with open('../data/kor/raw/kor_source.txt', 'r', encoding='utf-8') as f:
        sources = f.read().strip().split('\n')
    # target txt read
    with open('../data/kor/raw/kor_target.txt', 'r', encoding='utf-8') as f:
        targets = f.read().strip().split('\n')

    # dataclass
    source, target = [], []
    for line in sources:
        id, sent = line.split("\t")
        kt_tts = KT_TTS(id=id, sent=sent)
        source.append(kt_tts)
    for line in targets:
        id, sent = line.split("\t")
        kt_tts = KT_TTS(id=id, sent=sent)
        target.append(kt_tts)

    print(len(source), len(target))

    # txt to pkl
    with open('../data/kor/pkl/kor_source_filter.pkl', 'wb') as f:
        pickle.dump(source, f,  protocol=pickle.HIGHEST_PROTOCOL)
    with open('../data/kor/pkl/kor_target.pkl', 'wb') as f:
        pickle.dump(target, f,  protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/kor/pkl/kor_source_filter.pkl', 'rb') as f:
        source = pickle.load(f)

    with open('../data/kor/pkl/kor_target.pkl', 'rb') as f:
        target = pickle.load(f)
    print(len(source), len(target))
    print(source)