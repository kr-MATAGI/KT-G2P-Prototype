# KT G2P (Grapheme-to-Phoneme) 변환기

한국어 텍스트를 발음으로 변환하는 G2P(Grapheme-to-Phoneme) 시스템입니다. ELECTRA 기반의 인코더-디코더 모델과 자소제한 후처리 기법을 결합하여 높은 정확도의 발음 변환을 제공합니다.

## 주요 특징

- ELECTRA 기반의 인코더-디코더 구조
- Autoregressive 및 Non-autoregressive 디코더 지원
- 자소제한 후처리 기법 적용
- 앙상블 모델 지원
- WER(Word Error Rate) 및 PER(Phoneme Error Rate) 기반 평가

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 환경 (선택사항)
- PyTorch 1.11.0+cu115

## 설치 방법

### 1. 저장소 클론

```bash
git clone [repository_url]
cd KT-G2P-Prototype
```

### 2. 데이터 및 모델 다운로드

#### 2.1 데이터 다운로드
```bash
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12sTnpTVufzC564rrMP-zjJboDPoRC50K' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12sTnpTVufzC564rrMP-zjJboDPoRC50K" -O data.zip && rm -rf ~/cookies.txt
```

#### 2.2 모델 다운로드
```bash
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nwoGDFb7iqwpqPsW00dL7qw1nKkHazvf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nwoGDFb7iqwpqPsW00dL7qw1nKkHazvf" -O model.zip && rm -rf ~/cookies.txt
```

#### 2.3 압축 해제
- 데이터 파일: `KT_Project/data/` 디렉토리에 압축 해제
- 모델 파일: `KT_Project/test_model/` 디렉토리에 압축 해제

### 3. 환경 설정

#### 방법 1: pip 사용
```bash
pip install -r requirements.txt
```

#### 방법 2: Anaconda 사용
```bash
# 환경 생성
conda env create -f kt_proj.yaml python=3.8

# 환경 활성화
conda activate kt_proj

# PyTorch 설치 (CUDA 11.5 버전)
pip uninstall torch torchaudio torchvision
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu115
```

## 사용 방법

### 1. 설정 파일 수정
`config/kocharelectra_config.json` 파일에서 필요한 설정을 수정합니다:

```json
{
  "ckpt_dir": "ko-char-electra-encoder-decoder",
  "train_npy": "./data/data_busan/kor/npy/train.npy",
  "dev_npy": "./data/data_busan/kor/npy/dev.npy",
  "test_npy": "./data/data_busan/kor/npy/test.npy",
  "do_train": false,
  "do_eval": true,
  "device": "cuda",  // 또는 "cpu"
  ...
}
```

### 2. 모델 실행

#### 기본 G2P 테스트
```bash
python run_g2pk.py
```

#### 임의 문장 테스트
```bash
python ar_test.py --input=안녕하세요 \
                  --ckpt_path=test_model/ckpt-ar-end2end/ko-char-electra-encoder-decoder/checkpoint-17150 \
                  --config_path=config/kocharelectra_config.json \
                  --decoder_vocab_path=data/vocab/decoder_vocab/pron_eumjeol_vocab.json \
                  --jaso_dict_path=data/vocab/post_process/jaso_filter.json
```

## 프로젝트 구조

```
KT-G2P-Prototype/
├── config/                 # 설정 파일
├── data/                   # 데이터셋
├── model/                  # 모델 파일
├── utils/                  # 유틸리티 함수
├── test_utils/            # 테스트 유틸리티
├── run_electra_enc_dec.py # ELECTRA 인코더-디코더
├── run_electra_art_dec.py # Autoregressive 디코더
├── run_electra_nart_dec.py # Non-autoregressive 디코더
├── run_ensemble.py        # 앙상블 모델
├── run_g2pk.py           # 기본 G2P 테스트
└── ar_test.py            # 임의 문장 테스트
```

## 평가 지표

- WER (Word Error Rate): 단어 단위 오류율
- PER (Phoneme Error Rate): 음소 단위 오류율
- Sentence Accuracy: 문장 단위 정확도

## 라이선스

[라이선스 정보 추가 필요]

## 문의사항

[문의처 정보 추가 필요]
