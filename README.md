## Autoregressive + 자소제한 후처리 기법을 적용한 한국어 발음 변환기

### 실행하기전에...

### Git Access Token
 
 - Windows에서는 아래 Accesss 토큰 없이 다운로드가 가능한 것으로 보이나 Linux에서는 아래와 같이 password에 token 입력이 필요합니다.
 
 Username for ‘https://github.com’: <br>
 Password for ‘https://Username@github.com:

<br>

 ```
  ---비공개---
 ```
<br>

1. 실험에 필요한 데이터를 다운로드 해주세요.

  ```
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12sTnpTVufzC564rrMP-zjJboDPoRC50K' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12sTnpTVufzC564rrMP-zjJboDPoRC50K" -O data.zip && rm -rf ~/cookies.txt
  ```
<br>

1.2 테스트에 필요한 모델을 다운로드 해주세요

  ```
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nwoGDFb7iqwpqPsW00dL7qw1nKkHazvf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nwoGDFb7iqwpqPsW00dL7qw1nKkHazvf" -O model.zip && rm -rf ~/cookies.txt
  ```
<br>

1.3 다운로드가 완료된 후 아래 경로에 압축해제 해주세요.

 * 모든 파일은 덮어쓰기 해주세요.
 * 데이터 파일은 KT_Project/data 에서 압축해제
 * 모델 파일은 KT_Project/test_model/ 에서 압축해제

<br>

2. 실행에 필요한 requirements를 설치해주세요.

&nbsp;&nbsp;&nbsp;&nbsp; - requirements.txt를 이용해 설치 혹은 Anaconda 환경을 import 해서 필요한 패키지들을 설치해 주세요. <br>
&nbsp;&nbsp;&nbsp;&nbsp; - python3.8 이상을 권장합니다.

 - requirements.txt에 있는 패키지들은 Linux 환경기반으로 작성했습니다.

 ```
 pip install -r requirements.txt
 ```
 
 *** Anaconda 환경을 통한 설치 ***

 - 아래 명령어를 프로젝트 폴더에서 입력해주세요.

 ```
 conda env create -f kt_proj.yaml python=3.8 
 ```
 
 - 이후 Anaconda 환경을 활성화 해주세요.
 
 ```
 conda activate kt_proj
 ```
 
 - cuda 1.11.0+cu115 설치
 
  - 기존의 torch 버전을 삭제해주세요.
  
  ```
  pip uninstall torch torchaudio torchvision
  ```
  
  - torch 1.11.0+cu115 
  
  ```
  pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu115 
  ```

### 훈련 및 테스트

1. config 파일을 설정해주세요.

&nbsp;&nbsp;&nbsp;&nbsp; - 경로 : config/kocharelectra_config.json</p>

```json
{
  "ckpt_dir": "ko-char-electra-encoder-decoder", -> 모델이 저장될 가장 상위 폴더의 바로 아래 하위 폴더
  "train_npy": "./data/data_busan/kor/npy/train.npy",
  "dev_npy": "./data/data_busan/kor/npy/dev.npy",
  "test_npy": "./data/data_busan/kor/npy/test.npy",
  "evaluate_test_during_training": true,
  "eval_all_checkpoints": true,
  "save_optimizer": false,
  "do_train": false, -> 훈련을 수행할지
  "do_eval": true, -> 테스트를 수행할지
  "max_seq_len": 256,
  "num_train_epochs": 10,
  "weight_decay": 0.0,
  "gradient_accumulation_steps": 1,
  "adam_epsilon": 1e-8,
  "warmup_proportion": 0,
  "max_grad_norm": 1.0,
  "model_type": "electra-base",
  "model_name_or_path": "monologg/kocharelectra-base-discriminator",
  "output_dir": "./test_model/ckpt-ar-end2end", -> 모델이 저장될 가장 상위 폴더 명
  "seed": 42,
  "train_batch_size": 32,
  "eval_batch_size": 128,
  "logging_steps": 858, -> 모델 훈련시 몇 step 마다 검증 테스트를 수행할지
  "save_steps": 858, -> 모델 훈련시 몇 step 마다 저장을 할지
  "learning_rate": 5e-5,
  "device": "cpu" -> 2023.04.19 추가, 빈 칸이면 자동으로 테스트 환경에서 cuda 확인 [option: cuda | cpu]
}
```

2. 아래 명령어를 실행해주세요.

```
python run_g2p.py
```

### 임의의 문장을 테스트 하고 싶은 경우

```
python ar_test.py --input=안녕하세요 --ckpt_path=test_model/ckpt-ar-end2end/ko-char-electra-encoder-decoder/checkpoint-17150 --config_path=config/kocharelectra_config.json --decoder_vocab_path=data/vocab/decoder_vocab/pron_eumjeol_vocab.json --jaso_dict_path=data/vocab/post_process/jaso_filter.json
```
