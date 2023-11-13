import pickle
import argparse
import pandas as pd
import re
import torch

from model.transformers.utils import Params
from model.transformers.model.transformer import Transformer

class Convert_Eng:
    def __init__(self):
        self.params = Params('model/transformers/config/params.json')
        pickle_kor = open('model/transformers/pickles/kor.pickle', 'rb')
        self.kor = pickle.load(pickle_kor)
        self.kor_idx = self.kor.vocab.stoi['<eos>']

        pickle_eng = open('model/transformers/pickles/eng.pickle', 'rb')
        self.eng = pickle.load(pickle_eng)

        # select model and load trained model
        self.model = Transformer(self.params)
        self.model.load_state_dict(torch.load(self.params.save_model))
        self.model.to(self.params.device)
        self.model.eval()
    def predict(self, input_text):
        input = input_text if input_text.isupper() else input_text.lower()
        input = re.sub(r'[^a-zA-Z0-9 ]', '', input)

        # convert input into tensor and forward it through selected model
        tokenized = list(input)
        try:

            indexed = [self.eng.vocab.stoi[token] for token in tokenized]
            max_len = 64  # 최대 길이 설정
            if len(indexed) < max_len:
                # 입력이 `max_len`보다 짧은 경우 패딩 토큰을 추가합니다.
                indexed += [self.eng.vocab.stoi['<pad>']] * (max_len - len(indexed))
            else:
                # 입력이 `max_len`보다 긴 경우 잘라냅니다.
                indexed = indexed[:max_len]
            source = torch.LongTensor(indexed).unsqueeze(0).to(
                self.params.device)  # [1, source_len]: unsqueeze to add batch size
            target = torch.zeros(1, self.params.max_len).type_as(source.data)  # [1, max_len]
            encoder_output = self.model.encoder(source)
            next_symbol = self.kor.vocab.stoi['<sos>']

            for i in range(0, self.params.max_len):
                target[0][i] = next_symbol
                decoder_output, _ = self.model.decoder(target, source, encoder_output)  # [1, target length, output dim]
                prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
                next_word = prob.data[i]
                next_symbol = next_word.item()
            eos_idx = int(torch.where(target[0] == self.kor_idx)[0][0])
            target = target[0][:eos_idx].unsqueeze(0)

            # translation_tensor = [target length] filed with word indices
            target, attention_map = self.model(source, target)
            target = target.squeeze(0).max(dim=-1)[1]

            translated_token = [self.kor.vocab.itos[token] for token in target]
            translation = translated_token[:translated_token.index('<eos>')]
            translation = ''.join(translation)
            return [translation]
        except:
            return []

        # print(f'eng> {config.input}')
        # print(f'kor> {translation}')
        # display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])


if __name__ == '__main__':
    # data = pd.read_excel('kt_dic_match_word.xlsx', engine='openpyxl')
    convert = Convert_Eng()
    # result = []
    # for i in range(len(data)):
    #     inputs = data['kt_word'][i]
    #     try:
    #         pred = convert.predict(inputs)[0]
    #     except:
    #         pred = "error"
    #     print(inputs, pred)
    #     result.append(pred)
    # data['transformers']=result
    # data.to_excel('kt_dic_match_word_transformers.xlsx', index=False)
    print(convert.predict('m'))
