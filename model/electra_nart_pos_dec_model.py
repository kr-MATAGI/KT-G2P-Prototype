import torch
import torch.nn as nn

from model.nonauto_nmt.pos_nart_decoder import Decoder, softmax
from transformers import ElectraModel

#========================================================
class ElectraNartPosDecModel(nn.Module):
#========================================================
    def __init__(
            self,
            args, tokenizer, decoder, electra_model,
            src_vocab, dec_vocab
    ):
        super().__init__()
        self.args = args

        self.tokenizer = tokenizer
        self.decoder = decoder
        self.electra = electra_model

        self.src_vocab = src_vocab
        self.dec_vocab = dec_vocab

        self.device = args.device

    @classmethod
    def build_decoder(cls, args, dec_vocab):
        base_decoder_architecture(args)
        return Decoder(args=args, dec_vocab=dec_vocab, positional=True)

    @classmethod
    def build_model(cls, args, tokenizer, src_vocab, dec_vocab):
        decoder = cls.build_decoder(args, dec_vocab)
        electra = ElectraModel.from_pretrained(args.model_name_or_path, output_hidden_states=True)
        return ElectraNartPosDecModel(args, tokenizer, decoder, electra, src_vocab, dec_vocab)

    def prepare_masks(self, inputs):
        if inputs.ndimension() == 2:
            masks = (inputs.data != self.dec_vocab.index('[PAD]')).float()
        else:
            masks = (inputs.data[:, :, self.dec_vocab.index('[PAD]')] != 1).float()

        return masks

    def forward(
            self,
            src_tokens, prev_output_tokens, **kwargs
    ):
        src_masks = self.prepare_masks(src_tokens)
        electra_out = self.electra(input_ids=src_tokens, attention_mask=src_masks)
        electra_out = electra_out.hidden_states

        '''
            x = decoder_inputs,
            encoding = encodings
        '''
        decoder_out = self.decoder(x=src_tokens, encoding=electra_out, mask_src=src_masks)
        decoder_out = self.decoder.out(decoder_out)

        return decoder_out


def base_decoder_architecture(args):
    # maybe decoder
    args.d_model = 768
    args.d_hidden = 768
    args.n_layers = 6
    args.n_heads = 8
    args.drop_ratio = 0.1
    args.warmp = 16000
    args.input_orderless = True

    # decode
    args.length_ratio = 2
    args.decode_mode = 'argmax'
    args.beam_size = 1
    args.f_size = 1
    args.alpha = 1
    args.temperature = 1

    # need to know
    args.use_wo = True

