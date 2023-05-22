import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math

INF = 1e10
TINY = 1e-9

#===============================================================
def positional_encodings_like(x, t=None):   # hope to be differentiable
#===============================================================
    if t is None:
        positions = torch.arange(0, x.size(-2)) # .expand(*x.size()[:2])
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1) # 0 2 4 6 ... (256)
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    channels = 1 / (10000 ** Variable(channels))

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings

#===============================================================
def log_softmax(x):
#===============================================================
    if x.dim() == 3:
        return F.log_softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.log_softmax(x)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
#===============================================================
def matmul(x, y):
#===============================================================
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)

# F.softmax has strange default behavior, normalizing over dim 0 for 3D inputs
#===============================================================
def softmax(x):
#===============================================================
    if x.dim() == 3:
        return F.softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.softmax(x)

#===============================================================
def gumbel_softmax(input, beta=0.5, tau=1.0):
#===============================================================
    noise = input.data.new(*input.size()).uniform_()
    noise.add_(TINY).log_().neg_().add_(TINY).log_().neg_()
    return softmax((input + beta * Variable(noise)) / tau)

#===============================================================
class Decoder(nn.Module):
#===============================================================
    def __init__(self, args, dec_vocab, causal=True,
                positional=False, diag=False,
                highway=False, windows=None,
                noisy=False, cosine_output=False):

        super().__init__()

        if windows is None:
            windows = [-1 for _ in range(args.n_layers)]

        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal, diag, highway, windows[i], positional, noisy)
            for i in range(args.n_layers)])

        self.out = nn.Linear(args.d_model, len(dec_vocab))

        self.dropout = nn.Dropout(args.drop_ratio)
        self.d_model = args.d_model
        self.dec_vocab = dec_vocab
        self.length_ratio = args.length_ratio
        self.positional = positional
        self.orderless = args.input_orderless

    def forward(self, x, encoding, mask_src=None, mask_trg=None, input_embeddings=False, feedback=None, positions=None):

        if not input_embeddings:  # compute input embeddings
            if x.ndimension() == 2:
                x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
            elif x.ndimension() == 3:  # softmax relaxiation
                x = x @ self.out.weight * math.sqrt(self.d_model)  # batch x len x embed_size

        if not self.orderless:
            x += positional_encodings_like(x)
        x = self.dropout(x)

        for l, (layer, enc) in enumerate(zip(self.layers, encoding[1:])):
            x = layer(x, enc, mask_src=mask_src, mask_trg=mask_trg, feedback=feedback)
        return x

    def greedy(self, encoding, mask_src=None, mask_trg=None, feedback=None):

        encoding = encoding[1:]
        B, T, C = encoding[0].size()  # batch-size, decoding-length, size
        T *= self.length_ratio

        outs = Variable(encoding[0].data.new(B, T + 1).long().fill_(
                    self.dec_vocab.index('[CLS]')))
        hiddens = [Variable(encoding[0].data.new(B, T, C).zero_())
                    for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])

        eos_yet = encoding[0].data.new(B).byte().zero_()

        attentions = []

        for t in range(T):
            torch.cuda.nvtx.mark(f'greedy:{t}')
            hiddens[0][:, t] = self.dropout(
                hiddens[0][:, t] + F.embedding(outs[:, t], embedW))

            inter_attention = []
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t+1]
                x = self.layers[l].selfattn(hiddens[l][:, t:t+1], x, x)   # we need to make the dimension 3D
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l], mask_src, inter_attention))[:, 0]

            inter_attention = torch.cat(inter_attention, 1)
            attentions.append(inter_attention)

            _, preds = self.out(hiddens[-1][:, t]).max(-1)
            preds[eos_yet] = self.dec_vocab.index('[PAD]')

            eos_yet = eos_yet | (preds.data == self.dec_vocab.index['[SEP]'])
            outs[:, t + 1] = preds
            if eos_yet.all():
                break

        if feedback is not None:
            feedback['source'] = torch.cat(attentions, 2)

        return outs[:, 1:t+2]

    def beam_search(self, encoding, mask_src=None, mask_trg=None, width=2, alpha=0.6):  # width: beamsize, alpha: length-norm
        encoding = encoding[1:]
        W = width
        B, T, C = encoding[0].size()

        # expanding
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(
                B, W, T, C).contiguous().view(B * W, T, C)
        mask_src = mask_src[:, None, :].expand(B, W, T).contiguous().view(B * W, T)

        T *= self.length_ratio
        outs = Variable(encoding[0].data.new(B, W, T + 1).long().fill_(
            self.dec_vocab.index('[CLS]')))

        logps = Variable(encoding[0].data.new(B, W).float().fill_(0))  # scores
        hiddens = [Variable(encoding[0].data.new(B, W, T, C).zero_())  # decoder states: batch x beamsize x len x h
                    for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = encoding[0].data.new(B, W).byte().zero_()  # batch x beamsize, all the sentences are not finished yet.
        eos_mask = eos_yet.float().fill_(-INF)[:, :, None].expand(B, W, W)
        eos_mask[:, :, 0] = 0  # batch x beam x beam

        for t in range(T):
            hiddens[0][:, :, t] = self.dropout(
                hiddens[0][:, :, t] + F.embedding(outs[:, :, t], embedW))
            for l in range(len(self.layers)):
                x = hiddens[l][:, :, :t + 1].contiguous().view(B * W, -1, C)
                x = self.layers[l].selfattn(x[:, -1:, :], x, x)
                hiddens[l + 1][:, :, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding[l], encoding[l], mask_src)).view(
                        B, W, C)

            # topk2_logps: scores, topk2_inds: top word index at each beam, batch x beam x beam
            topk2_logps, topk2_inds = log_softmax(
                self.out(hiddens[-1][:, :, t])).topk(W, dim=-1)

            # mask out the sentences which are finished
            topk2_logps = topk2_logps * Variable(eos_yet[:, :, None].float() * eos_mask + 1 - eos_yet[:, :, None].float())
            topk2_logps = topk2_logps + logps[:, :, None]

            if t == 0:
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)

            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)
            eos_yet = eos_yet.gather(1, topk_beam_inds.data)

            logps = logps * (1 - Variable(eos_yet.float()) * 1 / (t + 2)).pow(alpha)
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            outs[:, :, t + 1] = topk_token_inds
            topk_beam_inds = topk_beam_inds[:, :, None, None].expand_as(
                hiddens[0])
            for i in range(len(hiddens)):
                hiddens[i] = hiddens[i].gather(1, topk_beam_inds)
            eos_yet = eos_yet | (topk_token_inds.data == self.dec_vocab.index('[SEP]'))
            if eos_yet.all():
                return outs[:, 0, 1:]
        return outs[:, 0, 1:]

#===============================================================
class DecoderLayer(nn.Module):
#===============================================================
    def __init__(self, args, causal=True, diag=False, highway=False,
                window=-1, positional=False, noisy=False):
        super().__init__()
        self.positional = positional
        self.selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal, diag, window,
                    use_wo=args.use_wo),
            args.d_model, args.drop_ratio)

        self.attention = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, noisy=noisy, use_wo=args.use_wo),  # only noisy when doing cross-attention
            args.d_model, args.drop_ratio)

        if positional:
            self.pos_selfattn = ResidualBlock(
            MultiHead2(args.d_model, args.d_model, args.n_heads,   # first try 1 positional head
                    args.drop_ratio, causal, diag, window,
                    use_wo=args.use_wo),
            args.d_model, args.drop_ratio, pos=2)

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.drop_ratio)

    def forward(self, x, encoding, p=None, mask_src=None, mask_trg=None, feedback=None):
        feedback_src = []
        feedback_trg = []
        x = self.selfattn(x, x, x, mask_trg, feedback_trg)   #

        if self.positional:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_selfattn(pos_encoding, pos_encoding, x, mask_trg, None, weights)  # positional attention
        x = self.feedforward(self.attention(x, encoding, encoding, mask_src, feedback_src))

        if feedback is not None:

            if 'source' not in feedback:
                feedback['source'] = feedback_src
            else:
                feedback['source'] += feedback_src

            if 'target' not in feedback:
                feedback['target'] = feedback_trg
            else:
                feedback['target'] += feedback_trg
        return x

#===============================================================
class ResidualBlock(nn.Module):
#===============================================================
    def __init__(self, layer, d_model, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))

#===============================================================
class MultiHead2(nn.Module):
#===============================================================
    def __init__(self, d_key, d_value, n_heads, drop_ratio,
                causal=False, diag=False, window=-1, noisy=False, use_wo=True):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag, window=window, noisy=noisy)
        self.wq = Linear(d_key, d_key, bias=use_wo)
        self.wk = Linear(d_key, d_key, bias=use_wo)
        self.wv = Linear(d_value, d_value, bias=use_wo)
        if use_wo:
            self.wo = Linear(d_value, d_key, bias=use_wo)
        self.use_wo = use_wo
        self.n_heads = n_heads

    def forward(self, query, key, value, mask=None, feedback=None, weights=None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)   # B x T x D
        B, Tq, D = query.size()
        _, Tk, _ = key.size()
        N = self.n_heads
        probs = []

        query, key, value = (x.contiguous().view(B, -1, N, D//N).transpose(2, 1).contiguous().view(B*N, -1, D//N)
                                for x in (query, key, value))
        if mask is not None:
            mask = mask[:, None, :].expand(B, N, Tk).contiguous().view(B*N, -1)
        outputs = self.attention(query, key, value, mask, probs, beta, tau, weights)  # (B x n) x T x (D/n)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)

        if feedback is not None:
            feedback.append(probs[0].view(B, N, Tq, Tk))

        if self.use_wo:
            return self.wo(outputs)
        return outputs

#===============================================================
class FeedForward(nn.Module):
#===============================================================
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

#===============================================================
class LayerNorm(nn.Module):
#===============================================================
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

#===============================================================
class Attention(nn.Module):
#===============================================================
    def __init__(self, d_key, drop_ratio, causal, diag=False, window=-1, noisy=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.diag = diag
        self.window = window
        self.noisy = noisy

    def forward(self, query, key, value=None, mask=None,
                feedback=None, beta=0, tau=1, weights=None):
        dot_products = matmul(query, key.transpose(1, 2))   # batch x trg_len x trg_len

        if weights is not None:
            dot_products = dot_products + weights   # additive bias

        if query.dim() == 3 and self.causal and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0))

        if self.window > 0:
            window_mask = key.data.new(key.size(1), key.size(1)).fill_(1)
            window_mask = (window_mask.triu(self.window+1) + window_mask.tril(-self.window-1)) * INF
            dot_products.data.sub_(window_mask.unsqueeze(0))

        if self.diag:
            inds = torch.arange(0, key.size(1)).long().view(1, 1, -1)
            if key.is_cuda:
                inds = inds.cuda(key.get_device())
            dot_products.data.scatter_(1, inds.expand(dot_products.size(0), 1, inds.size(-1)), -INF)
            # eye = key.data.new(key.size(1), key.size(1)).fill_(1).eye() * INF
            # dot_products.data.sub_(eye.unsqueeze(0))

        if mask is not None:
            # print(dot_products.data.size(), mask[:, None, :].size())
            if dot_products.dim() == 2:
                dot_products.data -= ((1 - mask) * INF)
            else:
                dot_products.data -= ((1 - mask[:, None, :]) * INF)

        if value is None:
            return dot_products

        logits = dot_products / self.scale
        if (not self.noisy): # or (not self.training):
            probs = softmax(logits)
        else:
            probs = gumbel_softmax(logits, beta=beta, tau=tau)

        if feedback is not None:
            feedback.append(probs.contiguous())

        return matmul(self.dropout(probs), value)

#===============================================================
class Linear(nn.Linear):
#===============================================================
    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)