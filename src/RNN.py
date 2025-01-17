import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import to_gpu
import json
import os
import numpy as np
import sys


def build_MLP(model, layer_name, layer_sizes, activation):
    setattr(model, layer_name, [])  # model.layer_name = []
    layers = getattr(model, layer_name)

    for i in range(len(layer_sizes) - 1):
        layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        layers.append(layer)
        model.add_module(layer_name + "-linear-" + str(i + 1), layer)

        # add batch normalization after first layer (wgan-gp disables use of BN layer, but stablize our training greatly)
        if i != 0:
            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            layers.append(bn)
            model.add_module(layer_name + "-bn-" + str(i + 1), bn)

        layers.append(activation)
        model.add_module(layer_name + "-activ-" + str(i + 1), activation)


class MLP_D(nn.Module):
    def __init__(self, input_dim, output_dim, arch_layers, activation=nn.LeakyReLU(0.2)):
        super(MLP_D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # shortform/longform/context layers
        # layer_sizes = [input_dim] + [int(x) for x in arch_layers.split('-')]
        layer_sizes_sf_lf = [128] + [256, 256]
        layer_sizes_context = [input_dim] + [512, 512]
        build_MLP(self, "shortform_layer", layer_sizes_sf_lf, activation)
        build_MLP(self, "longform_layer", layer_sizes_sf_lf, activation)
        build_MLP(self, "context_layer", layer_sizes_context, activation)

        # main network
        layer_sizes = [256 + 256 + 512] + [512, 256]
        build_MLP(self, "merge_layer", layer_sizes, activation)

        # a linear output layer
        layer = nn.Linear(layer_sizes[-1], output_dim)
        self.merge_layer.append(layer)
        self.add_module("output_layer", layer)

        self.init_weights()

    def forward(self, sf, lf, cont):
        for i, layer in enumerate(self.shortform_layer):
            sf = layer(sf)
        for i, layer in enumerate(self.longform_layer):
            lf = layer(lf)
        for i, layer in enumerate(self.context_layer):
            cont = layer(cont)

        x = torch.cat([sf, cont, lf], dim=1)
        for i, layer in enumerate(self.merge_layer):
            x = layer(x)

        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.shortform_layer + self.longform_layer + self.context_layer + self.merge_layer:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(0, init_std)
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)


class MLP_G(nn.Module):
    def __init__(self, input_dim, output_dim, noise_dim, arch_layers, activation=nn.LeakyReLU(0.2)):
        super(MLP_G, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # layer_sizes = [input_dim] + [int(x) for x in arch_layers.split('-')]
        layer_sizes_sf = [128] + [256, 256]
        layer_sizes_context = [input_dim] + [512, 512]
        build_MLP(self, "shortform_layer", layer_sizes_sf, activation)
        build_MLP(self, "context_layer", layer_sizes_context, activation)
        build_MLP(self, "noise_layer", [noise_dim, 128], activation)

        layer_sizes = [256 + 512 + 128] + [768, 512]
        build_MLP(self, "merge_layer", layer_sizes, activation)

        layer = nn.Linear(layer_sizes[-1], 128)
        self.merge_layer.append(layer)
        self.add_module("output_layer", layer)

        self.init_weights()

    def forward(self, noise_z, sf, cont):
        for i, layer in enumerate(self.shortform_layer):
            sf = layer(sf)
        for i, layer in enumerate(self.context_layer):
            cont = layer(cont)
        for i, layer in enumerate(self.noise_layer):
            noise_z = layer(noise_z)

        x = torch.cat([sf, cont, noise_z], dim=1)
        for i, layer in enumerate(self.merge_layer):
            x = layer(x)

        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.shortform_layer + self.context_layer + self.merge_layer:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(0, init_std)
            if hasattr(layer, 'bias'):
                layer.bias.data.fill_(0)


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_r=0.2,
                 hidden_init=False, dropout=0, gpu=True, embedding_sign=False, embedding=None):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_r = noise_r
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        if embedding_sign:
            self.embedding = embedding
            self.embedding_sign = True
        else:
            self.embedding = nn.Embedding(ntokens, emsize)
            self.embedding_sign = False
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)
        # self.encoder = nn.GRU(input_size=emsize,
        #                        hidden_size=nhidden,
        #                        num_layers=nlayers,
        #                        dropout=dropout,
        #                        batch_first=True)

        decoder_input_size = emsize + nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        # self.decoder = nn.GRU(input_size=decoder_input_size,
        #                        hidden_size=nhidden,
        #                        num_layers=1,
        #                        dropout=dropout,
        #                        batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        if not self.embedding_sign:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(1, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(1, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(1, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        length = torch.tensor(lengths)
        if torch.cuda.is_available():
            length = length.cuda()
        order = torch.argsort(length, descending=True)
        ordered_indices = torch.index_select(indices, dim=0, index=order)
        ordered_lengths = torch.index_select(length, dim=0, index=order)
        ordered_hidden = self.encode(ordered_indices, ordered_lengths, noise)

        argsorted_order = torch.argsort(order, descending=False)
        hidden = ordered_hidden.index_select(dim=0, index=argsorted_order)

        if encode_only:
            return hidden

        if hidden.requires_grad:
            hidden.register_hook(self.store_grad_norm)

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        if self.embedding_sign:
            embeddings = self.embedding[indices]
        else:
            embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.encoder(packed_embeddings)
        # state = ( h_t, c_t ), packed_output is the hidden state of every word packed_output[0] = [batch_size, maxlen, hidden_size]
        # And the hidden state of every last word of every sentence should be idnetical to the corresponding state[0], i.e h_t
        hidden = state[0][-1]
        hidden = hidden / torch.norm(hidden, p=2, dim=1, keepdim=True)  # 2-Norm of each vector

        if noise and self.noise_r > 0:
            gauss_noise = torch.normal(mean=torch.zeros(hidden.size()), std=self.noise_r)
            if torch.cuda.is_available():
                gauss_noise = gauss_noise.cuda()
            hidden = hidden + Variable(gauss_noise)

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen,
                                                1)  # [batch_size, hidden_size] -> [batch_size, maxlen, hidden_size]

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        # augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))
            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                probs = F.softmax(overvocab / temp, dim=-1)
                indices = torch.multinomial(probs, 1)
            indices = indices.unsqueeze(1)
            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)
        return max_indices

    def noise_anneal(self, fac):
        self.noise_r *= fac


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.l1_plot1 = nn.Linear(hidden_size, 100)
        self.l1_plot2 = nn.Linear(hidden_size, 100)
        self.l1_plot3 = nn.Linear(hidden_size, 100)
        self.l1_plot4 = nn.Linear(hidden_size, 100)
        self.l1_ending1 = nn.Linear(hidden_size, 100)
        self.l1_ending2 = nn.Linear(hidden_size, 100)
        # self.l2 = nn.Linear(60, 100)
        self.l2 = nn.Linear(600, 512)
        self.l3 = nn.Linear(512, 2)
        # self.l4=nn.Linear(100,2)
        self.dropout = nn.Dropout(0.6)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x, train_mode=True):
        l1_plot1_ = F.relu(self.l1_plot1(x[0]))
        l1_plot2_ = F.relu(self.l1_plot2(x[1]))
        l1_plot3_ = F.relu(self.l1_plot3(x[2]))
        l1_plot4_ = F.relu(self.l1_plot4(x[3]))
        l1_ending1_ = F.relu(self.l1_ending1(x[4]))
        l1_ending2_ = F.relu(self.l1_ending2(x[5]))
        # y = l1_plot1_ + l1_plot2_ + l1_plot3_ + l1_plot4_ + l1_ending1_ + l1_ending2_
        y = torch.cat((l1_plot1_, l1_plot2_, l1_plot3_, l1_plot4_, l1_ending1_, l1_ending2_), axis=1)
        # y = torch.cat((l1_plot3_, l1_plot4_, l1_ending1_, l1_ending2_), axis=1)
        y = self.dropout(y)
        y = F.relu(self.l2(y))
        # y = self.dropout(y)
        # y = F.relu(self.l3(y))
        # y = F.sigmoid(self.l3(y))
        # y = self.dropout(y)
        if train_mode:
            y = self.l3(y)
        else:
            y = self.output(self.l3(y))
        return y


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences
