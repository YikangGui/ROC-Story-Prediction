import argparse
import os
import time
import math
import numpy as np
import random
import sys
import shutil
import json
import string

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import run_cad
from utils import to_gpu, train_ngram_lm, get_ppl, create_exp_dir, save_ckpt, load_ckpt
from Corpus import LMCorpus, Dictionary
from models import Seq2Seq

args = run_cad.load_lm_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
# create corpus
# corpus = LMCorpusold(paths=args.data_path,
#                   maxlen=args.maxlen,
#                   fields=args.fields,
#                   token_level=args.token_level,
#                   vocab_size=args.vocab_size,
#                   lowercase=args.lowercase,
#                   cut_by_cnt=False)

corpus = LMCorpus(paths=args.data_path,
                  maxlen=args.maxlen,
                  fields=args.fields,
                  token_level=args.token_level,
                  vocab_size=args.vocab_size,
                  lowercase=args.lowercase,
                  cut_by_cnt=True)

# corpus = LMCorpus(paths=args.data_path,
#                   maxlen=args.maxlen,
#                   fields=args.fields,
#                   token_level=args.token_level,
#                   vocab_size=args.vocab_size,
#                   lowercase=args.lowercase,
#                   cut_by_cnt=False, create_dict=False,
#                   if_tokenize=False, if_vetorize=False)


# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens

# exp dir
create_exp_dir(os.path.join(args.save), ['train_ae.py', 'models.py', 'utils.py'],
               dict=corpus.dictionary.word2idx, options=args)


def logging(str, to_stdout=True):
    with open(os.path.join(args.save, 'log.txt'), 'a') as f:
        f.write(str + '\n')
    if to_stdout:
        print(str)


logging(str(vars(args)))

eval_batch_size = 32

train_batches_num = math.floor(corpus.train_num / args.batch_size)
test_batches_num = math.floor(corpus.test_num / eval_batch_size)

print("Loaded data!")
print("Training data! \t: %d examples, %d batches" % (corpus.train_num, train_batches_num))
print("Test data! \t: %d examples, %d batches" % (corpus.test_num, test_batches_num))

###############################################################################
# Build the models
###############################################################################
autoencoder = Seq2Seq(emsize=args.emsize,  # default = 300
                      nhidden=args.nhidden,  # default = 300
                      ntokens=args.ntokens,  # default = 30002
                      nlayers=args.nlayers,  # default = 1
                      noise_r=args.noise_r,  # default = 0.05
                      hidden_init=args.hidden_init,  # default = False
                      dropout=args.dropout,
                      # embedding_sign=True,
                      # embedding=Variable(torch.Tensor(corpus.spacy_vocab).float()).cuda())
                      embedding_sign=False)
# autoencoder = nn.DataParallel(autoencoder)

print(autoencoder)

# optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=args.lr_ae)
print(optimizer_ae)
if torch.cuda.is_available():
    autoencoder = autoencoder.cuda()
    one = torch.Tensor(1).fill_(1).cuda()
else:
    one = torch.Tensor(1).fill_(1)

# global vars
mone = one * -1

print(args.char_ckpt)
print(args.word_ckpt)

# if args.char_ckpt != '0':
#     char_ae_params, char_corpus, char_args = load_ckpt(args.char_ckpt)
#     char_vocab = Dictionary()
#     char_vocab.load_from_word2idx(char_corpus)
#     char_word2idx = char_corpus.dictionary.word2idx
#     autoencoder.load_state_dict(char_ae_params, strict=False)
#     corpus = char_corpus
#
# if args.word_ckpt != '0':
#     word_ae_params, word_corpus, word_args = load_ckpt(args.word_ckpt)
#     word_vocab = Dictionary()
#     word_vocab.load_from_word2idx(word_corpus)
#     word_word2idx = word_corpus.dictionary.word2idx
#     autoencoder.load_state_dict(word_ae_params, strict=False)
#     corpus = word_corpus


###############################################################################
# Training code
###############################################################################

def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    # print(13)
    autoencoder.eval()
    total_loss = 0.0
    # print(12)
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    # print(11)
    aeout_path = os.path.join(args.save, "autoencoder_epoch%d.txt" % epoch)
    output_file = open(aeout_path, "w+")
    # print(11)
    for i, batch in enumerate(data_source):
        # print("validate batch %d" % i)
        source, target, lengths = batch
        # print(10)
        if torch.cuda.is_available():
            source = Variable(source).cuda()
            target = Variable(target).cuda()
        else:
            source = Variable(source)
            target = Variable(target)
        # print(9)
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
        # print(8)
        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)
        # print(7)
        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += F.cross_entropy(masked_output, masked_target).data
        # print(6)
        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).data.item()
        bcnt += 1
        # print(5)
        # write to file
        max_values, max_indices = torch.max(output, 2)
        max_indices = \
            max_indices.view(output.size(0), -1).data.cpu().numpy()
        target = target.view(output.size(0), -1).data.cpu().numpy()
        # print(4)
        for t, idx in zip(target, max_indices):
            # real sentence
            chars = " ".join([corpus.dictionary.idx2word[x] for x in t
                              if corpus.dictionary.idx2word[x] != '<pad>'])
            output_file.write(chars + '\n')
            # autoencoder output sentence
            chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
            output_file.write(chars + '\n' * 2)
    # print(3)
    output_file.close()
    # print(1)
    return total_loss / test_batches_num, all_accuracies / bcnt


def train_ae(epoch, batch, total_loss_ae, start_time, i):
    autoencoder.train()
    optimizer_ae.zero_grad()

    source, target, lengths = batch

    if torch.cuda.is_available():
        source = Variable(source).cuda()  # 有开头标志
        target = Variable(target).cuda()  # 有结尾标志
    else:
        source = Variable(source)
        target = Variable(target)

    # print(source.shape)

    output = autoencoder(source, lengths, noise=True)  # output.size = batch_size* maxlen* vocab_size]

    mask = target.gt(0)  # target 中大于0的位置为True，小于等于的位置为False，组成1d tensor
    masked_target = target.masked_select(mask)  # 保留位置为True的值，组成1d tensor
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)  # 行为batch的长度，列为ntokens的长度
    flat_output = output.view(-1, ntokens)  # 变成二维矩阵，列为ntokens
    masked_output = flat_output.masked_select(output_mask).view(-1, ntokens)
    loss = F.cross_entropy(masked_output, masked_target)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data.item()
    if i % args.log_interval == 0:
        probs = F.softmax(masked_output, dim=-1)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data.item()
        cur_loss = total_loss_ae / args.log_interval
        elapsed = time.time() - start_time
        logging('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'.format(
            epoch, i, train_batches_num,
            elapsed * 1000 / args.log_interval,
            cur_loss, math.exp(cur_loss), accuracy))
        total_loss_ae = 0
        start_time = time.time()
    return total_loss_ae, start_time


def train():
    logging("Training text AE")

    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1

    best_test_loss = None
    impatience = 0
    print("Begin!\n")
    for epoch in range(1, args.epochs + 1):
        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logging("GAN training loop schedule: {}".format(niter_gan))

        total_loss_ae = 0
        epoch_start_time = time.time()
        start_time = time.time()

        # train ae
        train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)
        for niter, data in enumerate(train_data):
            # if niter == 1:
            #     print(data)
            total_loss_ae, start_time = train_ae(epoch, data,
                                                 total_loss_ae, start_time, niter)

            if niter % 10 == 0:
                autoencoder.noise_anneal(args.noise_anneal)
                logging('[{}/{}][{}/{}]'.format(
                    epoch, args.epochs, niter, train_batches_num))
        # eval
        # corpus是全局的，如果用过一次，生成器就遍历完了，下次在用就不会生成数据，所以test_data这里必须设成函数内局部变量
        test_data = corpus.batchify(corpus.test, eval_batch_size, shuffle=False)
        test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
        # print(2)
        logging('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                'test ppl {:5.2f} | acc {:3.3f}'.format(epoch,
                                                        (time.time() - epoch_start_time), test_loss,
                                                        math.exp(test_loss), accuracy))

        save_ckpt("ckpt_epoch%d" % epoch, args.save, autoencoder, args, corpus)
        if best_test_loss is None or test_loss < best_test_loss:
            impatience = 0
            best_test_loss = test_loss
            logging("New saving model: epoch {}. best valid score={:.6f}".format(epoch, best_test_loss))
            save_ckpt("ckpt_epoch%d-best@%.6f" % (epoch, best_test_loss),
                      args.save, autoencoder, args, corpus)
        else:
            if not args.no_earlystopping and epoch >= args.min_epochs:
                impatience += 1
                if impatience > args.patience:
                    logging("Ending training")
                    sys.exit()


if __name__ == '__main__':
    train()
