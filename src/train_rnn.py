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
from utils import *
from models import Seq2Seq, MLP
from Corpus import LMCorpus, CADCorpus, Dictionary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = run_cad.load_cadgan_args()
logger = init_logger(os.path.join(args.save, "exp_log.txt"))

# Set the random seed manually for reproducibility.
random.seed(args.seed) 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
# load pre-trained models and vocabs
word_ae_params, word_corpus, word_args = load_ckpt(args.word_ckpt)

# create corpus
word_vocab = Dictionary()
word_vocab.load_from_word2idx(word_corpus)

word_word2idx = word_corpus.dictionary.word2idx

corpus = CADCorpus(args.data_path,
                   maxlen=args.maxlen,
                   word_vocab=word_vocab,
                   lowercase=args.lowercase,
                   )

# save arguments
if not os.path.exists(args.save):
    os.makedirs(args.save)

# logger.info("Vocabulary Size: char vocab={}, word vocab={}".format(len(char_word2idx), len(word_word2idx)))
logger.info("Vocabulary Size: word vocab={}".format(len(word_vocab)))

# exp dir
create_exp_dir(os.path.join(args.save), ['train_rnn.py', 'models.py', 'utils.py'],
        dict=(word_word2idx), options=args)

logger.info(str(vars(args)))


###############################################################################
# Build the models
###############################################################################

spacy_vocab = np.load('spacy_matrix.npy')
spacy_vocab = Variable(torch.Tensor(spacy_vocab).float())
if torch.cuda.is_available():
    spacy_vocab.cuda()

word_ae = Seq2Seq(emsize=word_args.emsize,
                nhidden=word_args.nhidden,
                ntokens=word_args.ntokens,
                nlayers=word_args.nlayers,
                noise_r=word_args.noise_r,
                hidden_init=word_args.hidden_init,
                dropout=word_args.dropout,
                # embedding_sign=False)
                embedding_sign=True,
                embedding=spacy_vocab)

word_ae.load_state_dict(word_ae_params)

classifier = MLP(word_args.nhidden)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(list(word_ae.parameters()) + list(classifier.parameters()), lr=3e-4)

one = torch.FloatTensor([1])
mone = one * (-1)
lamda = torch.FloatTensor([10])

if torch.cuda.is_available():
    logger.info("Running on GPU")
    word_ae = word_ae.cuda()
    classifier = classifier.cuda()
    one = one.cuda()
    # D = D.cuda()
    # G = G.cuda()
else:
    logger.info("Running on CPU")




###############################################################################
# Training code
###############################################################################
def train_GAN(batch, train_mode=True):
    if train_mode:
        word_ae.train()
        classifier.train()
    else:
        word_ae.eval()
        classifier.eval()
    # optimizer_D.zero_grad()

    # + samples
    plot1, plot1_lengths = batch['plot1']
    plot2, plot2_lengths = batch['plot2']
    plot3, plot3_lengths = batch['plot3']
    plot4, plot4_lengths = batch['plot4']
    ending1, ending1_lengths = batch['ending1']
    ending2, ending2_lengths = batch['ending2']
    label = torch.FloatTensor(batch['label'])

    if torch.cuda.is_available():
        plot1 = plot1.cuda()
        plot2 = plot2.cuda()
        plot3 = plot3.cuda()
        plot4 = plot4.cuda()
        ending1 = ending1.cuda()
        ending2 = ending2.cuda()
        label = label.cuda()

    # plot1_encoding = word_ae(plot1, plot1_lengths, noise=False, encode_only=True)
    # plot2_encoding = word_ae(plot2, plot2_lengths, noise=False, encode_only=True)
    # plot3_encoding = word_ae(plot3, plot3_lengths, noise=False, encode_only=True)
    # plot4_encoding = word_ae(plot4, plot4_lengths, noise=False, encode_only=True)
    # ending1_encoding = word_ae(ending1, ending1_lengths, noise=False, encode_only=True)
    # ending2_encoding = word_ae(ending2, ending2_lengths, noise=False, encode_only=True)
    plot1_encoding = word_ae(plot1, plot1_lengths, noise=False, encode_only=True).detach()
    plot2_encoding = word_ae(plot2, plot2_lengths, noise=False, encode_only=True).detach()
    plot3_encoding = word_ae(plot3, plot3_lengths, noise=False, encode_only=True).detach()
    plot4_encoding = word_ae(plot4, plot4_lengths, noise=False, encode_only=True).detach()
    ending1_encoding = word_ae(ending1, ending1_lengths, noise=False, encode_only=True).detach()
    ending2_encoding = word_ae(ending2, ending2_lengths, noise=False, encode_only=True).detach()
    inputs = (plot1_encoding, plot2_encoding, plot3_encoding, plot4_encoding, ending1_encoding, ending2_encoding)
    # inputs = (plot3_encoding, plot4_encoding, ending1_encoding, ending2_encoding)
    output = classifier(inputs)
    m = nn.LogSoftmax(dim=1)
    pred = m(output)
    loss = criterion(pred, (label - 1).long())

    if train_mode:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # acc = np.mean(np.argmax(pred.detach().cpu().numpy(), axis=1) == label.cpu().numpy() - 1)
    acc = np.argmax(pred.detach().cpu().numpy(), axis=1)
    return loss.detach().cpu().numpy(), acc, label.cpu().numpy() - 1


def train():
    # gan: preparation
    if args.niters_gan_schedule != "":
        gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
    else:
        gan_schedule = []
    niter_gan = 1

    global global_step
    global_step = 0

    best_valid_acc = None
    eval_batch_size = args.batch_size

    impatience = 0
    for epoch in range(1, args.epochs+1):
        # re-batchify every epoch to shuffle the train and generate fake pairs
        train_data = corpus.batchify(corpus.train, args.batch_size, shuffle=True)

        test_data = corpus.batchify(corpus.test, eval_batch_size, shuffle=False)

        logger.info("Epoch %d" % epoch)
        logger.info("Loaded data!")
        logger.info("Training data! \t: %d examples, %d batches" % (len(corpus.train), len(train_data)))
        logger.info("Test data! \t: %d examples, %d batches" % (len(corpus.test), len(test_data)))

        # update gan training schedule
        if epoch in gan_schedule:
            niter_gan += 1
            logger.info("GAN training loop schedule: {}".format(niter_gan))

        epoch_start_time = time.time()
        start_time = time.time()

        # train
        total_pred, total_truth = [], []
        for i in range(len(train_data)):
            # update global_step here, might be used in TensorboardX later
            global_step += 1
            # train_GAN(train_data[i])

            loss, pred, truth = train_GAN(train_data[i])
            total_pred.extend(list(pred))
            total_truth.extend(list(truth))
        print('train', np.mean(np.array(total_pred) == np.array(total_truth)))

        # val
        print('=======================================================================')
        total_pred, total_truth = [], []
        for i in range(len(test_data)):
            loss, pred, truth = train_GAN(test_data[i], train_mode=False)
            total_pred.extend(list(pred))
            total_truth.extend(list(truth))
        print('val', np.mean(np.array(total_pred) == np.array(total_truth)))
        print()


if __name__ == '__main__':
    train()
