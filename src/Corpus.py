import os
import pickle
import string
import sys
import spacy
import torch
import numpy as np
import random
import shutil
import json
import math
import logging


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.word2idx['<target>'] = 4
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]

        # for w_id, (w, freq) in enumerate(sorted(vocab_list, key=lambda x:x[1], reverse=True)):
        #     print("\t[%d] %s \t: %d" % (w_id, w, freq))

        if cnt:
            # prune by count
            self.pruned_vocab = \
                {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
            # sort to make vocabulary deterministic
            self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("Original vocab {}; Pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def load_from_idx2word(self, idx2word):
        self.idx2word = idx2word
        self.word2idx = {v: k for k, v in self.idx2word.items()}

    def load_from_word2idx(self, word2idx):
        self.word2idx = word2idx.dictionary.word2idx
        # self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx2word = word2idx.dictionary.idx2word


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


class LMCorpus(object):
    def __init__(self, paths, maxlen, fields,
                 vocab_size=11000, min_freq=5,
                 cut_by_cnt=False, lowercase=False, token_level='word'):
        self.dictionary = Dictionary()
        self.spacy_vocab = []
        self.maxlen = 100
        self.lowercase = lowercase  # default = False
        self.cut_by_cnt = cut_by_cnt  # default = False
        self.min_freq = min_freq  # default = 5
        self.vocab_size = vocab_size  # default = 11000
        self.nlp = spacy.load('en_core_web_md')

        self.train = []
        self.test = []
        for path in paths:
            self.train_path = os.path.join(path, 'train_pretrain.txt')
            self.test_path = os.path.join(path, 'test_pretrain.txt')
            if os.path.exists(self.train_path):
                self.train.extend(self.tokenize(self.train_path, fields, token_level))
            if os.path.exists(self.test_path):
                self.test.extend(self.tokenize(self.test_path, fields, token_level))

        # make the vocabulary from training set
        if token_level == 'char':
            init_char = True
        else:
            init_char = False
        self.make_vocab(self.train, init_char=init_char)

        self.make_spacy_vocab()

        self.train = self.vectorize(self.train)
        self.test = self.vectorize(self.test)

        print('Data size: train %d, test %d' % (len(self.train), len(self.test)))
        self.train_num = len(self.train)
        self.test_num = len(self.test)

    def make_spacy_vocab(self):
        for key in self.dictionary.idx2word.keys():
            value = self.dictionary.idx2word[key]
            self.spacy_vocab.append(self.nlp(value).vector)
        self.spacy_vocab = np.asarray(self.spacy_vocab)
        np.save('spacy_matrix.npy', self.spacy_vocab)

    def make_vocab(self, sent_list, init_char=False):
        if init_char:
            max_idx = int(np.amax(list(self.dictionary.word2idx.values())))
            for char in string.printable:
                max_idx += 1
                self.dictionary.word2idx[char] = max_idx

        for sent in sent_list:
            for word in sent:
                self.dictionary.add_word(word)

        # prune the vocabulary
        if self.cut_by_cnt:
            self.dictionary.prune_vocab(k=self.min_freq, cnt=self.cut_by_cnt)
        else:
            self.dictionary.prune_vocab(k=self.vocab_size, cnt=self.cut_by_cnt)

    def tokenize(self, path, fields, token_level):
        """
        Tokenizes a text file.
        Each line is a json, values in multiple fields might be used. Split them to individual examples.
        """
        dropped = 0
        with open(path, 'r') as f:
            line_count = 0
            examples = []
            # for line in f:
            for line in f:
                line_count += 1
                json_ex = json.loads(line)
                # ignore null inputs
                text_exs = [json_ex[f] for f in fields
                            if f in json_ex and len(json_ex[f].strip()) > 0]

                for text_ex in text_exs:
                    # if self.lowercase:
                    #     text_ex = text_ex.lower().strip()
                    # else:
                    #     text_ex = text_ex.strip()
                    #
                    # if token_level == 'word':
                    #     tokens = text_ex.split(" ")
                    # else:
                    #     tokens = list(text_ex)
                    #
                    # if self.maxlen > 0 and len(tokens) > self.maxlen:
                    #     dropped += 1
                    #     continue
                    tokens = [token.text for token in self.nlp.tokenizer(text_ex)]
                    tokens = ['<sos>'] + tokens + ['<eos>']
                    examples.append(tokens)

            print("Number of data generated from {}: {} sentences out of {} examples. {} are dropped away.".
                  format(path, len(examples), line_count, dropped))
            return examples

    def vectorize(self, sent_list):
        # vectorize
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']

        return_list = []
        for tokens in sent_list:
            indices = [vocab[w] if w in vocab else unk_idx for w in tokens]
            return_list.append(indices)

        return return_list

    def batchify(self, data, bsz, shuffle=False):
        if shuffle:
            random.shuffle(data)
        nbatch = len(data) // bsz
        batches = []

        for i in range(nbatch):
            # Pad batches to maximum sequence length in batch
            batch = data[i * bsz:(i + 1) * bsz]
            # subtract 1 from lengths b/c includes BOTH starts & end symbols
            lengths = [len(x) - 1 for x in batch]
            # sort items by length (decreasing)
            batch, lengths = length_sort(batch, lengths)

            # source has no end symbol
            source = [x[:-1] for x in batch]
            # target has no start symbol
            target = [x[1:] for x in batch]

            # find length to pad to
            maxlen = max(lengths)
            for x, y in zip(source, target):
                zeros = (maxlen - len(x)) * [0]
                x += zeros
                y += zeros

            source = torch.LongTensor(np.array(source))
            target = torch.LongTensor(np.array(target)).view(-1)

            batches.append((source, target, lengths))

        return batches


class CADCorpus(object):
    def __init__(self, paths, maxlen, word_vocab, lowercase=False):
        self.word_dict = word_vocab
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.nlp = spacy.load('en_core_web_lg')

        self.train = []
        self.test = []

        for path in paths:
            self.train_path = os.path.join(path, 'train.txt')
            self.test_path = os.path.join(path, 'test.txt')
            # self.train_path = os.path.join(path, 'test.txt')
            # self.test_path = os.path.join(path, 'train.txt')
            if os.path.exists(self.train_path):
                train_cache_path = os.path.join(path, 'train123.vec')

                if not os.path.exists(train_cache_path):
                    print("Loading train")
                    plot1 = self.tokenize(self.train_path, fields=['plot1'])
                    plot2 = self.tokenize(self.train_path, fields=['plot2'])
                    plot3 = self.tokenize(self.train_path, fields=['plot3'])
                    plot4 = self.tokenize(self.train_path, fields=['plot4'])
                    ending1 = self.tokenize(self.train_path, fields=['ending1'])
                    ending2 = self.tokenize(self.train_path, fields=['ending2'])
                    label = self.tokenize(self.train_path, fields=['label'])
                    print("Vectorizing train")
                    tmp_train = zip(plot1, plot2, plot3, plot4, ending1, ending2, label)
                    vec_train = self.vectorize(tmp_train, word_vocab)
                    print("Dumping train")
                    pickle.dump(vec_train, open(train_cache_path, 'wb'))
                else:
                    vec_train = pickle.load(open(train_cache_path, 'rb'))

                self.train.extend(vec_train)

            if os.path.exists(self.test_path):
                test_cache_path = os.path.join(path, 'test123.vec')

                if not os.path.exists(test_cache_path):
                    plot1 = self.tokenize(self.test_path, fields=['plot1'])
                    plot2 = self.tokenize(self.test_path, fields=['plot2'])
                    plot3 = self.tokenize(self.test_path, fields=['plot3'])
                    plot4 = self.tokenize(self.test_path, fields=['plot4'])
                    ending1 = self.tokenize(self.test_path, fields=['ending1'])
                    ending2 = self.tokenize(self.test_path, fields=['ending2'])
                    label = self.tokenize(self.test_path, fields=['label'])
                    tmp_test = zip(plot1, plot2, plot3, plot4, ending1, ending2, label)
                    vec_test = self.vectorize(tmp_test, word_vocab)
                    pickle.dump(vec_test, open(test_cache_path, 'wb'))
                else:
                    vec_test = pickle.load(open(test_cache_path, 'rb'))

                self.test.extend(vec_test)

            print('#(%s examples): train %d, test %d' % (path, len(vec_train), len(vec_test)))

        print('#(total examples): train %d, test %d' % (len(self.train), len(self.test)))

    def tokenize(self, path, fields):
        """
        Tokenizes a text file.
        Each line is a json, values in multiple fields might be used. Split them to individual examples.
        """
        dropped = 0
        with open(path, 'r') as f:
            line_count = 0
            examples = []
            for line in f:
                line_count += 1
                json_ex = json.loads(line)
                # ignore null inputs
                text_exs = [json_ex[f] for f in fields
                            if f in json_ex and len(json_ex[f].strip())>0]

                if 'label' in fields:
                    examples.append(text_exs)

                else:
                    for text_ex in text_exs:
                        # if self.lowercase:
                        #     text_ex = text_ex.lower().strip()
                        # else:
                        #     text_ex = text_ex.strip()
                        #
                        # if token_level == 'word':
                        #     tokens = text_ex.split(" ")
                        # else:
                        #     tokens = list(text_ex)
                        #
                        # if self.maxlen > 0 and len(tokens) > self.maxlen:
                        #     dropped += 1
                        #     continue

                        tokens = [token.text for token in self.nlp.tokenizer(text_ex)]
                        tokens = ['<sos>'] + tokens + ['<eos>']
                        examples.append(tokens)

        return examples

    def vectorize(self, examples, word_vocab):
        """
        :param examples: a list of triples (short_form, long_form, context)
        :param word_vocab:
        :return:
        """
        return_examples = []
        for triple in examples:
            plot1, plot2, plot3, plot4, ending1, ending2, label = triple
            plot1_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in plot1]
            plot2_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in plot2]
            plot3_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in plot3]
            plot4_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in plot4]
            ending1_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in ending1]
            ending2_idx = [word_vocab.word2idx[w] if w in word_vocab.word2idx else word_vocab.word2idx['<oov>'] for w in ending2]

            return_examples.append({
                "plot1": plot1_idx,
                "plot2": plot2_idx,
                "plot3": plot3_idx,
                "plot4": plot4_idx,
                "ending1": ending1_idx,
                "ending2": ending2_idx,
                "label": label
            })

        return return_examples

    def batchify(self, examples, bsz, shuffle=False):
        """
        Each example is a dict containing three items: short, long, context
        Therefore we need to pad three matrices
        :param examples:
        :param bsz:
        :param shuffle:
        :return:
        """
        keys = list(examples[0].keys())

        if shuffle:
            random.shuffle(examples)
        nbatch = len(examples) // bsz
        batches = []

        for i in range(nbatch):
            # Pad batches to maximum sequence length in batch
            batch_examples = examples[i * bsz:(i + 1) * bsz]
            batch = {}

            for key in keys:
                if key == 'label':
                    data = [int(x[key][0]) for x in batch_examples]
                    batch[key] = data
                else:
                    data = [x[key] for x in batch_examples]
                    # subtract 1 from lengths b/c includes BOTH starts & end symbols
                    lengths = [len(x) for x in data]

                    # do not sort here (sort items by length decreasing)
                    # data, lengths = length_sort(data, lengths)

                    # find length to pad to
                    maxlen = max(lengths)
                    for x in data:
                        zeros = (maxlen - len(x)) * [0]
                        x += zeros

                    data = torch.LongTensor(np.array(data))
                    lengths = torch.LongTensor(np.array(lengths))
                    batch[key] = (data, lengths)
            batches.append(batch)

        return batches
