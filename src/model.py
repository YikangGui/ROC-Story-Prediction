import spacy
import csv
import numpy as np
import sys
from bert_embedding import BertEmbedding
from tqdm import tqdm
from collections import Counter
import collections
import os


TRAIN_PATH = '../SupportingMaterials/ROC-Story-Cloze-Data.csv'
VAL_PATH = '../SupportingMaterials/ROC-Story-Cloze-Val.csv'
NAME_REPLACED_VAL_PATH = '../SupportingMaterials/replaced_val_1.csv'
SPACY_CORPUS = 'en_core_web_lg'

# Change the test file here!!!
ORIGINAL_TEST_PATH = '../SupportingMaterials/ROC-Story-Cloze-Test-Release.csv'
TRICK_TEST_PATH = '../SupportingMaterials/test-ner.csv'
TEST_WITH_LABEL = '../SupportingMaterials/ROC-Story-Cloze-Test.csv'


print('Loading spacy corpus...')
nlp = spacy.load(SPACY_CORPUS)
print('Loading BERT...')
bert = BertEmbedding()


connectives = 'but, still, yet, however, nevertheless, nonetheless, on the contrary, even though, although, ' \
              'despite'.split(', ')


def read_file(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return your_list


def doc2vec_spacy(text):
    doc_vec = nlp(str(text)).vector
    return doc_vec


def doc2vec_bert(text):
    doc_vec = np.sum(bert([text])[0][1], axis=0)
    return doc_vec


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    punctuation = [',', '.', '!', '?', ';', ':', '(', ')']
    return [tok.lemma_.lower() for tok in nlp.tokenizer(text) if tok.text not in punctuation]


class CosineSimilarity(object):
    def __init__(self, file_path, idf=False, connect=False, embedding=doc2vec_spacy):
        print(f'Using {embedding}')
        self.embedding = embedding
        self.connect = connect
        self.file = np.array(read_file(file_path))[1:]
        assert self.file.shape[1] == 8, 'No label!'
        if not idf:
            print('Embedding plot...')
            if connect:
                self.weights = []
                for story in self.file:
                    level = 1
                    weight = []
                    for plot in story[1:5]:
                        for connect in connectives:
                            if connect in plot:
                                level += 1
                        weight.append(np.ones(300) * level)
                    self.weights.append(weight)
                self.weights = np.array(self.weights)
            else:
                # self.weights = np.ones((self.file.__len__(), 4, 300))
                self.weights = np.tile([np.repeat(1, 300), np.repeat(2, 300), np.repeat(3, 300), np.repeat(4, 300)],
                                       (self.file.__len__(), 1, 1))
            self.plot = np.asarray([list(map(embedding, plot[1:5])) for plot in tqdm(self.file)])
            print('Embedding ending...')
            self.ending = np.asarray([list(map(embedding, ending[5:7])) for ending in tqdm(self.file)]).transpose(1, 0, 2)
        # print(self.ending.shape[0])
        assert self.ending.shape[0] == 2
        self.label = np.array(self.file[:, 7], dtype=np.int) - 1

    def calc_similarity(self):
        # self.plot2vec = np.mean(self.plot, axis=1)
        self.plot2vec = np.average(self.plot, weights=self.weights, axis=1)
        # assert self.ending.shape == self.plot2vec.shape
        nominator = np.sum(np.asarray([self.plot2vec * ending for ending in self.ending]), axis=2)
        plot2vec_ = np.linalg.norm(self.plot2vec, axis=1)[None]
        plot2vec_ = np.concatenate([plot2vec_, plot2vec_])
        ending_ = np.linalg.norm(self.ending, axis=2)
        assert plot2vec_.shape == ending_.shape
        denominator = ending_ * plot2vec_
        similarity = nominator / denominator
        return similarity

    def acc(self):
        self.similarity = self.calc_similarity()
        return np.mean(np.argmax(self.calc_similarity(), axis=0) == self.label)

    def predict(self, path):
        self.test_file = read_file(path)[1:]
        if self.connect:
            weights = []
            for story in self.file:
                level = 1
                weight = []
                for plot in story[1:5]:
                    for connect in connectives:
                        if connect in plot:
                            level += 1
                    weight.append(np.ones(300) * level)
                weights.append(weight)
            self.test_weights = np.array(weights)
        else:
            self.test_weights = np.ones((self.test_file.__len__(), 4, 300))
        self.test_plot = np.asarray([list(map(self.embedding, plot[1:5])) for plot in tqdm(self.test_file)])
        self.test_ending = np.asarray([list(map(self.embedding, ending[5:7])) for ending in tqdm(self.test_file)]).transpose(1, 0, 2)
        self.test_plot2vec = np.average(self.test_plot, weights=self.test_weights, axis=1)

        self.test_nominator = np.sum(np.asarray([self.test_plot2vec * ending for ending in self.test_ending]), axis=2)
        plot2vec_ = np.linalg.norm(self.test_plot2vec, axis=1)[None]
        self.test_plot2vec_ = np.concatenate([plot2vec_, plot2vec_])
        self.test_ending_ = np.linalg.norm(self.test_ending, axis=2)
        self.test_denominator = self.test_ending_ * plot2vec_
        similarity = self.test_nominator / self.test_denominator
        prediction = np.argmax(similarity, axis=0)
        return prediction


class WordModel(object):
    def __init__(self, file, idf=False, embedding='spacy'):
        if file == 'train':
            file_path = TRAIN_PATH
        elif file == 'val':
            file_path = VAL_PATH
        elif file == 'test':
            file_path = ORIGINAL_TEST_PATH
        else:
            raise ValueError('Wrong file!')
        print('Using', file)
        self.mode = file
        self.file = read_file(file_path)[1:]
        if self.file[0].__len__() == 8:
            self.infer_mode = False
        elif self.file[0].__len__() == 7:
            self.infer_mode = True
        else:
            raise ValueError('Wrong shape of file!')
        self.wordcount = {}
        self.word_df = {}
        self.word2idx = {}
        self.idx2word = {}
        self.embedding_matrix = None
        self.idf = idf
        self.tokenized_sent = []
        self.vectorized_sent = []
        self.label = []
        self.story_num = len(self.file)

        self.embedding_mode = embedding
        if embedding == 'spacy':
            print('Using SpaCy...')
            self.embedding = doc2vec_spacy
            self.embedding_size = 300
        elif embedding == 'bert':
            print('Using BERT')
            self.embedding = doc2vec_bert
            self.embedding_size = 768
        else:
            raise ValueError('Wrong embedding!')

        self.doc_freq()
        # self.make_vocab()
        # self.make_embedding_matrix()
        # self.vectorize()

    def doc_freq_story(self, story, label):
        """

        :param story: [plot1, plot2, plot3, plot4, ending1, ending2]
        :return:
        """
        tokenized_sent = list(map(tokenize_en, story))
        self.tokenized_sent.append(tokenized_sent)
        self.label.append(label)
        return list(set(sum(tokenized_sent, [])))

    def doc_freq(self):
        all = sum([self.doc_freq_story(f[1:7], f[7]) for f in tqdm(self.file)], [])
        word_df = Counter(all)
        self.word_df = collections.OrderedDict(sorted(word_df.items(), key=lambda kv: kv[1], reverse=True))

    def make_vocab(self):
        [self.word2idx.setdefault(k, index) for index, k in enumerate(self.word_df.keys())]
        [self.idx2word.setdefault(index, k) for index, k in enumerate(self.word_df.keys())]

    def make_embedding_matrix(self):
        print('Making embedding matrix...')
        cache_path = '../save/word_model/'
        cache_path = cache_path + self.mode + '_' + self.embedding_mode + '_embedding.npy'
        if not os.path.exists(cache_path):
            self.embedding_matrix = np.zeros((self.word2idx.__len__(), self.embedding_size))
            [self.assign_embedding(self.embedding_matrix, i, self.embedding(text)) for i, text in tqdm(self.idx2word.items())]
            np.save(cache_path, self.embedding_matrix)
        else:
            self.embedding_matrix = np.load(cache_path)

    def assign_embedding(self, matrix, index, content):
        matrix[index] = content

    def vectorize(self):
        print('Vectorizing stories...')
        self.vectorized_sent = [[[self.word2idx[token] for token in sent] for sent in story] for story in self.tokenized_sent]
        # self.story_num = len(self.vectorized_sent)

    def baseline(self):
        print('Running Baseline...')
        count = 0
        for index, story in enumerate(self.vectorized_sent):
            plot1 = np.mean(self.embedding_matrix[story[0]], axis=0)
            plot2 = np.mean(self.embedding_matrix[story[1]], axis=0)
            plot3 = np.mean(self.embedding_matrix[story[2]], axis=0)
            plot4 = np.mean(self.embedding_matrix[story[3]], axis=0)
            plot = np.mean([plot1, plot2, plot3, plot4], axis=0)
            ending1 = np.mean(self.embedding_matrix[story[4]], axis=0)
            ending2 = np.mean(self.embedding_matrix[story[5]], axis=0)

            similarity1 = np.dot(plot, ending1) / (np.linalg.norm(plot) * np.linalg.norm(ending1))
            similarity2 = np.dot(plot, ending2) / (np.linalg.norm(plot) * np.linalg.norm(ending2))

            count += (np.argmax([similarity1, similarity2]) == (int(self.label[index]) - 1))

        return count / self.story_num

    def baseline_idf(self):
        print('Running Baseline with IDF...')
        count = 0
        for index, story in enumerate(self.vectorized_sent):
            weight1 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[0]]
            plot1 = np.average(self.embedding_matrix[story[0]], weights=weight1, axis=0)
            weight2 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[1]]
            plot2 = np.average(self.embedding_matrix[story[1]], weights=weight2, axis=0)
            weight3 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[2]]
            plot3 = np.average(self.embedding_matrix[story[2]], weights=weight3, axis=0)
            weight4 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[3]]
            plot4 = np.average(self.embedding_matrix[story[3]], weights=weight4, axis=0)
            plot = np.mean([plot1, plot2, plot3, plot4], axis=0)
            weight5 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[4]]
            ending1 = np.average(self.embedding_matrix[story[4]], weights=weight5, axis=0)
            weight6 = [self.idf_weight(self.word_df[self.idx2word[s]]) for s in story[5]]
            ending2 = np.average(self.embedding_matrix[story[5]], weights=weight6, axis=0)

            similarity1 = np.dot(plot, ending1) / (np.linalg.norm(plot) * np.linalg.norm(ending1))
            similarity2 = np.dot(plot, ending2) / (np.linalg.norm(plot) * np.linalg.norm(ending2))

            count += (np.argmax([similarity1, similarity2]) == (int(self.label[index]) - 1))

        return count / self.story_num

    def baseline_nva_feature(self, path):
        self.file1 = read_file(path)[1:]
        print('Running manual feature...')
        count = 0
        for index, story in enumerate(self.file1):
            print(index)

            plot1 = nlp(story[0])
            plot1_pos = {'NOUN': [plot1.vector], 'VERB': [plot1.vector], 'ADJ': [plot1.vector], 'ADV': [plot1.vector]}
            for token in plot1:
                if token.pos_ in plot1_pos.keys():
                    plot1_pos[token.pos_].append(token.vector)
            plot1_n = np.mean(plot1_pos['NOUN'], axis=0)
            plot1_v = np.mean(plot1_pos['VERB'], axis=0)
            plot1_a = np.mean([np.mean(plot1_pos['ADJ'], axis=0), np.mean(plot1_pos['ADV'], axis=0)], axis=0)
            plot1_vec = np.concatenate([plot1_n, plot1_v, plot1_a])

            plot2 = nlp(story[1])
            plot2_pos = {'NOUN': [plot2.vector], 'VERB': [plot2.vector], 'ADJ': [plot2.vector], 'ADV': [plot2.vector]}
            for token in plot1:
                if token.pos_ in plot2_pos.keys():
                    plot2_pos[token.pos_].append(token.vector)
            plot2_n = np.mean(plot2_pos['NOUN'], axis=0)
            plot2_v = np.mean(plot2_pos['VERB'], axis=0)
            plot2_a = np.mean([np.mean(plot2_pos['ADJ'], axis=0), np.mean(plot2_pos['ADV'], axis=0)], axis=0)
            plot2_vec = np.concatenate([plot2_n, plot2_v, plot2_a])

            plot3 = nlp(story[2])
            plot3_pos = {'NOUN': [plot3.vector], 'VERB': [plot3.vector], 'ADJ': [plot3.vector], 'ADV': [plot3.vector]}
            for token in plot3:
                if token.pos_ in plot3_pos.keys():
                    plot3_pos[token.pos_].append(token.vector)
            plot3_n = np.mean(plot3_pos['NOUN'], axis=0)
            plot3_v = np.mean(plot3_pos['VERB'], axis=0)
            plot3_a = np.mean([np.mean(plot3_pos['ADJ'], axis=0), np.mean(plot3_pos['ADV'], axis=0)], axis=0)
            plot3_vec = np.concatenate([plot3_n, plot3_v, plot3_a])

            plot4 = nlp(story[3])
            plot4_pos = {'NOUN': [plot4.vector], 'VERB': [plot4.vector], 'ADJ': [plot4.vector], 'ADV': [plot4.vector]}
            for token in plot4:
                if token.pos_ in plot4_pos.keys():
                    plot4_pos[token.pos_].append(token.vector)
            plot4_n = np.mean(plot4_pos['NOUN'], axis=0)
            plot4_v = np.mean(plot4_pos['VERB'], axis=0)
            plot4_a = np.mean([np.mean(plot4_pos['ADJ'], axis=0), np.mean(plot4_pos['ADV'], axis=0)], axis=0)
            plot4_vec = np.concatenate([plot4_n, plot4_v, plot4_a])

            ending1 = nlp(story[4])
            ending1_pos = {'NOUN': [ending1.vector], 'VERB': [ending1.vector], 'ADJ': [ending1.vector], 'ADV': [ending1.vector]}
            for token in ending1:
                if token.pos_ in ending1_pos.keys():
                    ending1_pos[token.pos_].append(token.vector)
            ending1_n = np.mean(ending1_pos['NOUN'], axis=0)
            ending1_v = np.mean(ending1_pos['VERB'], axis=0)
            ending1_a = np.mean([np.mean(ending1_pos['ADJ'], axis=0), np.mean(ending1_pos['ADV'], axis=0)], axis=0)
            ending1_vec = np.concatenate([ending1_n, ending1_v, ending1_a])

            ending2 = nlp(story[5])
            ending2_pos = {'NOUN': [ending2.vector], 'VERB': [ending2.vector], 'ADJ': [ending2.vector], 'ADV': [ending2.vector]}
            for token in ending2:
                if token.pos_ in ending2_pos.keys():
                    ending2_pos[token.pos_].append(token.vector)
            ending2_n = np.mean(ending2_pos['NOUN'], axis=0)
            ending2_v = np.mean(ending2_pos['VERB'], axis=0)
            ending2_a = np.mean([np.mean(ending2_pos['ADJ'], axis=0), np.mean(ending2_pos['ADV'], axis=0)], axis=0)
            ending2_vec = np.concatenate([ending2_n, ending2_v, ending2_a])

            plot_vec = np.mean([plot1_vec, plot2_vec, plot3_vec, plot4_vec], axis=0)

            similarity1 = np.dot(plot_vec, ending1_vec) / (np.linalg.norm(plot_vec) * np.linalg.norm(ending1_vec))
            similarity2 = np.dot(plot_vec, ending2_vec) / (np.linalg.norm(plot_vec) * np.linalg.norm(ending2_vec))

            count += (np.argmax([similarity1, similarity2]) == (int(self.label[index]) - 1))

        return count / self.story_num

    @staticmethod
    def idf_weight(x):
        # return x
        if x > 600:
            return 1e-8
        # else:
        #     return 1.0
        return 1 / (1 + np.exp(x - 400))


if __name__ == '__main__':
    file = read_file(VAL_PATH)
    print('\nBaseline Model (Cosine Similarity):')
    cos = CosineSimilarity(VAL_PATH, embedding=doc2vec_spacy, connect=True)
    pred = cos.predict(ORIGINAL_TEST_PATH)
    # np.savetxt('../SupportingMaterials/samplePrediction.txt', pred, fmt='%d')
    #
    # cos_ = CosineSimilarity(VAL_PATH, embedding=doc2vec_spacy, connect=False)
    # pred_ = cos.predict(TRICK_TEST_PATH)
    #
    # print('Different in index:', np.where(pred != pred_)[0].tolist())
    # print()

    # print(f'Acc: {np.round(cos.acc() * 100, 3)}%\n')

    # print('=========================================================================')
    # print('Baseline Model (Cosine Similarity):')
    # word_model = WordModel('val', embedding='spacy')
    # acc = word_model.baseline()
    # acc1 = word_model.baseline_nva_feature(VAL_PATH)
    # print(f'Acc: {np.round(acc1 * 100, 3)}%\n')
    #
    # print('=========================================================================')
    # print('Baseline Model with idf(Cosine Similarity):')
    # word_model = WordModel('val', embedding='spacy')
    # acc = word_model.baseline_idf()
    # print(f'Acc: {np.round(acc * 100, 3)}%\n')

    # val = read_file(VAL_PATH)
    test = read_file(TEST_WITH_LABEL)
    t = np.array(test)
    print(np.mean(np.array(t[1:, 7], dtype=np.int) - 1 == pred))
