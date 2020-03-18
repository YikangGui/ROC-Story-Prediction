"""
@author: Yang Yuan
@file: NLP_final.py
@time: 2019/12/8 14:36
"""
import pandas as pd
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

"""
    input: each comment from dataset
    output: the sentence after lemmatization and removing stopwords

"""
def word_lemmatizer(list_words):
    wordnet_lemmatizer = WordNetLemmatizer()
    res = []
    for sentence in list_words:
        sent = []
        for each_sentence in sentence:
            for word, tag in pos_tag(word_tokenize(each_sentence)):
                if word.isalpha() or word.isalnum():
                    if word not in stopwords.words('english'):
                        wntag = tag[0].lower()
                        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                        if not wntag:
                            lemma = word
                        else:
                            lemma = wordnet_lemmatizer.lemmatize(word, wntag)
                        sent.append(lemma.lower())
            sent.append("\n")
        res.append(sent)
    return res


def generate_corpus(f):
    data = pd.read_csv(f)
    temp = data.iloc[:, 1:7]
    stories = []
    for i in range(0, len(temp)):
        stories.append(temp.iloc[i:i + 1, :])
    sentences = []
    for i in stories:
        story = []
        for j in i:
            for k in i[j]:
                story.append(k)
        sentences.append(story)
    corpus = word_lemmatizer(sentences)
    return corpus


def save_corpus(corpus):
    f = open('corpus.txt', 'w')
    for i in corpus:
        f.write("$")
        for j in i:
            try:
                f.write(j)
            except UnicodeEncodeError:
                print(j)
                continue
            f.write(" ")
        f.write("\n")
    f.close()


def blockread(fh, sep):
    buf = ""
    while True:
        while sep in buf:
            pos = buf.index(sep)
            yield buf[:pos]
            buf = buf[pos + len(sep):]
        chunk = fh.read(4096)
        if not chunk:
            yield buf
            break
        buf += chunk
    return buf


def generate_dataset(sentences, data):
    story_list = []
    for i in sentences:
        story = []
        l = i.split("\n")
        for j in l:
            story.append(j)
        if story != ['']:
            story_list.append(story)
    pos_list = []
    for i in range(len(story_list) - 1):
        story = []
        for j in range(4):
            story.append(story_list[i][j])
        if data['AnswerRightEnding'][i] == 1:
            story.append(story_list[i][4])
        else:
            story.append(story_list[i][5])
        pos_list.append(story)
    neg_list = []
    for i in range(len(story_list) - 1):
        story = []
        for j in range(4):
            story.append(story_list[i][j])
        if data['AnswerRightEnding'][i] == 2:
            story.append(story_list[i][4])
        else:
            story.append(story_list[i][5])
        neg_list.append(story)
    pos = []
    for i in pos_list:
        sentence = []
        for j in i:
            for k in j.split():
                sentence.append(k)
        pos.append(sentence)
    neg = []
    for i in neg_list:
        sentence = []
        for j in i:
            for k in j.split():
                sentence.append(k)
        neg.append(sentence)
    x_pos = []
    for i in pos:
        x_pos.append(str(i))
    x_neg = []
    for i in neg:
        x_neg.append(str(i))
    x = x_pos + x_neg
    y_pos = [1] * len(x_pos)
    y_neg = [0] * len(x_neg)
    y = y_pos + y_neg
    xy = pd.DataFrame(x, y)
    xy_shuffled = shuffle(xy)
    x_shuffled = []
    for i in xy_shuffled[0]:
        x_shuffled.append(i)
    y_shuffled = xy_shuffled.index
    return x_shuffled, y_shuffled


"""
    input: corpus
    output: tagged document for doc2vec training
"""
def read_corpus(data, tokens_only=False):
    for i, line in enumerate(data):
        tokens = simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield TaggedDocument(tokens, [i])


if __name__ == '__main__':
    data = pd.read_csv('../SupportingMaterials/ROC-Story-Cloze-Data.csv')
    corpus = generate_corpus("../SupportingMaterials/ROC-Story-Cloze-Data.csv")
    save_corpus(corpus)
    temp = []
    f = open('corpus.txt')
    for each in blockread(f, '$'):
        temp.append(each)
    x_shuffled, y_shuffled = generate_dataset(temp, data)
    train_corpus = list(read_corpus(x_shuffled))
    model = Doc2Vec(vector_size=50, min_count=2, epochs=5)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    ranks = []
    vectors = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        vectors.append(inferred_vector)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
    lr = LogisticRegression(solver='lbfgs', max_iter=4000)
    lr_dense = lr.fit(vectors, y_shuffled)
    print("cross validation scores: ")
    scores_doc2vec = cross_val_score(lr_dense, vectors, y_shuffled, cv=5)
    print(scores_doc2vec)
    # Since this method is too slow and the validation accuracy is only about 55%,
    # so we did not use this method to evaluate test set.
