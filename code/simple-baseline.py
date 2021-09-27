!pip3 install scikit-learn
!pip3 install pymagnitude

from google.colab import drive
drive.mount('/content/gdrive')

from nltk.corpus import conll2002
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from pymagnitude import *
import numpy as np

# Change to local location of the cross-lingual embeddings
vectors_en = Magnitude("/content/gdrive/My Drive/englishtraining.magnitude")
vectors_es = Magnitude("/content/gdrive/My Drive/spanishtesting.magnitude")

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, language):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    if language == "en": vectors = vectors_en
    else: vectors = vectors_es

    o = str(o)
    embedding = vectors.query(word)

    features = []
    for place in range(vectors.dim):
        features.append((o + 'embedding' + str(place), embedding[place]))
    return features
    

def word2features(sent, i, language):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = np.array([])
    if language == "en": vectors = vectors_en
    else: vectors = vectors_es

    # the window around the token
    for o in [-1,0,1]:
        embedding = np.zeros(vectors.dim)
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            embedding = vectors.query(word)
        features = np.concatenate((features, embedding))
    
    return features

def read_conll(file_path):
    sentences = []
    cur_sent = []
    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            if line == "":
                sentences.append(cur_sent)
                cur_sent = []
            else:
                cur_sent.append(tuple(line.split()))
    return sentences

if __name__ == "__main__":
    # Load the training data
    train_sents = read_conll('/content/gdrive/My Drive/eng.train')
    #train_sents = read_conll('nltk_data/corpora/conll2002/esp.train')
    test_sents = read_conll('/content/gdrive/My Drive/nltk_data/corpora/conll2002/esp.testb')
    # train_sents = train_sents[1:10]
    # test_sents = test_sents[1:10]
    
    train_feats = []
    train_labels = []

    count = 0

    for sent in train_sents:
        count += 1
        print(count)
        for i in range(len(sent)):
            feats = word2features(sent,i,"en")
            train_feats.append(feats)
            train_labels.append(sent[i][-1])


    scaler = preprocessing.StandardScaler(with_mean=False).fit(train_feats)
    X_train = scaler.transform(train_feats)

    # TODO: play with other models
    model = LogisticRegression(random_state=0, solver="sag", verbose=1)
    model.fit(X_train, train_labels)

    print(np.sum(np.abs(model.coef_[:, 0:300])))
    print(np.sum(np.abs(model.coef_[:, 300:600])))
    print(np.sum(np.abs(model.coef_[:, 600:900])))

    test_feats = []
    test_labels = []

    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i,"es")
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = scaler.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("/content/gdrive/My Drive/results.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python score.py results.txt")