__author__ = 'HyNguyen'
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def dosomething():
    with open("duc04.encode.pickle", mode="rb") as f:
        clusters = pickle.load(f)

    len_doc = []
    for cluster_id,cluster in clusters:
        origin_sentences = []
        for document_id, document in cluster:
            if len(document) == 1:
                print("ttdt")
            len_doc.append(len(document))

    len_doc = np.array(len_doc)
    print(np.max(len_doc), np.min(len_doc), np.mean(len_doc))

def split_file_by_delimiter(delimiter = " . "):
    fo = open("xxx.txt",mode="w")
    with open("/Users/HyNguyen/Documents/Research/Data/vn_express.tok.txt", mode="r") as f:
        for line in f:
            sents = line.split(delimiter)
            fo.write(" .\n".join(sents))
    fo.close()

# utility functions
def cosine(a, b):
    c =  np.dot(a,b)
    d =  np.linalg.linalg.norm(a)*np.linalg.linalg.norm(b)
    if d == 0:
        return 0
    return c/d

def ilovecat_ihatecat():
    w2vmodel = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/GoogleNews-vectors-negative300.bin",binary=True)
    print w2vmodel.n_similarity(['i', 'love', 'cat'], ['i', 'hate', 'cat'])

from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    with open("duc04vector/pvdbow.pickle", mode="rb") as f:
        pvdm_vec = pickle.load(f)

    for key,value in pvdm_vec.iteritems():
        print key
        print value
        break

    print("ttdt")


    # with open("duc04.encode.pickle", mode="rb") as f:
    #     clusters = pickle.load(f)
    #
    # len_doc = []
    # for cluster_id,cluster in clusters:
    #     origin_sentences = []
    #     for document_id, document in cluster:
    #         for sentence_id, sentence in document:
    #             len_doc.append(len(sentence))
    # len_doc = np.array(len_doc)
    # print(np.max(len_doc), np.min(len_doc), np.mean(len_doc))

