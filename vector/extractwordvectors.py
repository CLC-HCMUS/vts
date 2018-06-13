__author__ = 'HyNguyen'

import time
import numpy as np

from gensim.models import word2vec
from nltk.corpus import brown
from nltk.corpus import treebank
import nltk
import xml.etree.ElementTree as ET
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_word_from_data( data = "duc04", for_w2v = "word2vec"):
    data_dir = ""
    if data == "duc04":
        data_dir = "/Users/HyNguyen/Documents/Research/Data/duc2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
    elif data == "duc05":
        data_dir = "/Users/HyNguyen/Documents/Research/Data/duc2005/DUC2005_Summarization_Documents/duc2005_docs"

    vocab = {}

    for counter,cluster_id in enumerate(os.listdir(data_dir)):
        if cluster_id[0] == ".":
            continue
        list_file = os.listdir(data_dir + "/" + cluster_id)
        for file_name in list_file:
            if file_name[0] == ".":
                continue
            tree = ET.parse(data_dir + "/" + cluster_id + "/" + file_name)
            root = tree.getroot()
            text_tag = None
            if data == "duc04":
                text_tag = root._children[3]
            elif data == "duc05":
                text_tag = root._children[5]
            text = text_tag.text.replace("\n", " ")
            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                if for_w2v == "word2vec":
                    words = nltk.word_tokenize(sentence.strip())
                else:
                    words = nltk.word_tokenize(sentence.strip().lower())
                for word in words:
                    if word in vocab:
                        vocab[word] = +1
                    else:
                        vocab[word] = 1
    return vocab

def savecode():
    print("save code")
    # vocab_4_w2v = collect_word_from_data("duc05", for_w2v="word2vec")
    # vocab_4_glove = collect_word_from_data("duc05", for_w2v="glove")
    # with open("vocab.duc04.w2v.pickle", mode="wb") as f:
    #     pickle.dump(vocab_4_w2v,f)
    # with open("vocab.duc04.glove.pickle", mode="wb") as f:
    #     pickle.dump(vocab_4_glove,f)

    # vocab = {}
    # w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/PycharmProjects/summarynew/model/word2vec.txt",binary=False)
    # for key in w2v.vocab:
    #     vocab[key] = 1
    # with open("vocab.w2v.pickle", mode="wb") as f:
    #     pickle.dump(w2v.vocab,f)

    # w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/GoogleNews-vectors-negative300.bin",binary=True)
    # with open("vocab.w2v.pickle", mode="wb") as f:
    #     pickle.dump(w2v.vocab,f)
    #
    # glove = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/glove.400k.txt",binary=False)
    # with open("vocab.glove.pickle", mode="wb") as f:
    #     pickle.dump(glove.vocab,f)



    fo = open("check.duc04.cnndaily.txt",mode="w")
    with open("vocab.duc04.w2v.pickle",mode="rb") as f:
        vocab_duc04 = pickle.load(f)

    with open("vocab.cnndaily.pickle",mode="rb") as f:
        vocab_glove = pickle.load(f)

    for key in vocab_duc04.keys():
        if key not in vocab_glove:
            fo.write(key + "\n")
    fo.close()

if __name__ == "__main__":

    # fo = open("check.duc04.w2v.txt",mode="w")
    with open("vocab.duc04.w2v.pickle",mode="rb") as f:
        vocab_duc04 = pickle.load(f)

    print len(vocab_duc04.keys())

    # with open("vocab.w2v.pickle",mode="rb") as f:
    #     vocab_glove = pickle.load(f)
    #
    # words = []
    # for key in vocab_duc04.keys():
    #     if key not in vocab_glove:
    #         words.append(key)
    #
    # words = sorted(words)
    # for word in words:
    #     fo.write(word + "\n")
    # fo.close()

