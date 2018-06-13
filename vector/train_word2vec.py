__author__ = 'HyNguyen'

import gensim
# setup logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', required=True, type=str)
    parser.add_argument('-fo', required=True, type=str)
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('-worker', type=int, default=4)
    parser.add_argument('-mincount', type=int, default=5)
    parser.add_argument('-size', type=int, default=100)

    args = parser.parse_args()
    fi = args.fi
    fo = args.fo
    mode = args.model
    woker = args.worker
    min_count = args.mincount
    size = args.size

    mode = 0
    if mode == "sg":
        mode =1

    model = gensim.models.word2vec.Word2Vec(workers=woker,min_count=min_count,size=size,sg=mode)
    sentences = gensim.models.word2vec.LineSentence(fi)
    model.build_vocab(sentences)
    model.train(sentences)
    model.save(fo)
