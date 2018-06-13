__author__ = 'HyNguyen'

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict

if __name__ == "__main__":
    with open("tf_idf_vectorizer_100_01.pickle", mode="rb") as f:
        tfidf_vectorizer = pickle.load(f)

    test_str = "the cat sat on the mat in the house"

    words = set(test_str.split())
    sorted_words = sorted(words)

    idfs_dict = {}

    for word in sorted_words:
        index = tfidf_vectorizer.vocabulary_.get(word,-1)
        if index == -1:
            idfs_dict[word] = 0
        else:
            idfs_dict[word] = tfidf_vectorizer.idf_[index]

    xxx = sorted(idfs_dict.items())

    result_str = ""

    for key, value in xxx:
        result_str += "'{0}':{1}, ".format(key,value)

    print result_str