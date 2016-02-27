__author__ = 'Giahy'

from gensim.models import word2vec
from gensim import utils
from numpy import array, average
import os

class Word2VecModel:
    def __init__(self, file_name):
        self.embsize = 100
        self.model = None
        if os.path.isfile(file_name):
            self.model = word2vec.Word2Vec.load_word2vec_format(file_name)

        if self.model is None:
            file = '../test/corpus_lower.txt'
            sentences = word2vec.LineSentence(file)
            model = word2vec.Word2Vec(sentences, size=self.embsize, window=5, min_count=5, workers=4)

            model.save_word2vec_format(file_name)

        new_dict = {}
        for key in self.model.vocab:
            new_dict[key.encode("UTF-8")] = self.model.vocab[key]
        self.model.vocab = new_dict
        self.null_word =  average(self.model.syn0, axis=0)
        self.count_except = 0
        self.count_try = 0

    def getWordEmbeddingFromString(self, word_str):
        lower_str = word_str.lower().encode('UTF-8')
        try:
            a = array(self.model[lower_str]).T
            self.count_try +=1
            return a
        except:
            b = array(self.null_word)
            self.count_except +=1
            return b

    def parseInstanceFromSentence(self, sentence_str):
        words_str = utils.to_unicode(sentence_str).split(' ')
        return array([self.getWordEmbeddingFromString(word_str) for word_str in words_str]).T

    def parseFromCorpusSentence(self, corpus_sent):
        return array([self.getWordEmbeddingFromString(word_i) for word_i in corpus_sent]).T

if __name__ == '__main__':
    w2vmodel = Word2VecModel('./vn_word2vec_model_27/100')
    # new_dict = {}
    # for key in model.vocab.keys():
    #     a = model.vocab[key]
    #     new_dict[key.encode("UTF-8")] = model.vocab[key]
    # model.vocab = new_dict
    # model.save_word2vec_format(file_name)
    a = w2vmodel

