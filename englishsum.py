__author__ = 'HyNguyen'

import numpy as np
import time
import nltk
import xml.etree.ElementTree as ET
import pickle
import Levenshtein
from copy import deepcopy
import os
from stemming.porter2 import stem
from submodular.multsum_preprocess import preprocess as himpreprocess
from gensim.models import word2vec,doc2vec
import subprocess
# from skipthoughts import skipthoughts
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from submodular import multsum_clustering
from submodular.analyze_sentiment import negative_emotions,positive_emotions,analyze_sentiment

import itertools

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# utility functions
def cosine(a, b):
    c =  np.dot(a,b)
    d =  np.linalg.linalg.norm(a)*np.linalg.linalg.norm(b)
    if d == 0:
        return 0
    return c/d
def clear_eye(a):
    for idx in range(a.shape[0]):
        a[idx,idx] = 0
    return a

def euclid(a,b):
    return np.linalg.linalg.norm(a-b)

def similarity(a,b, mode):
    if (mode == 0):
        return cosine(a,b)
    elif mode == 1:
        return euclid(a,b)

def intersectionSet(a,b):
    return list( set(a) & set(b))

def sim_string(a,b):
    c = " ".join(a)
    d = " ".join(b)
    distance = Levenshtein.ratio(c, d)
    return distance

def scale_0_1(matrix):
    maxeval = np.max(matrix)
    mineval = np.min(matrix)
    matrix = (matrix - mineval)/(maxeval - mineval)
    return matrix

def check_sent_in_flatsents(flat_sents, sent_check):
    for sent in flat_sents:
        distance = sim_string(sent, sent_check)
        if distance > 0.6:
            return False
    return True

def filter_flat_sentence(flat_sentences, tags):
    results_sent = []
    result_tags = []
    for sent,tag in zip(flat_sentences,tags):
        if check_sent_in_flatsents(results_sent,sent) is True:
            results_sent.append(sent)
            result_tags.append(tag)
    return results_sent,result_tags

def get_stopwords(stopwordsFilename):
  f = open(stopwordsFilename, 'r')
  stopwords = f.read().split()
  return stopwords

# bigram-iftdf
def get_sentences_bags_flat(flat_sentences, stopwords):
  """
  Params:
    stopwords: list stopwords
    documents: documents: list document,
                                document, list sentence,
                                                sentence, list word
  Returns:
    sentence_bag_list: list sentence bag (dict of bi-gram words)
  """
  sentence_bag_list = list()

  first = True
  for sentence in flat_sentences:
      #if first:
      #  print 'First sentence, (creating bags).'
      #  print sentence
      current_sentence = dict()
      if len(sentence) > 0:
          #words = re.split(REGEX_SPACE, sentence)
          # Input is already split. It is now a list of words.
          prev = None
          for w in sentence:
              w = w.replace("_", "").replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace("-", "")
              if not w:
                  continue
              w = w.lower()
              stemmed = stem(w)
              if prev is not None:
                  bigram = prev+" "+stemmed
                  current_sentence[bigram] = current_sentence.get(bigram,0)+1
                  #end, bigrams

              if w not in stopwords:
                  current_sentence[stemmed] = current_sentence.get(stemmed,0)+1
                  prev = stemmed
              else:
                  prev = w

      sentence_bag_list.append(current_sentence)
      first = False
  return sentence_bag_list

def get_def_sentsims(flat_sentences, stopwords, idfs):


    sentences_bags = get_sentences_bags_flat(flat_sentences, stopwords) #list of dicts, from word to wordcount
    #print(sentences_bags)

    vocabulary_s = set() #list of strings
    for sentence in sentences_bags:
        for term in sentence:
            vocabulary_s.add(term)

    vocabulary = list(vocabulary_s)
    vocabulary.sort()

    #print vocabulary

    vocabularyIndices = dict()
    for i in range(0,len(vocabulary)):
        vocabularyIndices[vocabulary[i]] = i

    # creating arrays for containing sentence vectors
    # each row is a sentence, each column corresponds to a word.
    sentenceTFIDFVectors = np.zeros((len(vocabulary),len(sentences_bags)))
    sentenceIDFVectors = np.zeros((len(vocabulary),len(sentences_bags)))

    if not idfs:
        # The following is what lin-bilmes did, if documents contains each document in a cluster.
        with open("model/idfs-en.pickle", mode="rb") as f:
            idfs = pickle.load(f)
    #print(idfs)

    # Denominators for the cosine similarity computation: #/
    tfidfden = np.zeros((len(sentences_bags)))
    idfden = np.zeros((len(sentences_bags)))
    for i in range(0, len(sentences_bags)):
        for term in sentences_bags[i]:
            tf = sentences_bags[i][term]
            idf = idfs.get(term,None)
            if not idf:
                #Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
                idf = 1.0
                idfs[term] = idf
                #print("No idf for "+term+"! ")

            if not tf:
                print("No tf for "+term+"! STRANGE!")
            #Double tfidf = ((1+Math.log10(tf))*idf) #manning coursera nlp-course
            tfidf = tf*idf #lin-bilmes paper.

            sentenceTFIDFVectors[vocabularyIndices[term]][i] = tfidf
            sentenceIDFVectors[vocabularyIndices[term]][i] = idf

            tfidfden[i] += tfidf * tfidf
            idfden[i] += idf * idf

        tfidfden[i] = np.sqrt(tfidfden[i])
        idfden[i] = np.sqrt(idfden[i])
    #------
    # Numerators for the cosine similarity computation: */
    tfidfsim = np.eye(len(sentences_bags))
    idfdist = np.zeros((len(sentences_bags),len(sentences_bags)))
    sentenceTFIDFEuclidean = np.zeros((len(sentences_bags),len(sentences_bags)))

    for i in range(0,len(sentences_bags)):
        for j in range(0,len(sentences_bags)):
            euclideanSum = 0.0; tfidfnum = 0.0; idfnum = 0.0
            for term in sentences_bags[i]:
                tf_i = sentences_bags[i].get(term,0)
                tf_j = sentences_bags[j].get(term,0)
                idf = idfs[term]
                if not idf:
                    #Ugly hack. Because of some mismatch in sentence splitting on DUC, you sometimes get idfs not found for some bigrams. Will treat as if present in one document.
                    idf = 1.0
                    idfs[term] = idf
                    print("No idf for "+term+"! ")

                euclideanSum += np.power(tf_i*idf-tf_j*idf, 2)

                #tfidf =  ((1+Math.log10(tf))*idf) #manning coursera nlp-course
                tfidf_i = tf_i*idf #lin-bilmes paper.
                tfidf_j = tf_j*idf #lin-bilmes paper.
                tfidfnum += tfidf_i * tfidf_j
                idfnum += idf * idf

            if tfidfden[i]==0 or tfidfden[j]==0:
                tfidfsim[i][j] = tfidfsim[j][i] = 0.0
            else:
                tfidfsim[i][j] = tfidfsim[j][i] = tfidfnum / (tfidfden[i] * tfidfden[j])
            if idfden[i]==0 or idfden[j]==0:
                idfdist[i][j] = idfdist[j][i] = 1.0
            else:
                idfdist[i][j] = idfdist[j][i] = 1.0 - idfnum / (idfden[i] * idfden[j])
            sentenceTFIDFEuclidean[i][j] = sentenceTFIDFEuclidean[j][i] = np.sqrt(euclideanSum)

    ret_dict = dict()
    ret_dict["tfidf_cosine"] = tfidfsim
    ret_dict["tfidf_euclidean"] = sentenceTFIDFEuclidean
    ret_dict["idf_dist"] = idfdist
    ret_dict["idf_vectors"] = sentenceIDFVectors
    ret_dict["tfidf_vectors"] = sentenceTFIDFVectors

    #for i in range(0,sentenceIDFVectors.shape[0]):
    #  for j in range(0,sentenceIDFVectors.shape[1]):
    #    print(str(sentenceIDFVectors[i][j])+" ")
    #  print("\n")

    return ret_dict

# sentiment

def get_sim_sentiment_matrix(sentences):
    return analyze_sentiment(sentences, negative_emotions, positive_emotions)

# get similarity matrix
def get_levenshtein_matrix(flat_sentences):
    levenshtein_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    for i in range(0,len(flat_sentences)):
        for j in range(i+1,len(flat_sentences)):
            distance = sim_string(flat_sentences[i],flat_sentences[j])
            levenshtein_matrix[i][j] = distance
            levenshtein_matrix[j][i] = levenshtein_matrix[i][j]

    maxeval = np.max(levenshtein_matrix)
    mineval = np.min(levenshtein_matrix)

    levenshtein_matrix = (levenshtein_matrix - mineval)/(maxeval - mineval)
    for i in range(0,levenshtein_matrix.shape[0]):
        levenshtein_matrix[i][j] = 0
    return levenshtein_matrix

# def get_skipthought_matrix(flat_sentences, origin_tags, skipthought_model):
#
#     sents = []
#     for sentence,tag in zip(flat_sentences, origin_tags):
#         sents.append(" ".join(sentence))
#
#     sents_vector = skipthoughts.encode(skipthought_model, sents, verbose=False)
#
#     # with open("vector/duc04vector/skipthought.pickle",mode="rb") as f:
#     #     skt_vec = pickle.load(f)
#     #
#     # print len(flat_sentences), len(origin_tags) ,sents_vector.shape
#     # assert len(flat_sentences) == sents_vector.shape[0]
#     # assert len(origin_tags) == sents_vector.shape[0]
#
#     skipthought_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
#     sent_vec = []
#     for i in range(0,sents_vector.shape[0]):
#         sent_vec.append(sents_vector[i])
#         # print(origin_tags[i])
#         # print(skt_vec[origin_tags[i]])
#         # print(sents_vector[i])
#         # assert np.array_equal(sents_vector[i],skt_vec[origin_tags[i]]) == True
#         # print(" ")
#         for j in range(i+1, sents_vector.shape[0]):
#             skipthought_matrix[i][j] = cosine(sents_vector[i],sents_vector[j])
#             skipthought_matrix[j][i] = skipthought_matrix[i][j]
#
#     skipthought_matrix = scale_0_1(skipthought_matrix)
#     sent_vec = np.array(sent_vec)
#
#     return skipthought_matrix, sent_vec



def get_filtered_sim(origin_sentences):

    flat_sentences = []
    stopwords = get_stopwords("english_stopwords.txt")
    for sentence in origin_sentences:
        sent_tmp = []
        for word in sentence:
            if word.isalnum():
                word = word.lower()
                if word not in stopwords:
                    stemmed = stem(word)
                    sent_tmp.append(stemmed)
        flat_sentences.append(sent_tmp)
    # print len(flat_sentences)

    filter_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    for i in range(0,len(flat_sentences)):
        for j in range(i+1,len(flat_sentences)):
            if len(flat_sentences[i]) == 0 or  len(flat_sentences[j]) == 0:
                continue
            intersection_word = intersectionSet(flat_sentences[i],flat_sentences[j])
            filter_matrix[i][j] = (len(intersection_word)*1.0)/np.sqrt(len(flat_sentences[i])+len(flat_sentences[j]))
            filter_matrix[j][i] = filter_matrix[i][j]
    filter_matrix = scale_0_1(filter_matrix)
    return filter_matrix

def get_trcmparer_sim(origin_sentences):

    flat_sentences = []
    stopwords = get_stopwords("english_stopwords.txt")
    for sentence in origin_sentences:
        sent_tmp = []
        for word in sentence:
            if word.isalnum():
                word = word.lower()
                if word not in stopwords:
                    stemmed = stem(word)
                    sent_tmp.append(stemmed)
        flat_sentences.append(sent_tmp)
    # print len(flat_sentences)

    trcmp_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    for i in range(0,len(flat_sentences)):
        for j in range(i+1,len(flat_sentences)):
            if len(flat_sentences[i]) == 0 or  len(flat_sentences[j]) == 0:
                continue
            intersection_word = intersectionSet(flat_sentences[i],flat_sentences[j])
            trcmp_matrix[i][j] = (len(intersection_word)*1.0)/(np.log(len(flat_sentences[i])+1)+np.log(len(flat_sentences[j])+1))
            trcmp_matrix[j][i] = trcmp_matrix[i][j]
    trcmp_matrix = scale_0_1(trcmp_matrix)
    return trcmp_matrix

def get_word_embedding(word, w2vmodel):
    if word in w2vmodel:
        return w2vmodel[word]
    elif stem(word) in w2vmodel:
        return w2vmodel[stem(word)]
    elif word.lower() in w2vmodel:
        return w2vmodel[word.lower()]
    elif stem(word.lower()) in w2vmodel:
        return w2vmodel[stem(word.lower())]
    else:
        return None
def get_sent_embedding_w2v(sent, w2vmodel, mode, tfidf_vectorizer = None):
    result_vec = []
    set_words = set(sent)

    for w_idx,word in enumerate(set_words):
        if word.isalnum() == False:
            continue
        word_vector = get_word_embedding(word,w2vmodel)
        if word_vector is None:
            # print word
            continue
        tf_word = sent.count(word)
        if tfidf_vectorizer is not None:
            stemmed = stem(word)
            if word in tfidf_vectorizer.vocabulary_:
                idf_word = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
            elif stemmed in tfidf_vectorizer.vocabulary_:
                idf_word = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[stemmed]]
            else:
                idf_word = 0
            tf_idf_word = tf_word * idf_word
            tf_idf_array = np.array([tf_idf_word])
            word_vector = np.concatenate((word_vector, tf_idf_array))
        result_vec.append(word_vector)
    result_vec = np.array(result_vec)
    if mode == "mean":
        return np.mean(result_vec,axis=0)
    else:
        return np.sum(result_vec,axis=0)

def get_w2v_sim_matrix(flat_sents, w2vmodel, mode, tfidf_vectorizer = None):
    w2v_idf_matrix = np.zeros((len(flat_sents), len(flat_sents)),dtype=np.float32)
    sent_vec = []
    for i in range(0,len(flat_sents)):
        vec_i = get_sent_embedding_w2v(flat_sents[i],w2vmodel,mode=mode,tfidf_vectorizer=tfidf_vectorizer)
        sent_vec.append(vec_i)
        for j in range(i+1,len(flat_sents)):
            if len(flat_sents[j]) == 0 or len(flat_sents[i]) == 0:
                w2v_idf_matrix[i][j] = 0
                w2v_idf_matrix[j][i] = 0
                continue
            vec_j = get_sent_embedding_w2v(flat_sents[j],w2vmodel,mode=mode,tfidf_vectorizer = tfidf_vectorizer)
            value = cosine(vec_i,vec_j)
            if np.isnan(value) or np.isinf(value):
                print("got nan or inf")
                value = 0
            w2v_idf_matrix[i][j] = cosine(vec_i,vec_j)
            w2v_idf_matrix[j][i] = w2v_idf_matrix[i][j]
    w2v_idf_matrix = scale_0_1(w2v_idf_matrix)
    sent_vec = np.array(sent_vec)
    return w2v_idf_matrix, sent_vec

def get_doc2vec_matrix2(flat_sentences, flat_tags, d2v_model):

    d2v_idf_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    sent_vec = []
    for tag, sent in zip(flat_tags, flat_sentences):
        if tag in d2v_model.docvecs:
            sent_vec.append(d2v_model.docvecs[tag])
        else:
            print("HyHy attentions: ", tag, sent)
            sent_vec.append(None)

    for i in range(0,len(sent_vec)):
        vec_i = sent_vec[i]
        if vec_i is None:
            continue
        for j in range(i+1,len(sent_vec)):
            vec_j = sent_vec[j]
            if vec_j is None:
                continue
            d2v_idf_matrix[i][j] = cosine(vec_i,vec_j)
            d2v_idf_matrix[j][i] = d2v_idf_matrix[i][j]

    d2v_idf_matrix = scale_0_1(d2v_idf_matrix)
    sent_vec = np.array(sent_vec)
    return d2v_idf_matrix, sent_vec

def get_iftdf_sim_matrix(flat_sentences, tfidf_vectorizer):

    tfidf_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    sentences = [" ".join(words) for words in flat_sentences ]
    sent_vec = np.array(tfidf_vectorizer.transform(sentences).A, dtype=np.float32)
    for i in range(0,len(sent_vec)):
        vec_i = sent_vec[i]
        if vec_i is None:
            continue
        for j in range(i+1,len(sent_vec)):
            vec_j = sent_vec[j]
            if vec_j is None:
                continue
            tfidf_matrix[i][j] = cosine(vec_i,vec_j)
            tfidf_matrix[j][i] = tfidf_matrix[i][j]

    tfidf_matrix = scale_0_1(tfidf_matrix)
    sent_vec = np.array(sent_vec)
    return tfidf_matrix, sent_vec

def get_d2v_w2v_matrix(flat_sentences, flat_tags, d2v_vec, w2vmodel , alpha):
    sent_vec = []
    for tag, sent in zip(flat_tags, flat_sentences):
        sent_vec_d2v = None
        if tag in d2v_vec:
            sent_vec_d2v = d2v_vec[tag]
        else:
            print("HyHy attentions: ", tag, sent)

        sent_vec_w2v = get_sent_embedding_w2v(sent,w2vmodel,mode="mean",tfidf_vectorizer=None)

        if sent_vec_w2v is None or sent_vec_d2v is None:
            print("HyHy attentions none none: ", tag, sent)
            break
        sent_vec.append(alpha*sent_vec_w2v + (1-alpha)*sent_vec_d2v)

    d2v_w2v_idf_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    for i in range(0,len(sent_vec)):
        vec_i = sent_vec[i]
        if vec_i is None:
            continue
        for j in range(i+1,len(sent_vec)):
            vec_j = sent_vec[j]
            if vec_j is None:
                continue
            d2v_w2v_idf_matrix[i][j] = cosine(vec_i,vec_j)
            d2v_w2v_idf_matrix[j][i] = d2v_w2v_idf_matrix[i][j]

    d2v_w2v_idf_matrix = scale_0_1(d2v_w2v_idf_matrix)
    sent_vec = np.array(sent_vec)
    return d2v_w2v_idf_matrix, sent_vec

def get_sents_vec_from_tags_dictvec(tags, dict_vec):
    sents_vec = []
    for tag in tags:
        sents_vec.append(dict_vec[tag])
    sents_vec = np.array(sents_vec,dtype=np.float32)
    return sents_vec

def get_sim_matrix_from_sents_vec(sents_vec):
    w2v_idf_matrix = np.zeros((sents_vec.shape[0], sents_vec.shape[0]),dtype=np.float32)
    for i in range(0,sents_vec.shape[0]):
        vec_i = sents_vec[i]
        for j in range(i+1,sents_vec.shape[0]):
            vec_j = sents_vec[j]
            w2v_idf_matrix[i][j] = cosine(vec_i,vec_j)
            w2v_idf_matrix[j][i] = w2v_idf_matrix[i][j]
    w2v_idf_matrix = scale_0_1(w2v_idf_matrix)
    return w2v_idf_matrix

def get_agreegate_matrix(matrices, mode = "multi"):
    agree_matrix = matrices[0]
    for i in range(1,len(matrices)):
        if mode == "multi":
            agree_matrix*= matrices[i]
        elif mode == "add":
            agree_matrix+= matrices[i]
    agree_matrix = scale_0_1(agree_matrix)
    return agree_matrix

def get_linear_combine(matrices, alpha):
    assert len(matrices) == 2
    return alpha*matrices[0] + (1-alpha)*matrices[1]

class SummaryFunction(object):

    @staticmethod
    def summary_is_too_short(selected, length_sen, lengthUnit = "word", summarySize = 100):
        """
        check selected sentence is too short
        Params:
          selected: list sentence index
          documents: documents
          lengthUnit: UNIT_CHARACTERS, UNIT_WORDS, UNIT_SENTENCES -- count
          summarySize: size for lengthUnit
        Return:
          True: len(selected) < summarySize
          False: len(selected) > summarySize
        """
        if lengthUnit == "word":
            length_summary = sum([length_sen[idx] for idx in selected ])
            return length_summary < summarySize
        elif lengthUnit == "sent":
            return len(selected) < summarySize

    @staticmethod
    def getK(N):
        K = (int)(0.2 * N + 0.5)
        if K == 0: K = 1
        return K

    @staticmethod
    def L1(S, w, alpha, a):
        """
        Params:
          S: summary list sentence
          w: similarity matrix
          alpha: alpha of submodular in L1
          a: involve with alpha
        Return:
          res: quality of S
        """
        if not alpha:
            alpha = a/(1.0*w.shape[0])
        res = 0.0
        for i in range(0, w.shape[0]):
            sum_val = 0.0; sumV = 0.0
            for j in S:
                sum_val += w[i][j]
            for k in range(0,w.shape[0]):
                sumV += w[i][k]
            sumV *= alpha
            res += min(sum_val, sumV)
        return res

    @staticmethod
    def R1(S, w, clustering, K):
        """
        Params:
          S: summary list sentence
          w: similarity matrix
          clustering: identify cluster of sentence, clustering[i] is cluster number of sentence i
          K: number of cluster
        Return:
          res: quality of S
        """
        N = w.shape[0]
        res = 0.0
        for k in range(0, K):
            sum_val = 0.0
            for j in S:
                if (clustering [j] == k):
                    # sumV is called r_j in paper.
                    sumV = 0.0
                    for i in range(0,N):
                        sumV += w [i][j]
                    sum_val += sumV / N
            res += np.sqrt(sum_val)
        return res


    @classmethod
    def submodular(cls, similarity_matrix, sentvector, flat_sent, lengthUnit = "word", summarySize = 100, use_agreegate = True):
        # Lin-Bilmes 2010: trade off
        A = 5.0
        DEFAULT_LAMBDA = 6.0

        length_sent = [len(words) for words in flat_sent]

        # print "Clustering ... ",
        start = time.time()
        K = cls.getK(similarity_matrix.shape[0])

        if use_agreegate is True:
            clustering = multsum_clustering.get_clustering_by_similarities(similarity_matrix,K)
        else:
            kmean = KMeans(n_clusters=K)
            clustering = kmean.fit_predict(sentvector)
        end_clustering = time.time()
        # print "time: ", end_clustering - start
        selected = list()
        # print "Sub-modular ... ",
        while cls.summary_is_too_short(selected, length_sent, lengthUnit, summarySize):
            max_val = 0.0
            argmax = None
            for i in range(0,similarity_matrix.shape[0]):
                if i not in selected:# and i not in discarded:
                    selected.append(i)
                    curr = cls.L1 (selected, similarity_matrix, None, A) + DEFAULT_LAMBDA * cls.R1(selected, similarity_matrix, clustering, K)
                    # as in Lin-Bilmes 2010: */
                    if curr > max_val:
                        argmax = i
                        max_val = curr
                    selected.remove(i)
            if argmax != None:
                selected.append(argmax) #internal: zero-based.
            else:
                break

        if len(selected) > 1:
            if sum([length_sent[idx] for idx in selected ]) > summarySize + 10:
                selected.pop()

        end_sub = time.time()
        # print "time: ", end_sub - end_clustering
        return sorted(selected)

    @classmethod
    def mmr_simple(cls, sim_matrix, flat_sent, eliminate_thres = 0.5, summarySize = 100):

        #threshold
        budget = summarySize

        length_sent = [len(words) for words in flat_sent]

        n = len(sim_matrix)
        # rank the sentence
        score = np.sum(sim_matrix,axis = 1)

        summary_set = []
        #select the sentence
        while (budget > 0 and np.sum(score) != -1 * n):
            select_sentence = np.argmax(score)
            score[select_sentence] = -1

            ##check the eliminate threshold
            # if sim(select,s) > thresh --> fail: s \in summary
            flag = True
            for s in summary_set:
                if (sim_matrix[select_sentence, s] > eliminate_thres): flag = False

            if (flag == True):
                if (length_sent[select_sentence] <= budget):
                    summary_set.append(select_sentence)
                    budget -= length_sent[select_sentence]

        if sum([length_sent[idx] for idx in summary_set ]) > summarySize + 10:
            summary_set.pop()

        return sorted(summary_set)


    @classmethod
    def mmr_pagerank(cls, sim_matrix, flat_sent , eliminate_thres = 0.5, summarySize = 100):

        import networkx as nt

        budget = summarySize

        length_sent = [len(words) for words in flat_sent]

        graph = nt.Graph()

        n = np.size(sim_matrix,axis=0)

        for i in range(n):
            for j in range(n):
                graph.add_edge(i, j, distance_edge=sim_matrix[i, j])

        page_rank_score = nt.pagerank(graph,weight="distance_edge")

        score = []
        for i in range(n):
            score.append(page_rank_score[i])

        summary_set = []

        while (budget > 0 and np.sum(score) != -1*n):
            select_sentence = np.argmax(score)
            score[select_sentence] = -1

            ##check
            flag = True
            for s in summary_set:
                if (sim_matrix[select_sentence,s] > eliminate_thres): flag = False

            if (flag == True):
                if (length_sent[select_sentence] < budget):
                    summary_set.append(select_sentence)
                    budget -= length_sent[select_sentence]

        return summary_set


    @classmethod
    def summary_by_name(cls, sim_matrix, sentvector_sk , flat_sent, summary_name = "submodular", summarySize = 100, mmr_threshold = 0.5, use_agreegate=False):
        if summary_name == "submodular":
            return cls.submodular(sim_matrix, sentvector_sk, flat_sent, summarySize=summarySize, use_agreegate=use_agreegate)
        elif summary_name == "simple-mmr":
            return cls.mmr_simple(sim_matrix, flat_sent, eliminate_thres=mmr_threshold, summarySize=summarySize)
        elif summary_name == "pagerank-mmr":
            return cls.mmr_pagerank(sim_matrix, flat_sent, eliminate_thres=mmr_threshold, summarySize=summarySize)



def create_summary_format_duc2004(result_path = "data" ,
                                  tfidf_vectorizer = None,
                                  feature_name = "unigramtfidf-trcomparer",
                                  summary_name = "submodular",
                                  summary_size = 100,
                                  use_aggregate = True):

    """
    Params:
        feature_name: feature for sentence representation
        summary_name: feature for optimize algorithms
        result_path: directory for generate result folder <result_path>/<summary_name>_<feature_name>
        summary_size: maximum number of word in summary
        w2vmodel_path: path to *.model of word2vec or glove if use word2vec or glove model
        d2vmodel_path: path to *.model of doc2vec if use doc2vec model
        sktmodel_path: path to model folder if use skip-thought model
    """


    result_path += "/duc04/{0}_{1}".format(summary_name,feature_name)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with open("vector/duc04.encode.pickle", mode="rb") as f:
        clusters = pickle.load(f)

    origin_tags = []
    for cluster_id, cluster in clusters:
        origin_sentences = []
        for document_id, document in cluster:
            for sentence_id, sentence in document:
                origin_sentences.append(sentence)
                origin_tags.append("{0}_{1}_{2}".format(cluster_id,document_id,sentence_id))

        origin_sentences,origin_tags = filter_flat_sentence(origin_sentences, origin_tags)
        #use this for preprocess. just deepcopy them.
        flat_sentences = deepcopy(origin_sentences)


        assert len(origin_tags) == len(flat_sentences)

        # load model
        if use_aggregate is False:
            if feature_name == "unigramtfidf":
                if tfidf_vectorizer is None:
                    with open("model/tf_idf_vectorizer_100_01.pickle", mode="rb") as f:
                        tfidf_vectorizer = pickle.load(f)
            # elif feature_name == "skipthought":
            #     if sktmodel is None:
            #         sktmodel = skipthoughts.load_model("/Users/HyNguyen/Documents/skipthoughts/models")
        else:
            tokens = feature_name.split("-")
            for token in tokens:
                if token == "unigramtfidf":
                    if tfidf_vectorizer is None:
                        print "Use aggregate, load "+ token + "...",
                        with open("model/tf_idf_vectorizer_100_01.pickle", mode="rb") as f:
                            tfidf_vectorizer = pickle.load(f)
                        print "Done"
                # elif token == "skipthought":
                #     if skt_vec is None:
                #         print "Use aggregate, load "+ token + "...",
                #         with open("vector/duc04vector/skipthought.pickle",mode="rb") as f:
                #             skt_vec = pickle.load(f)
                #         print "Done"


        sents_vec = None
        matrices = []
        # get similarity matrix
        start_sim = time.time()
        if use_aggregate == False:
            if feature_name == "unigramtfidf":
                sim_matrix, sents_vec = get_iftdf_sim_matrix(flat_sentences,tfidf_vectorizer)
            # elif feature_name == "skipthought":
            #     sim_matrix, sents_vec = get_skipthought_matrix(flat_sentences,origin_tags,sktmodel)
            #     # duc04_skt_sim_matrix[cluster_id] = sim_matrix
        else:
            tokens = feature_name.split("-")
            for token in tokens:
                if token == "unigramtfidf":
                    sim_matrix, sents_vec = get_iftdf_sim_matrix(flat_sentences,tfidf_vectorizer)
                    matrices.append(sim_matrix)
                # elif token == "skipthought":
                #     sents_vec = get_sents_vec_from_tags_dictvec(origin_tags,skt_vec)
                #     sim_matrix = get_sim_matrix_from_sents_vec(sents_vec)
                #     matrices.append(sim_matrix)
                elif token == "trcomparer":
                    sim_matrix = get_trcmparer_sim(flat_sentences)
                    matrices.append(sim_matrix)

            if tokens[-1].isdigit():
                sim_matrix = get_linear_combine(matrices, float(tokens[-1])*0.1)
            else:
                sim_matrix = get_agreegate_matrix(matrices)


        summary_set = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name,summarySize=summary_size,use_agreegate=use_aggregate)
        summary_sets = [summary_set]

        for idx, summary_set in enumerate(summary_sets):
            return_string = "\n".join( [" ".join(origin_sentences[idx_sent]) for idx_sent in summary_set ] )
            print return_string
            print ""
            with open(result_path + "/" + cluster_id + ".txt", mode="w") as f:
                f.write(return_string)



def check_exist_model(exist_model, element):
    for model in exist_model:
        if len(set(model) & set(element)) == len(element):
            if len(model) == len(element):
                return True
    return False


if __name__ == "__main__":



    aggregate_feature = ["unigramtfidf","trcomparer","skipthought"]

    summary_name = "submodular"
    feature_name = "unigramtfidf-trcomparer"

    create_summary_format_duc2004(
                feature_name=feature_name,
                summary_name=summary_name,
                result_path="data",use_aggregate=True)
