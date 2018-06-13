__author__ = 'HyNguyen'
import os
import pickle
import numpy as np
import codecs
import nltk
from copy import deepcopy
from gensim.models import word2vec, doc2vec
import time
# from skipthoughts.training import tools


def clear_eye(a):
    for idx in range(a.shape[0]):
        a[idx,idx] = 0
    return a

def load_mds_cluster(cluster_path):
    if os.path.exists(cluster_path):
        files_name = os.listdir(cluster_path)
        doc = []
        ref1 = []
        ref2 = []
        for file_name in files_name:
            if file_name.find('.body.tok.txt') != -1 :
                with codecs.open(cluster_path + '/' + file_name) as f:
                    sents = f.readlines()
                    sents_wordlist = []
                    for sent in sents:
                        words = sent.split()
                        if len(words) > 10:
                            sents_wordlist.append(words)
                    doc.append(sents_wordlist)
            elif file_name.find(".ref1.tok") != -1 :
                with codecs.open(cluster_path + '/' + file_name) as f:
                    sents = f.readlines()
                    sents_wordlist = []
                    for sent in sents:
                        words = sent.split()
                        if len(words) > 10:
                            sents_wordlist.append(words)
                    ref1+=sents_wordlist
            elif file_name.find(".ref2.tok") != -1:
                with codecs.open(cluster_path + '/' + file_name) as f:
                    sents = f.readlines()
                    sents_wordlist = []
                    for sent in sents:
                        words = sent.split()
                        if len(words) > 10:
                            sents_wordlist.append(words)
                    ref2+=sents_wordlist
        return {"doc":doc, "ref1":ref1, "ref2":ref2}
    else:
        return None

from englishsum import filter_flat_sentence, cosine, scale_0_1, SummaryFunction


# unigram Tfidf
def get_iftdf_sim_matrix(flat_sentences, tfidf_vectorizer):

    tfidf_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
    sentences = [u" ".join(words) for words in flat_sentences ]
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
    tfidf_matrix = clear_eye(tfidf_matrix)
    sent_vec = np.array(sent_vec)
    return tfidf_matrix, sent_vec


# bigram Tfidf
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
        if len(sentence) < 2: continue
        words = []
        for w in sentence:
            w = w.lower()
            w = w.replace(u"_", u"").replace(u".", u"").replace(u",", u"").replace(u"!", u"").replace(u"?", "").replace(u"-", u"")

            if not w or w.isalnum() is False or w in stopwords:
                continue
            words.append(w)

        if len(words) == 1:
            continue

        for i in range(len(words)-1):
            current_sentence[words[i] + u" " + words[i+1]] = current_sentence.get(words[i] + u" " + words[i+1],0)+1
        sentence_bag_list.append(current_sentence)
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


# W2v
def get_word_embedding(word, w2vmodel):
    if word in w2vmodel:
        return w2vmodel[word]
    elif word.lower() in w2vmodel:
        return w2vmodel[word.lower()]
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
            stemmed = word
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
    if len(result_vec) == 0:
        return np.zeros((w2vmodel.vector_size,), np.float32)
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
            w2v_idf_matrix[i][j] = value
            w2v_idf_matrix[j][i] = w2v_idf_matrix[i][j]
    w2v_idf_matrix = scale_0_1(w2v_idf_matrix)
    w2v_idf_matrix = clear_eye(w2v_idf_matrix)
    sent_vec = np.array(sent_vec)
    return w2v_idf_matrix, sent_vec


#d2v
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
    d2v_idf_matrix = clear_eye(d2v_idf_matrix)
    sent_vec = np.array(sent_vec)

    return d2v_idf_matrix, sent_vec


from englishsum import get_sim_matrix_from_sents_vec,get_sents_vec_from_tags_dictvec,get_agreegate_matrix, intersectionSet, get_stopwords

def get_stopwords(stopwordsFilename):
    with codecs.open(stopwordsFilename, 'r', encoding="utf8") as f:
        stopwords = f.read().split()
    return stopwords

def get_filtered_sim(origin_sentences):

    flat_sentences = []
    stopwords = get_stopwords("vn_stopwords.txt")
    for sentence in origin_sentences:
        sent_tmp = []
        for word in sentence:
            if word.isalnum():
                word = word.lower()
                if word not in stopwords:
                    stemmed = word
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
    stopwords = get_stopwords("vn_stopwords.txt")
    for sentence in origin_sentences:
        sent_tmp = []
        for word in sentence:
            if word.isalnum():
                word = word.lower()
                if word not in stopwords:
                    stemmed = word
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

def create_summary_format_vietnamesemds(
                                  ref_path="",
                                  result_path = "data" ,
                                  stopwords =None ,
                                  tfidf_vectorizer = None,
                                  w2vmodel = None, w2vmdssim = None,
                                  d2vmodel = None, d2v_vec = None,
                                  sktmodel = None, sktmdssim = None,
                                  biidfs = None, pvdm_vec = None, pvdbow_vec = None,
                                  feature_name = "linkTFIDF",
                                  summary_name = "submodular", use_tfidf = False, summary_size = 100, use_aggregate = True):

    with open("vector/vietnamesemds.encode.pickle", mode="rb") as f:
        clusters = pickle.load(f)

    groups = [85,130,180,220,270,340]
    counter = 0
    result_path += "/{0}_{1}".format(summary_name,feature_name)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    compute_times = []
    for group in groups:
        summary_size = group
        path_to_group_model = ref_path + "/" + str(group)
        result_path_group = "{0}/{1}".format(result_path,group)
        if not os.path.isdir(result_path_group):
            os.mkdir(result_path_group)

        files_in_result_path_group = os.listdir(result_path_group)
        for file_name in os.listdir(path_to_group_model):
            if file_name[0] == ".": continue

            if len([file_exist for file_exist in files_in_result_path_group if file_exist.find(file_name) != -1]) != 0:
                print("continue group",group, "filename", file_name)
                continue

            counter +=1
            cluster_id, _, _, _ = file_name.split(".")
            _ , cluster_id = cluster_id.split("_")
            cluster_id_str, cluster  = clusters[int(cluster_id)]

            origin_sentences = []
            origin_tags = []

            for document_id, document in cluster:
                if document_id.find("ref") != -1: continue
                for sentence_id, sentence in document:
                    origin_sentences.append(sentence)
                    origin_tags.append("{0}_{1}_{2}".format(cluster_id_str,document_id,sentence_id))

            # Xem du lieu nen quyet dinh bo dong nay
            # origin_sentences, origin_tags = filter_flat_sentence(origin_sentences, origin_tags)
            flat_sentences = deepcopy(origin_sentences)

            assert len(origin_tags) == len(flat_sentences)


            if use_aggregate is False:
                if feature_name == "unigramtfidf":
                    if tfidf_vectorizer is None:
                        with open("model/tf_idf_vectorizer_100_05-vn.pickle", mode="rb") as f:
                            tfidf_vectorizer = pickle.load(f)
                elif feature_name == "bitfidf":
                    if biidfs is None:
                        with open("model/biidfs-vn.pickle", mode="rb") as f:
                            biidfs = pickle.load(f)
                elif feature_name == "w2vcbow":
                    if w2vmodel is None:
                        w2vmodel = word2vec.Word2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_vn/w2v_bow/w2v_bow.model")
                elif feature_name == "w2v_sg_mean":
                    if w2vmodel is None:
                        w2vmodel = word2vec.Word2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_vn/w2v_sg/w2v_sg.model")
                elif feature_name == "glove_mean":
                    if w2vmodel is None:
                        w2vmodel = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/w2v_d2v_vn/glove/glove.100.new.txt", binary=False)
                elif feature_name == "pvdm":
                    if d2vmodel is None:
                        d2vmodel = doc2vec.Doc2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_vn/d2v_dm/d2v_dm.model")
                elif feature_name == "pvdbow":
                    if d2vmodel is None:
                        d2vmodel = doc2vec.Doc2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_vn/d2v_dbow/d2v_dbow.model")

            else:
                tokens = feature_name.split("-")
                for token in tokens:
                    if token == "unigramtfidf":
                        if tfidf_vectorizer is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("model/tf_idf_vectorizer_100_05-vn.pickle", mode="rb") as f:
                                tfidf_vectorizer = pickle.load(f)
                            print "Done"
                    elif token == "w2vcbow":
                        if w2v_vec is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/w2vcbow.pickle",mode="rb") as f:
                                w2v_vec = pickle.load(f)
                            print "Done"
                    elif token == "w2vmdssim":
                        if w2vmdssim is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/w2vmdssim.pickle",mode="rb") as f:
                                w2vmdssim = pickle.load(f)
                            print "Done"
                    elif token == "sktmdssim":
                        if sktmdssim is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/sktmdssim.pickle",mode="rb") as f:
                                sktmdssim = pickle.load(f)
                            print "Done"
                    elif token == "skipthought":
                        if skt_vec is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/skipthought.pickle",mode="rb") as f:
                                skt_vec = pickle.load(f)
                            print "Done"
                    elif token == "pvdm":
                        if pvdm_vec is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/pvdm.pickle",mode="rb") as f:
                                pvdm_vec = pickle.load(f)
                            print "Done"
                    elif token == "pvdbow":
                        if pvdbow_vec is None:
                            print "Use aggregate, load "+ token + "...",
                            with open("vector/mdsvector/pvdbow.pickle",mode="rb") as f:
                                pvdbow_vec = pickle.load(f)
                            print "Done"

            sents_vec = None

            start_sim = time.time()
            if use_aggregate == False:
                if feature_name == "unitfidf":
                    sim_matrix, sent_vec = get_iftdf_sim_matrix(flat_sentences,tfidf_vectorizer)
                elif feature_name == "bitfidf":
                    sentsims = get_def_sentsims(flat_sentences,stopwords=[],idfs=biidfs)
                    sim_matrix, sent_vec = sentsims["tfidf_cosine"], sentsims["tfidf_vectors"].T
                    sim_matrix = clear_eye(sim_matrix)
                elif feature_name == "w2v_bow_mean":
                    sim_matrix, sent_vec = get_w2v_sim_matrix(flat_sentences,w2vmodel=w2vmodel,mode="mean", tfidf_vectorizer=None)
                elif feature_name == "w2v_sg_mean":
                    sim_matrix, sent_vec = get_w2v_sim_matrix(flat_sentences,w2vmodel=w2vmodel,mode="mean", tfidf_vectorizer=None)
                elif feature_name == "glove_mean":
                    sim_matrix, sent_vec = get_w2v_sim_matrix(flat_sentences,w2vmodel=w2vmodel,mode="mean", tfidf_vectorizer=None)
                elif feature_name == "d2v_dm":
                    sim_matrix, sent_vec = get_doc2vec_matrix2(flat_sentences, origin_tags, d2vmodel)
                elif feature_name == "d2v_dbow":
                    sim_matrix, sent_vec = get_doc2vec_matrix2(flat_sentences, origin_tags, d2vmodel)
            else:
                matrices = []
                tokens = feature_name.split("-")
                for token in tokens:
                    if token == "unigramtfidf":
                        sim_matrix, sents_vec = get_iftdf_sim_matrix(flat_sentences,tfidf_vectorizer)
                        matrices.append(sim_matrix)
                    elif token == "w2vcbow":
                        sents_vec = get_sents_vec_from_tags_dictvec(origin_tags,w2v_vec)
                        sim_matrix = get_sim_matrix_from_sents_vec(sents_vec)
                        matrices.append(sim_matrix)
                    elif token == "skipthought":
                        sents_vec = get_sents_vec_from_tags_dictvec(origin_tags,skt_vec)
                        sim_matrix = get_sim_matrix_from_sents_vec(sents_vec)
                        matrices.append(sim_matrix)
                    elif token == "pvdm":
                        sents_vec = get_sents_vec_from_tags_dictvec(origin_tags,pvdm_vec)
                        sim_matrix = get_sim_matrix_from_sents_vec(sents_vec)
                        matrices.append(sim_matrix)
                    elif token == "pvdbow":
                        sents_vec = get_sents_vec_from_tags_dictvec(origin_tags,pvdbow_vec)
                        sim_matrix = get_sim_matrix_from_sents_vec(sents_vec)
                        matrices.append(sim_matrix)
                    elif token == "trcomparer":
                        sim_matrix = get_trcmparer_sim(flat_sentences)
                        matrices.append(sim_matrix)
                    elif token == "filtered":
                        sim_matrix = get_filtered_sim(flat_sentences)
                        matrices.append(sim_matrix)
                    elif token == "w2vmdssim":
                        sim_matrix = w2vmdssim[cluster_id_str]
                        matrices.append(sim_matrix)
                    elif token == "sktmdssim":
                        sim_matrix = sktmdssim[cluster_id_str]
                        matrices.append(sim_matrix)
                    print("feature_name: {0}, shape_matrix {1}".format(token, sim_matrix.shape))
                sim_matrix = get_agreegate_matrix(matrices)
            end_sim = time.time()

            if summary_name == "submodular":
                summary_set = SummaryFunction.summary_by_name(sim_matrix,sents_vec,flat_sentences,summary_name,summarySize=summary_size,use_agreegate=use_aggregate)
                summary_sets = [summary_set]
            else:
                summary_sets = []
                not_zero_idx = sim_matrix != 0.0
                eliminate_thres = np.mean(sim_matrix[not_zero_idx])
                min_not_zero = np.min(sim_matrix[not_zero_idx])
                max_not_zero = np.max(sim_matrix[not_zero_idx])
                mean_not_zero = np.mean(sim_matrix[not_zero_idx])
                var_not_zero = np.var(sim_matrix[not_zero_idx])
                change_value = (max_not_zero - mean_not_zero)/5
                print " min_sim {0}, max_sim {1}, threshold ".format(min_not_zero,max_not_zero),
                for theta_idx in range(5):
                    theta_mmr = mean_not_zero + change_value*theta_idx
                    sumxx = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=theta_mmr)
                    summary_sets.append(sumxx)
                # max_min05 = 0.5*(max_not_zero-min_not_zero)+min_not_zero
                # max_min06 = 0.6*(max_not_zero-min_not_zero)+min_not_zero
                # max_min07 = 0.7*(max_not_zero-min_not_zero)+min_not_zero
                # max_min08 = 0.8*(max_not_zero-min_not_zero)+min_not_zero
                # max_min09 = 0.9*(max_not_zero-min_not_zero)+min_not_zero
                # sum05 = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=max_min05)
                # sum06 = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=max_min06)
                # sum07 = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=max_min07)
                # sum08 = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=max_min08)
                # sum09 = SummaryFunction.summary_by_name(sim_matrix, sents_vec, flat_sentences,summary_name=summary_name,summarySize=summary_size, mmr_threshold=max_min09)
                # summary_sets = [sum05, sum06, sum07, sum08, sum09]
                    print "{0},".format(theta_mmr),
                print " "''

            end_summary = time.time()

            compute_times.append([end_sim - start_sim, end_summary - end_sim])
            if sents_vec is None:
                sents_vec = np.array([1])
            print "Group {0}, Summary_name {1}, sent_vec.shape {5}, feature_name {2}, time_sim {3}, time_summary {4}".format(str(group),
                                                                                                          summary_name,
                                                                                                          feature_name,
                                                                                                          end_sim - start_sim,
                                                                                                          end_summary - end_sim, sents_vec.shape)
            for idx, summary_set in enumerate(summary_sets):
                return_string = "\n".join( [" ".join(origin_sentences[idx_sent]) for idx_sent in summary_set ] )
                file_summary = result_path_group + "/" + file_name+ "."+str(idx)
                with codecs.open(file_summary, mode="w", encoding="utf8") as f:
                    f.write(return_string)
        print ("Finished group ", group)

    if len(compute_times) == 0:
        return None
    compute_times = np.array(compute_times).mean(axis=0)
    print("Time result: average similarity: {0}, average optimize {1}".format(compute_times[0], compute_times[1]))
    return compute_times


# def compute_idfs():
#
#     vietnamesemds = "data/VietnameseMDS-grouped/clusters"
#     documents = []
#     for cluster_id in range(1,201):
#         if cluster_id == 178:
#             continue
#         cluster_path = vietnamesemds+"/cluster_"+str(cluster_id)
#         print ("Processed cluster: ", cluster_id)
#         if os.path.exists(cluster_path):
#             files_name = os.listdir(cluster_path)
#             docs = []
#             for file_name in files_name:
#                 if file_name.find('.tok.txt') != -1 and file_name.find('.sum.') == -1:
#                     with codecs.open(cluster_path + '/' + file_name) as f:
#                         sents = f.readlines()
#                         sents_wordlist = []
#                         for sent in sents:
#                             words = sent.split()
#                             if len(words) > 10:
#                                 sents_wordlist.append(words)
#                         docs.append(sents_wordlist)
#             documents+=docs
#
#     idfs = get_idfs_from_doc_collection(documents,[])
#     with open("model/idfs_vn.pickle", mode="wb") as f:
#         pickle.dump(idfs,f)

# from generate_setting import generate_vietnamesemds, gen_perl_file_mds

import itertools


from englishsum import check_exist_model

if __name__ == "__main__":

    """
    submodular
    simple-mmr
    pagerank-mmr
    """
    folder_result = "data/VietnameseMDS-grouped/mysystem"

    fo = open("save_time_vn.txt", mode="w")

    list_summary =  ["simple-mmr"]
    list_feature = []
    # aggregate_feature = ["skipthought"]
    #aggregate_feature = ["filtered","trcomparer","unigramtfidf","pvdm","pvdbow","w2vmdssim","sktmdssim"]
    aggregate_feature = ["unigramtfidf","sktduc04sim","trcomparer"]
    for L in range(1, len(aggregate_feature)+1):
        for subset in itertools.combinations(aggregate_feature, L):
            list_feature.append("-".join(subset))

    folders_name = [folder_file for folder_file in os.listdir(folder_result) if os.path.isfile(folder_result+"/"+folder_file) is False]
    exist_model = []
    for folder_name in folders_name:
        summary_name, combine_feature_name = folder_name.split("_")
        fn = combine_feature_name.split("-")
        exist_model.append(fn+[summary_name])


    for summary_name in list_summary:
        for feature_name in list_feature:

            if check_exist_model(exist_model,[summary_name] + feature_name.split("-")) is True:
                print("pass",summary_name,feature_name)
                continue

            compute_times = create_summary_format_vietnamesemds(feature_name=feature_name,
                                                                summary_name=summary_name,
                                                                ref_path="data/VietnameseMDS-grouped/model",
                                                                result_path=folder_result, use_aggregate=True)
            if compute_times is not None:
                fo.write("{0}, {1}, {2}, {3}\n".format(summary_name, feature_name, compute_times[0], compute_times[1]))
            dir_peer_path = "{2}/{0}_{1}".format(summary_name,feature_name,folder_result)
    fo.close()
