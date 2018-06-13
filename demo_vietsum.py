__author__ = 'HyNguyen'

import argparse
import pickle
import os
# from skipthoughts.training import tools
import numpy as np
from englishsum import get_iftdf_sim_matrix, get_agreegate_matrix, SummaryFunction
from vietsum import get_trcmparer_sim
tfidf_vectorizer = None
skt_model = None
import codecs

# def get_skipthought_matrix(flat_sentences, skipthought_model):
#
#     sents = []
#     for sentence in flat_sentences:
#         sents.append(" ".join(sentence))
#
#     sents_vector = tools.encode(skipthought_model, sents, verbose=False)
#
#     skipthought_matrix = np.zeros((len(flat_sentences), len(flat_sentences)),dtype=np.float32)
#     sent_vec = []
#     for i in range(0,sents_vector.shape[0]):
#         sent_vec.append(sents_vector[i])
#         for j in range(i+1, sents_vector.shape[0]):
#             skipthought_matrix[i][j] = cosine(sents_vector[i],sents_vector[j])
#             skipthought_matrix[j][i] = skipthought_matrix[i][j]
#
#     skipthought_matrix = scale_0_1(skipthought_matrix)
#     sent_vec = np.array(sent_vec)
#
#     return skipthought_matrix, sent_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-input_dir', type=str, default="data/cluster200")
    parser.add_argument('-output_dir', type=str)


    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if tfidf_vectorizer is None:
        print "Use aggregate, load BOW TFIDF model ...",
        with open("model/tf_idf_vectorizer_100_05-vn.pickle", mode="rb") as f:
            tfidf_vectorizer = pickle.load(f)

    # if skt_model is None:
    #     print "Use aggregate, load SKT model ...",
    #     with open("/Users/HyNguyen/Documents/skt_vn/model_skt.pickle", mode="rb") as f:
    #         skt_model = pickle.load(f)
    #     print "Done"

    flat_sentences = []
    list_file = [file_name for file_name in os.listdir(input_dir) if file_name.find(".body.tok.txt") != -1]
    for file_name in list_file:
        if file_name[0] == ".":continue
        with codecs.open(input_dir + "/" + file_name, mode="r",encoding="utf8") as f:
            for line in f:
                words = line.split()
                if len(words) < 5: continue
                flat_sentences.append(line.split())

    sim_matrix1, sents_vec1 = get_iftdf_sim_matrix(flat_sentences,tfidf_vectorizer)
    # sim_matrix2, sents_vec2 = get_skipthought_matrix(flat_sentences,skt_model)
    sim_matrix3 = get_trcmparer_sim(flat_sentences)
    # matrices = [sim_matrix1,sim_matrix2,sim_matrix3]
    matrices = [sim_matrix1, sim_matrix3]
    sim_matrix = get_agreegate_matrix(matrices)


    sumxx = SummaryFunction.summary_by_name(sim_matrix, sents_vec1, flat_sentences,summary_name="submodular",summarySize=85)
    return_string = "\n".join([" ".join(flat_sentences[idx_sent]) for idx_sent in sumxx])

    print return_string