__author__ = 'MichaelLe'


from vector import *
import numpy as np

def build_sim_matrix(senList, mode):
    ########################
    # senList: list of sentence to build sim_matrix
    # ****** note: each element in senList must be np.array 1-d or equivalent
    ########################
    # 1. Create the similarity matrix for each pair of sentence in document
    # ***** note: the last row of matrix is the sum of similariry
    # between a specific sentence and the whole document (include this sentence)
    ########################
    numSen = np.size(senList,0)
    simM = np.ones((numSen + 1, numSen))
    for i in range(numSen):
        for j in range(i,numSen,1):
            simM[i,j] = similarity(senList[i],senList[j], mode)
            simM[j,i] = simM[i,j]
    for i in range(numSen):
        simM[numSen,i] = np.sum(simM[:numSen,i])
    return simM

def get_sim_for_set(sim_matrix, sen, set_sen):
    #################################
    #sim_matrix: matrix of simmilarity of all pairs of sentence in documents
    #sen: order of sentence in document
    #set_sen: the set of order of sentence
    #####################################
    # 1. Calculate the similarity of a specific sentence and a set of sentence
    #  by linear combination
    ##################################
    sum_cov = 0
    for s in set_sen:
        sum_cov = sum_cov + sim_matrix[sen,s]
    return sum_cov


def scoreMMR1(sim_matrix, sen, n, summary, lamda):
    ########################################################################
    #sim_matrix: matrix of simmilarity of all pairs of sentence in documents
    #sen: order of sentence in document
    #n: the number of sentence in document
    #summary: list of sentence is selected to put into summary
    #lamda: trade-off coefficent
    ########################################################################
    # Calculate the MMR score (1 version):
    #   In this version, the similarity of one sentence and a set
    #   is only the linear combination of similarity of sentence with each sentence in this set.
    ########################################################################
    sim1 = sim_matrix[n,sen]
    sim2 = get_sim_for_set(sim_matrix,sen,summary)
    return lamda*sim1 - (1-lamda)*sim2


def stopCondition(len_sen_mat, summary, max_word):
    ################################################################
    # len_sen_mat: matrix of length of all sentence in document
    # summary: the order of sentence in summary
    # max_word: the maximum of number of word for a summary
    # **** note: len_sen_mat must be a 1-d np.array or equivalent
    #            so that it can be access element through list
    ################################################################
    # 1. return 1 if the length of summary > max_word or 0 otherwise
    ################################################################
    length_summary = np.sum(len_sen_mat[summary])
    if length_summary > max_word:
        return 1
    else:
        return 0


def summaryMMR11(document, len_sen_mat,lamda, max_word, mode):
    ################################################################
    # len_sen_mat: matrix of length of all sentence in document
    # document: the set of all sentence
    # max_word: the maximum of number of word for a summary
    # **** note: len_sen_mat must be a 1-d np.array or equivalent
    #            so that it can be access element through list
    ################################################################
    # return the set of sentence in summary
    ################################################################

    sim_matrix = build_sim_matrix(document, mode)
    n = len(document)
    summary = [ ]
    while (stopCondition(len_sen_mat,summary,max_word) == 0):
        score_matrix = np.zeros(n)
        for i in range(n):
            if (i in summary) == False:
                score_matrix[i] = scoreMMR1(sim_matrix,i,n,summary, lamda)
        selected_sen = np.argmax(score_matrix)
        summary.append(selected_sen)
    return summary
