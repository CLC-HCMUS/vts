__author__ = 'MichaelLe'

import numpy as np
import math
from sklearn.cluster import KMeans


def kmean_summary(V,len_sen_mat, max_word):
    '''
    parameter:
    ---------
        V: List of vector sentence representation
        len_sen_mat: matrix of length of all sentences in V
        max_word: max of word in summary
    ----------
    return:
        list of number of sentences which are selected for summary
    '''

    ## because Kmean is only applied to 2-d array
    ##  --> V have 3-d (no of sentences, dim of one sentence, 1) -- pratice
    V_numpy = np.array(V).reshape((len(V),V[0].shape[0]))

    avg_len_sen = np.average(len_sen_mat)

    numcluster = int(math.ceil(max_word/avg_len_sen))

    cluster_re = KMeans(n_clusters = numcluster,n_init= 100).fit_transform(V_numpy)

    summary = np.argmin(cluster_re,axis = 0)

    return summary
