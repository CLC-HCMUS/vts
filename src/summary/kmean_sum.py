__author__ = 'MichaelLe'

import numpy as np
import math
from sklearn.cluster import KMeans


def kmean_summary(V,len_word_max, max_word):
    '''

    '''

    V_numpy = np.array(V).reshape((len(V),V[0].shape[0]))
    avg_len_sen = np.average(len_word_max)

    numcluster = int(math.ceil(max_word/avg_len_sen))

    cluster_re = KMeans(n_clusters = numcluster,n_init= 100).fit_transform(V_numpy)

    summary = np.argmin(cluster_re,axis = 0)

    return summary
