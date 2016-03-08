__author__ = 'MichaelLe'

from mmr import mmrelevance
from submodular import submodular
import kmean_sum



def do_summarize(V,n, P, L, alpha, galma, numberofWord, mode_str):

    modeList = {"sub_cosine":0, "sub_euclid":1,"mmr_cosine":2,"mmr_euclid":3,"kmean_simple":4,
                "mmr_kmean_cosine":5,"mmr_kmean_euclid":6,"mmr_pagerank_cosine":7,
                "mmr_pagerank_euclid":8}
    mode = modeList[mode_str]
    k = 2
    if (mode == 0) or mode == 1: ## cosine distance
        return sorted(submodular.SubmodularFunc(V,n, P, L, alpha, galma, numberofWord, mode))
    elif mode == 2 or mode == 3:
        return sorted(mmrelevance.summaryMMR11(V, L, galma, numberofWord, mode-2))
    elif mode == 4:
        return sorted(kmean_sum.kmean_summary(V,L,numberofWord))
    elif mode == 5 or mode == 6:
        return sorted(mmrelevance.summaryMMR_centroid_kmean(V,L,galma,numberofWord,mode-5))
    elif mode == 7 or mode == 8:
        return sorted(mmrelevance.mmr_pagerank(V, L, alpha, numberofWord, mode - 7))


