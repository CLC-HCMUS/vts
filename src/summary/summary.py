__author__ = 'MichaelLe'

from mmr import mmrelevance
from submodular import submodular
import kmean_sum



def do_summarize(V,n, P, L, alpha, galma, numberofWord, mode_str):

    modeList = {"sub_cosine":0, "sub_euclid":1,"mmr_cosine":2,"mmr_euclid":3,"kmean_simple":4}
    mode = modeList[mode_str]
    k = 2
    if (mode == 0) or mode == 1: ## cosine distance
        return sorted(submodular.SubmodularFunc(V,n, P, L, alpha, galma, numberofWord, mode))
    elif mode == 2 or mode == 3:
        return sorted(mmrelevance.summaryMMR11(V, L, galma, numberofWord, mode-2))
    elif mode == 4:
        return sorted(kmean_sum.kmean_summary(V,L,numberofWord))


