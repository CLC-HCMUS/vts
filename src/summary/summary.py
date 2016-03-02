__author__ = 'MichaelLe'

from mmr import mmrelevance
from submodular import submodular

def do_summarize(V,n, P, L, alpha, galma, numberofWord, mode):
    k = 2
    if (mode == 0) or mode == 1: ## cosine distance
        return sorted(submodular.SubmodularFunc(V,n, P, L, alpha, galma, numberofWord, mode))
    elif mode == 2 or mode == 3:
        return sorted(mmrelevance.summaryMMR11(V,L,galma,numberofWord, mode-2))


