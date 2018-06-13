__author__ = 'HyNguyen'

import numpy as np
import logging
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim

def get_vector_duc04():
    d2v_model = gensim.models.Doc2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_en/d2v-dbow/d2v-dbow.model")

    with open("duc04.encode.pickle", mode="rb") as f:
        clusters = pickle.load(f)

    origin_tags=[]
    for cluster_id,cluster in clusters:
        for document_id, document in cluster:
            for sentence_id, sentence in document:
                origin_tags.append("{0}_{1}_{2}".format(cluster_id,document_id,sentence_id))

    duc04_vec = {}
    for tag in origin_tags:
        if tag in d2v_model.docvecs:
            duc04_vec[tag] = d2v_model.docvecs[tag]
        else:
            print("hyhy attention")

    with open("duc04vector/pvdbow.pickle", mode="wb") as f:
        pickle.dump(duc04_vec,f)



def get_vector_mds():
    d2v_model = gensim.models.Doc2Vec.load("/Users/HyNguyen/Documents/w2v_d2v_vn/d2v_dbow/d2v_dbow.model")

    with open("vietnamesemds.encode.pickle", mode="rb") as f:
        clusters = pickle.load(f)

    origin_tags=[]
    for xxx in clusters:
        if xxx is None:
            continue
        else:
            cluster_id,cluster = xxx
        for document_id, document in cluster:
            for sentence_id, sentence in document:
                origin_tags.append("{0}_{1}_{2}".format(cluster_id,document_id,sentence_id))

    vietnamesemds_vec = {}
    for tag in origin_tags:
        if tag in d2v_model.docvecs:
            vietnamesemds_vec[tag] = d2v_model.docvecs[tag]
        else:
            print("hyhy attention")

    with open("mdsvector/pvdbow.pickle", mode="wb") as f:
        pickle.dump(vietnamesemds_vec,f)

if __name__ == "__main__":
    get_vector_duc04()