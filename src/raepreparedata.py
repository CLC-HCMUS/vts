__author__ = 'HyNguyen'

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import codecs
from nn.instance import Instance
from nn.rae import RecursiveAutoencoder
from vec.word2vec import Word2VecModel

import os



def str_2_vec(clusters_path, output_path, vectormodel_path, weigth_rae_path):

    word_vector_file = vectormodel_path

    theta_opt = np.load(weigth_rae_path)

    word_vectors = Word2VecModel(word_vector_file)

    # embsize = word_vectors.embsize

    rae = RecursiveAutoencoder.build(theta_opt, 100)

    clusters_path = clusters_path

    clusters = []
    for cluster_index in range(1,201):
        if cluster_index == 178:
            clusters.append(None)
            continue
        cluster_id_path = clusters_path + 'cluster_' + str(cluster_index)
        if os.path.exists(cluster_id_path):
            files_name = os.listdir(cluster_id_path)
            cluster = {}
            for file_name in files_name:
                file_prefix = file_name.find('.body.tok.txt')
                if file_prefix > 0 :
                    file_id_str = file_name[:file_prefix]
                    file = codecs.open(cluster_id_path + '/' + file_name, encoding="UTF-8")
                    sentences = []
                    for line in file.readlines():
                        if len(line) < 50:
                            continue
                        instance = Instance.parse_from_str(line,model=word_vectors)
                        #vector_addtion
                        # sentence_vec = np.sum(instance.words_embedding,axis=1).reshape((instance.words_embedding.shape[0],1))

                        #RAE-Auto Encoder
                        root_node, rec_error = rae.forward(instance.words_embedding)
                        # print rec_error/(instance.words_embedding.shape[1] -1)

                        instance.sentence_embedding = root_node.p
                        instance_list = instance.to_list()
                        sentences.append(instance_list)
                    file.close()
                elif file_name.find(".ref1.tok.txt") != -1:
                    fi = codecs.open(cluster_id_path + '/' + file_name)
                    content = fi.read()
                    cluster["ref1.length"] = content.count(" ")
                    cluster["ref1.content"] = content
                    fi.close()
                elif file_name.find(".ref2.tok.txt") != -1:
                    fi = codecs.open(cluster_id_path + '/' + file_name)
                    content = fi.read()
                    cluster["ref2.length"] = content.count(" ")
                    cluster["ref2.content"] = content
                    fi.close()
                cluster[file_id_str] = sentences
        clusters.append(cluster)
        print 'finish ' + str(cluster_index)
    clusters = np.array(clusters)
    np.save(output_path,np.array(clusters))



import argparse

if __name__ == '__main__':

    """
    Usage sample:

    python raepreparedata.py -clusterpath data/vietnamesemds/ -outputpath data/vietnamesemds.out -vectormodel model/word2vec/100 -raeweight model/rae/100.vn.npy

    """

    parser = argparse.ArgumentParser(description='Parse process ')
    parser.add_argument('-clusterpath', required=True, type = str, help="path to 200 clusters to, e.g : data/vietnamesemds/")
    parser.add_argument('-outputpath', required=True, type = str, help="path to output to, e.g : data/vietnamesemds.out")
    parser.add_argument('-vectormodel', required=True, type = str, help="path to word2vec model, e.g : model/word2vec/100")
    parser.add_argument('-raeweight', required=True, type = str, help="path to rae model, e.g : model/rae/100.vn.npy")
    args = parser.parse_args()

    clusterpath = args.clusterpath
    outputpath = args.outputpath
    vectormodel = args.vectormodel
    raeweight = args.raeweight

    str_2_vec(clusters_path=clusterpath,
                               output_path=outputpath,
                               vectormodel_path=vectormodel,
                               weigth_rae_path=raeweight)



    # str_2_vec(clusters_path="data/vietnamesemds/",
    #                            output_path="data/vietnamesemds.out" ,
    #                            vectormodel_path="model/word2vec/100",
    #                            weigth_rae_path="model/rae/100.vn.npy")
    # summary("data/vietnamesemds.out.npy")
    # write_summary('summarytext/', file_name_npy)



