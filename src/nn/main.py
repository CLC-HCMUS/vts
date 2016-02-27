__author__ = 'HyNguyen'

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import codecs
from nn.instance import Instance
from nn.rae import RecursiveAutoencoder
from vec.word2vec import Word2VecModel
from submodular import submodular
import os

def insideMatrix(a, V):
    n = len(V)
    for i in range(0, n):
        if a == V[i]:
            return True
    return False

def read_cluster_hy_format(cluster_hy_format_file):
    clusters = np.load(cluster_hy_format_file)
    sum1 = 0
    for cluster in clusters:
        V = []
        P = []
        L = []
        if cluster !=None:
            for text_id in cluster.keys():
                p = []
                instances = cluster[text_id]
                for instance in instances:
                    #print(instance[1])   #vector (100,1)
                    instance.append(False)
                    p.append(instance[1])
                    V.append(instance[1])
                    L.append(len(instance[0].split()))
                P.append(p)
            alpha = 0.7
            galma = 0.3
            numberofWord = 200
            summarize = sorted(submodular.maximizeF(V, P, alpha, galma, L, numberofWord))
            print (summarize)
            i = 0
            k = 0
            for text_id in cluster.keys():
                list_instance = cluster[text_id]
                for instance in list_instance:
                    if insideMatrix(k, summarize) == True:
                        instance[2] = True
                        k = k + 1
                    else:
                        k = k + 1

        np.save(cluster_hy_format_file,clusters)

def generate_cluster_hy_format(direct_folder_clusters, cluster_hy_format_file):

    word_vector_file = '../vec/vn_word2vec_model_27/100'
    theta_opt = np.load('hynguyen_theta_opt.npy')

    word_vectors = Word2VecModel(word_vector_file)
    embsize = word_vectors.embsize

    rae = RecursiveAutoencoder.build(theta_opt, 100)



    direct_folder_clusters = direct_folder_clusters

    clusters = []
    for cluster_index in range(1,11):
        if cluster_index == 178:
            clusters.append(None)
            continue
        directs_folder_cluster = direct_folder_clusters + 'cluster_' + str(cluster_index)
        if os.path.exists(directs_folder_cluster):
            files_name = os.listdir(directs_folder_cluster)
            cluster = {}
            for file_name in files_name:
                file_prefix = file_name.find('.body.tok.txt')
                if file_prefix > 0 :
                    file_id_str = file_name[:file_prefix]
                    file = codecs.open(directs_folder_cluster + '/' + file_name, encoding="UTF-8")
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
                cluster[file_id_str] = sentences
        clusters.append(cluster)
        print 'finish ' + str(cluster_index)
    clusters = np.array(clusters)
    np.save(cluster_hy_format_file,np.array(clusters))

def write_system_by_cluster_hy_format(direct_folder_systems, cluster_hy_format_file):
    systems_direct = direct_folder_systems
    clusters = np.load(cluster_hy_format_file)
    idx = 1
    for cluster in clusters:
        if cluster == None:
            idx+=1
            continue

        file_direct = systems_direct + 'cluster_' + str(idx) + ".txt"
        file_out = open(file_direct,'w')
        for text_id in cluster.keys():
            instances = cluster[text_id]
            for instance in instances:
                if(instance[2] == True):
                    file_out.write(instance[0].encode('utf8'))
        idx+=1
        file_out.close()

if __name__ == '__main__':
    file_name = 'file_cluster_hy_format'
    file_name_npy = file_name + '.npy'
    generate_cluster_hy_format('../clusters/', file_name)
    read_cluster_hy_format(file_name_npy)
    write_system_by_cluster_hy_format('../systems/',file_name_npy)



