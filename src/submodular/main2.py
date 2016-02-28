__author__ = 'HyNguyen'

from numpy import *
import submodular

def insideMatrix(a, V):
    n = len(V)
    for i in range(0, n):
        if a == V[i]:
            return True
    return False


def read_cluster_hy_format(cluster_hy_format_file):
    clusters = load(cluster_hy_format_file)
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

        save('file_cluster_hy_format_2411_result.npy',clusters)
read_cluster_hy_format('file_cluster_hy_format_2411.npy')


