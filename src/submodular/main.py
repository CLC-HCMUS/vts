__author__ = 'MichaelLe'

import submodular
import loadFile
from numpy import *
from numpy import *
import numpy as np
from tempfile import TemporaryFile

clusters = load('cluster_my_format.npy')
sum = 0
for cluster in clusters:
        if cluster != None:
            # cluster la 1 dictionary
            # - 'key'    : id cua van ban
            # - 'value'  : list[instance]
            V = []
            P = []
            L = []
            for text_id in cluster.keys():
                list_instance = cluster[text_id]  # lay value cua key 'text_id'
                p = []
                for instance in list_instance:
                    instance.append(False)
                    p.append(instance[0])
                    V.append(instance[0])
                    L.append(len(instance[1].split()))
                P.append(p)
            sum = sum + len(V)

print (sum)


def insideMatrix(a, V):
    n = len(V)
    for i in range(0, n):
        if a == V[i]:
            return True
    return False


def suma(clusters, alpha, galma, numberofWord):
    for cluster in clusters:
        if cluster != None:
            # cluster la 1 dictionary
            # - 'key'    : id cua van ban
            # - 'value'  : list[instance]
            V = []
            P = []
            L = []
            for text_id in cluster.keys():
                list_instance = cluster[text_id]  # lay value cua key 'text_id'
                p = []
                for instance in list_instance:
                    instance.append(False)
                    p.append(instance[0])
                    V.append(instance[0])
                    L.append(len(instance[1].split()))
                P.append(p)

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
    return clusters


# alpha = 0.4
# galma = 0.6
# numberofWord = 200
#
# global str
# for i in arange(0,1):
#      galma = 0.4
#      for j in arange(0,3):
#          str1 = str(int(alpha*10))+'_'
#          str2 = str(int(galma*10))+'_'
#          str3 = str(numberofWord)
#
#          strname = 'result' + str1 + str2 +str3
#
#          clusters = load('cluster_my_format.npy')
#          su = suma(clusters,alpha,galma,numberofWord)
#          np.save(strname,su)
#          galma = galma + 0.2
#      alpha = alpha + 0.2
#
# alpha = 0.8
# for i in arange(0,1):
#      galma = 0.2
#      for j in arange(0,4):
#          str1 = str(int(alpha*10))+'_'
#          str2 = str(int(galma*10))+'_'
#          str3 = str(numberofWord)
#
#          strname = 'result' + str1 + str2 +str3
#
#          clusters = load('cluster_my_format.npy')
#          su = suma(clusters,alpha,galma,numberofWord)
#          np.save(strname,su)
#          galma = galma + 0.2
#      alpha = alpha  + 0.2

