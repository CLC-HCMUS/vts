__author__ = 'HyNguyen'

from submodular import submodular
from mmr import mmrelevance as mmr
from summary import summary as smr

import numpy as np

def insideMatrix(a, V):
    n = len(V)
    for i in range(0, n):
        if a == V[i]:
            return True
    return False


def summary(cluster_format_npy):
    clusters = np.load(cluster_format_npy)
    sum1 = 0
    counter = 0
    for cluster in clusters:
        counter +=1
        V = []
        P = []
        L = np.array([])
        k = 0
        if cluster !=None:
            # cluster["ref1.length"] la do dai cua ban tom tat thu nhat
            # cluster["ref2.length"] la do dai cua ban tom tat thu hai
            # cluster["text_id"]
            for text_id in cluster.keys():
                p = []
                instances = cluster[text_id]
                if isinstance(instances, list) == False:
                    # dong nay la de loai di truong hop cluster["ref1.length"] cluster["ref2.length"]
                    continue
                for instance in instances:
                    #print(instance[1])   #vector (100,1)
                    instance.append(False)
                    p.append(k)
                    k = k + 1
                    V.append(instance[1])
                    L = np.append(L,len(instance[0].split()))
                P.append(p)
            alpha = 0.7
            galma = 0.3
            n = len(V)
            lamda = 0.3
            numberofWord = 200 #find_group(cluster["ref1.length"],cluster["ref1.length"])
            mode = 0
            ##########
            # mode = 0: submodular + cosine
            # mode = 1: submodular + euclid
            # mode = 2: mmr + cosine
            # mode = 3: mmr + euclid
            # ***** note: galma is the lamda in mmr
            ##########
            summarize = smr.do_summarize(V,n, P, L, alpha, galma, numberofWord, mode)
            print (summarize)
            i = 0
            k = 0
            for text_id in cluster.keys():
                list_instance = cluster[text_id]
                if isinstance(list_instance, list) == False:
                    continue
                for instance in list_instance:
                    if insideMatrix(k, summarize) == True:
                        instance[2] = True
                        k = k + 1
                    else:
                        k = k + 1

        np.save(cluster_format_npy,clusters)

#summary('data/vietnamesemds.out.npy')

def select_sentence(sentences, max_length):

    count = 0
    sentences_result = []

    for sentence in sentences:
        sentences_result.append(sentence)
        count += sentence.count(' ')
        if count > max_length:
            break

    return sentences_result

def write_summary(direct_folder_system, direct_folder_model ,cluster_hy_format_file):
    systems_direct = direct_folder_system
    clusters = np.load(cluster_hy_format_file)
    idx = 1
    for cluster in clusters:
        if cluster == None:
            idx+=1
            continue
        count = 0
        sentences = []
        for text_id in cluster.keys():
            instances = cluster[text_id]
            if isinstance(instances,list ) == False:
                continue
            for instance in instances:
                if(instance[2] == True):
                    sentences.append(instance[0].encode('utf8'))

        group1_num = find_group(0, cluster["ref1.length"])
        group2_num = find_group(0, cluster["ref2.length"])

        sentences_1 = select_sentence(sentences, group1_num)
        sentences_2 = select_sentence(sentences, group2_num)

        file_direct = systems_direct + "/" + str(group1_num)  + '/cluster_' + str(idx) + ".ref1.txt"
        file_out = open(file_direct,'w')
        file_out.writelines(sentences_1)
        file_out.close()

        file_direct = direct_folder_model + "/" + str(group1_num)  + '/cluster_' + str(idx) + ".ref1.txt"
        file_out = open(file_direct,'w')
        file_out.write(cluster["ref1.content"])
        file_out.close()

        file_direct = systems_direct + "/" + str(group2_num)  + '/cluster_' + str(idx) + ".ref2.txt"
        file_out = open(file_direct,'w')
        file_out.writelines(sentences_2)
        file_out.close()

        file_direct = direct_folder_model + "/" + str(group2_num)  + '/cluster_' + str(idx) + ".ref2.txt"
        file_out = open(file_direct,'w')
        file_out.write(cluster["ref2.content"])
        file_out.close()
        idx+=1

def staticstic(cluster_hy_format_file):
    clusters = np.load(cluster_hy_format_file)
    print clusters.shape
    idx = 1
    histogram = [0] * 600
    for cluster in clusters:
        try:
            print idx, cluster["ref2.length"], cluster["ref1.length"]
            histogram[cluster["ref2.length"]] +=1
            histogram[cluster["ref1.length"]] +=1
            idx+=1
        except:
            print idx

    return histogram

import argparse
import matplotlib.pyplot as plt


def find_group(ref1length, ref2length):

    maxlength = max(ref1length, ref2length)
    if maxlength >=40 and maxlength <= 120:
        return 80
    elif maxlength>120 and maxlength <=200:
        return 160
    elif maxlength>200 and maxlength <=280:
        return 240
    else:
        return 320

if __name__ == "__main__":
    """
    Usage sample:
    python summary.py -inputpath data/vietnamesemds.out.npy -outputsystem summary_system -outputmodel summary_model
    """

    parser = argparse.ArgumentParser(description='Parse process ')

    parser.add_argument('-inputpath', required=True, type = str, help="path to 200 cluster np format, e.g : data/vietnamesemds.out.npy")
    parser.add_argument('-outputsystem', required=True, type = str, help="output path to generate summary, e.g : summarytext/")
    parser.add_argument('-outputmodel', required=True, type = str, help="output path to generate summary, e.g : summarytext/")

    args = parser.parse_args()

    inputpath = args.inputpath
    outputsystem = args.outputsystem
    outputmodel = args.outputmodel


    summary(inputpath)
    write_summary(outputsystem, outputmodel, inputpath)

    # his = staticstic("data/vietnamesemds.out.npy")
    # print(np.sum(his))
    # plt.plot(range(len(his)),his,'go',label = 'count')
    # plt.axis([0,400,0,10])
    # plt.xticks(np.arange(0, 400, 20))
    # plt.yticks(np.arange(0, 10, 2))
    # plt.legend()
    # plt.show()

    # 40 120 200 280 ...
    #
    # 80 160 240 320