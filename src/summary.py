__author__ = 'HyNguyen'

from submodular import submodular
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
    for cluster in clusters:
        V = []
        P = []
        L = []
        if cluster !=None:
            # cluster["ref1.length"] la do dai cua ban tom tat thu nhat
            # cluster["ref2.length"] la do dai cua ban tom tat thu hai
            # cluster["text_id"]
            for text_id in cluster.keys():
                p = []
                instances = cluster[text_id]
                if isinstance(instances, int):
                    # dong nay la de loai di truong hop cluster["ref1.length"] cluster["ref2.length"]
                    continue
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
                if isinstance(list_instance, int):
                    continue
                for instance in list_instance:
                    if insideMatrix(k, summarize) == True:
                        instance[2] = True
                        k = k + 1
                    else:
                        k = k + 1

        np.save(cluster_format_npy,clusters)

def write_summary(direct_folder_systems, cluster_hy_format_file):
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
            if isinstance(instances,int):
                continue
            for instance in instances:
                if(instance[2] == True):
                    file_out.write(instance[0].encode('utf8'))
        idx+=1
        file_out.close()

import argparse

if __name__ == "__main__":
    """
    Usage sample:

    python summary.py -inputpath data/vietnamesemds.out.npy -outputpath summarytext/

    """

    parser = argparse.ArgumentParser(description='Parse process ')
    parser.add_argument('-inputpath', required=True, type = str, help="path to 200 cluster np format, e.g : data/vietnamesemds.out.npy")
    parser.add_argument('-outputpath', required=True, type = str, help="output path to generate summary, e.g : summarytext/")
    args = parser.parse_args()

    inputpath = args.inputpath
    outputpath = args.outputpath

    summary(inputpath)
    write_summary(outputpath, inputpath)
