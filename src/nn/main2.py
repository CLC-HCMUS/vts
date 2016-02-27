__author__ = 'HyNguyen'

from numpy import *

def read_cluster_hy_format(cluster_hy_format_file):
    clusters = load(cluster_hy_format_file)
    for cluster in clusters:
        if cluster == None:
            continue
        for text_id in cluster.keys():
            instances = cluster[text_id]
            for instance in instances:
                print instance[0]   #raw_sentence
                print instance[1]   #vector (100,1)

if __name__ == '__main__':
    read_cluster_hy_format('file_cluster_hy_format.npy')