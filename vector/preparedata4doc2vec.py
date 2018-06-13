__author__ = 'HyNguyen'
import xml.etree.ElementTree as ET
import os
import nltk
import numpy as np
import pickle
import codecs

def prepareducdata_4_doc2vec():
    doc_path = "../data/duc04"
    fo_name = "duc04_all_sent.txt"

    fo = open(fo_name,mode="w")

    clusters = []
    for cluster_id in os.listdir(doc_path):
        if cluster_id[0] == ".":
            continue
        cluster = []
        list_file = os.listdir(doc_path + "/" + cluster_id)
        for file_name in list_file:
            if file_name[0] == ".":
                continue
            tree = ET.parse(doc_path + "/" + cluster_id + "/" + file_name)
            root = tree.getroot()
            text_tag = root._children[3]
            if text_tag.tag == "TEXT":
                text = text_tag.text.replace("\n", " ")
            sents = nltk.sent_tokenize(text.strip())
            document = []
            for sent_i,sent in enumerate(sents):
                words = nltk.word_tokenize(sent.strip())
                if len(words) < 8:
                    continue
                document.append( (sent_i ,words))
            cluster.append((file_name,document))
        clusters.append((cluster_id,cluster))

    for cluster in clusters:
        cluster_id, cluster_documents = cluster
        cluster_string = ""
        for cluster_document in cluster_documents:
            document_id, document = cluster_document
            doc_string = ""
            for sentence in document:
                sent_id, sent = sentence
                doc_string += " ".join(sent)
                sent_string = "{0}_{1}_{2} {3}\n".format(cluster_id,document_id,sent_id," ".join(sent))
                fo.write(sent_string)
            cluster_string += doc_string
            doc_string = "{0}_{1} {2}\n".format(cluster_id,document_id,doc_string)
            fo.write(doc_string)
        cluster_string = "{0} {1}\n".format(cluster_id,cluster_string)
        fo.write(cluster_string)
    fo.close()

    with open("duc04.encode.pickle", mode="wb") as f:
        pickle.dump(clusters,f)

def load_mds_cluster(cluster_path):
    if os.path.exists(cluster_path):
        files_name = os.listdir(cluster_path)
        doc = []
        for file_name in files_name:
            if file_name.find('.body.tok.txt') != -1 or file_name.find(".ref1.tok") != -1 or file_name.find(".ref2.tok") != -1 :
                with codecs.open(cluster_path + '/' + file_name, mode="r", encoding="utf8") as f:
                    sents = f.readlines()
                    sents_wordlist = []
                    for sent_i,sent in enumerate(sents):
                        words = sent.split()
                        if len(words) > 10:
                            sents_wordlist.append((sent_i,words))
                    doc.append((file_name,sents_wordlist))
        return doc
    else:
        return None

import codecs
def preparevietnamesemdsdata_4_doc2vec():

    fo_name = "vietnamesemds.tagged.txt"

    fo = codecs.open(fo_name, mode="w", encoding="utf8")

    vietnamesemds = "../data/VietnameseMDS-grouped/clusters"
    clusters = []
    clusters.append(None)
    for cluster_id in range(1,201):
        if cluster_id == 178:
            clusters.append(None)
            continue
        cluster = load_mds_cluster(vietnamesemds+"/cluster_"+str(cluster_id))
        clusters.append(("cluster"+str(cluster_id),cluster))
        print "Processed cluster: ", cluster_id

    for cluster in clusters:
        if cluster is None:
            continue
        cluster_id, cluster_documents = cluster
        cluster_string = ""
        for cluster_document in cluster_documents:
            document_id, document = cluster_document
            doc_string = ""
            for sentence in document:
                sent_id, sent = sentence
                doc_string += u" ".join(sent)
                sent_string = u"{0}_{1}_{2} {3}\n".format(cluster_id,document_id,sent_id," ".join(sent))
                fo.write(sent_string)
            cluster_string += doc_string
            doc_string = u"{0}_{1} {2}\n".format(cluster_id,document_id,doc_string)
            fo.write(doc_string)
        cluster_string = u"{0} {1}\n".format(cluster_id,cluster_string)
        fo.write(cluster_string)
    fo.close()

    with open("vietnamesemds.encode.pickle", mode="wb") as f:
        pickle.dump(clusters,f)

if __name__ == "__main__":
    # prepareducdata_4_doc2vec()
    preparevietnamesemdsdata_4_doc2vec()