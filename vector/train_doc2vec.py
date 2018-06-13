__author__ = 'HyNguyen'

import gensim.models
# setup logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import numpy as np

class MyTaggedLineDocument(object):

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    tokens = utils.to_unicode(line).split()
                    words = tokens[1:]
                    tag = tokens[0]
                    yield TaggedDocument(words, [tag])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', required=True, type=str)
    parser.add_argument('-fo', required=True, type=str)
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('-worker', type=int, default=4)
    parser.add_argument('-mincount', type=int, default=5)
    parser.add_argument('-size', type=int, default=100)

    args = parser.parse_args()
    fi = args.fi
    fo = args.fo
    modelname = args.model
    woker = args.worker
    min_count = args.mincount
    size = args.size

    dm =1
    if modelname == "dm":
        dm = 1
    elif modelname == "dbow":
        dm = 0

    # fi, fo đường dẫn của file input (trainning file), đường dẫn save file model
    model = gensim.models.Doc2Vec(workers=woker,min_count=min_count,size=size,dm=dm)
    sentences = MyTaggedLineDocument(fi)
    model.build_vocab(sentences)
    model.train(sentences)
    model.save(fo)
