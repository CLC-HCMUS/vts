__author__ = 'HyNguyen'

import nltk
import os

if __name__ == "__main__":
    dir_path = "model_duc"
    dirout_path = "model_duc2"
    for file in os.listdir(dir_path):
        if file[0] == ".":
            continue
        with open(dir_path + "/" + file, mode="r") as f:
            doc = f.read()
        lines = nltk.sent_tokenize(doc)
        sents = []
        for line in lines:
            sents.append(nltk.word_tokenize(line))
        with open(dirout_path + "/" + file, mode="w") as f:
            string = "\n".join([" ".join(words) for words in sents])
            f.write(string)
