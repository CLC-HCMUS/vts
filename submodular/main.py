__author__ = 'HyNguyen'

from stemming.porter2 import stem
import pickle


if __name__ == "__main__":
    # with open("emotion_words_negative.txt",mode="r") as f:
    #     doc = f.read()
    #     neg_words = doc.split()
    #     print(len(neg_words))
    #     stemmed_neg_words = [stem(word) for word in neg_words]
    #
    # with open("emotion_words_positive.txt",mode="r") as f:
    #     doc = f.read()
    #     pos_words = doc.split()
    #     print(len(pos_words))
    #     stemmed_pos_words = [stem(word) for word in pos_words]
    #
    # with open("emotion_stemmed_words_pos_neg.pickle", mode="w") as f:
    #     pickle.dump((stemmed_neg_words,stemmed_pos_words),f)
    with open("emotion_stemmed_words_pos_neg.pickle", mode="r") as f:
        stemmed_neg_words,stemmed_pos_words = pickle.load(f)

    print("ttdt")





