import os, re
import string
import operator
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np
import random
from utils import pattern_repl as pr

import nltk
nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
stop = set(stopwords.words("english"))


# Root file location:
root = "./data/Reuters"



class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, max_size):
        self.dictionary = Dictionary()
        self.tokenize(max_size)

    # Tokenizing files to create the dictionaries:
    def tokenize(self, max_size):
        path = "./data/Reuters/raw/"

        print("Reading...")
        files = return_all_data_from_file(path)

        word_counts = {}
        print("Creating Dictionary from File...")
        tokens = 0
        for f in range(len(files)):
            if (f % 1000 == 0):
                print("Reading [" + str(f) + "/" + str(len(files)) + "] files...")
            file = files[f]
            tokens = 0
            for line in file.split("\n"):
                if (len(line) <= 1):
                    continue
                words = parse(line)
                tokens += len(words)
                for word in words:
                    if (word not in word_counts):
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1


        # Keeping the MAX_LENGTH most words:
        sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1))
        sorted_counts.reverse()
        for (word, freq) in sorted_counts[0:max_size]:
            self.dictionary.add_word(word)


# Parsing a sentence (removing punctuation --> stopwords --> stemming):
def parse(sentence):
    ### REPLACE PATTERNS ###
    replacer = pr.RegexpReplacer()
    result = replacer.replace(sentence)

    ### REMOVE PUNCTUATION ###
    result = re.sub("[^\w\s]", " ", result)
    ### REMOVE STOPWORDS AND STEMMING ###
    result = [porter.stem(i.lower()) for i in wordpunct_tokenize(result)
    if i.lower() not in stop]
    return result


### FOR MAKING DICTIONARY ###
def return_all_data_from_file(path):
    all_words = []
    print("Loading from path: ", path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if (".DS_Store" not in file):
                new_file = []
                file = open(os.path.join(root, file), "r")
                file = file.read()
                all_words.append(file)
    return all_words



def create_word_vectors(words, sen_len, corpus):
    text = []
    sentence = torch.FloatTensor(np.zeros(sen_len))
    count = 0
    for word in words:
        if word in corpus.dictionary.word2idx:
            if count > sen_len - 1:
                text.append(sentence)
                sentence = torch.FloatTensor(np.zeros(sen_len))
                count = 0
            sentence[count] = corpus.dictionary.word2idx[word]
            count += 1
    return text



