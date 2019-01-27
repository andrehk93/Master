import os
import re
import operator
import torch
import numpy as np
from utils.text import pattern_repl as pr

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
stop = set(stopwords.words("english"))


class Dictionary(object):
    def __init__(self):
        self.word2idx = {"$pad": 0}
        self.idx2word = ["$pad"]
        self.oov_index = 1

    def add_word(self, word):
        # NEED TO SAVE MAX INDEX FOR "UNKNOWN" WORDS, AND 0 INDEX FOR PADDING:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.oov_index += 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, max_size, root, stopwords=True):
        self.root = root
        self.stopwords = stopwords
        self.dictionary = Dictionary()
        self.tokenize(max_size)

    # Tokenizing files to create the dictionaries:
    def tokenize(self, max_size):

        print("Reading from " + self.root)
        files = return_all_data_from_file(self.root)

        lenghts = 0.0
        i = 0.0

        word_counts = {}
        print("Creating Dictionary from File...")
        for f in range(len(files)):
            if f % 1000 == 0:
                print("Reading [" + str(f) + "/" + str(len(files)) + "] files...")
            file = files[f]
            for line in file.split("\n"):
                if len(line) <= 1:
                    continue
                words = parse(line, self.stopwords)
                lenghts += len(words)
                i += 1
                for word in words:
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1

        print("Average length of sentences: ", (lenghts/i))
        input("OK")
        # Keeping the MAX_LENGTH most words:
        sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1))
        sorted_counts.reverse()
        for (word, freq) in sorted_counts[0:max_size]:
            self.dictionary.add_word(word)


# Parsing a sentence (removing punctuation --> stopwords --> stemming):
def parse(sentence, stopwords):
    """
    # Replace patterns
    replacer = pr.RegexpReplacer()
    result = replacer.replace(sentence)
    """
    # Remove unnecessary punctuation
    result = re.sub("[^\w\s]", "", sentence)
    """
    # Remove Stopwords (optional)
    if stopwords:
        print("Removing stopwords...")
        result = [porter.stem(i.lower()) for i in wordpunct_tokenize(result)
                  if i.lower() not in stop]
    else:
        result = [porter.stem(i.lower()) for i in result.split(" ")
                  if i.lower() not in stop]
    """
    res = []
    for word in result.split(" "):
        if len(word) > 0:
            res.append(word.lower())
    return res


# Returns all data from a path
def return_all_data_from_file(path):
    all_words = []
    print("Loading from path: ", path)
    for root, dirs, files in os.walk(path):
        if len(files) >= 10:
            for file in files:
                if ".DS_Store" not in file:
                    new_file = open(os.path.join(root, file), "r")
                    new_file = new_file.read()
                    all_words.append(new_file)
    return all_words


def create_word_vectors(words, sen_len, corpus):
    text = []
    sentence = torch.LongTensor(np.zeros(sen_len))
    count = 0
    for word in words:
        word = word.lower()
        # In Vocabulary:
        if word in corpus.dictionary.word2idx:

            if count > sen_len - 1:
                text.append(sentence)
                sentence = torch.LongTensor(np.zeros(sen_len))
                count = 0
            sentence[count] = corpus.dictionary.word2idx[word]
            count += 1

        # Out of Vocabulary, but still "counts":
        else:
            if count > sen_len - 1:
                text.append(sentence)
                sentence = torch.LongTensor(np.zeros(sen_len))
                count = 0
            sentence[count] = corpus.dictionary.oov_index
            count += 1
    if len(text) == 0:
        text.append(sentence)

    return text



