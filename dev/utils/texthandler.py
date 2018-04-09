import os, re
import string
import operator
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np
import random
import pattern_repl as pr

import nltk
nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
stop = set(stopwords.words("english"))


# Root file location:
polarity_filename = "data/polarity/"



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
        path_pos = [polarity_filename + "pos/"]
        path_neg = [polarity_filename + "neg/"]

        print("Reading...")
        pos_files = return_all_data_from_file(path_pos)
        neg_files = return_all_data_from_file(path_neg)
        files = [pos_files, neg_files]

        word_counts = {}
        print("Creating Dictionary from File...")
        tokens = 0
        for f in range(len(files)):
            print("Reading [" + str(f + 1) + "/" + str(len(files)) + "] directories...")
            file_list = files[f]
            for file in file_list:
                tokens = 0
                for line in file.split("."):
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


# Dataset used for testing out the network after training:
class PredictionDataset(Dataset):
    def __init__(self, corpus, sen_len):
        self.data = []
        self.label = []
        self.corpus = corpus
        self.sen_len = sen_len
        strInput = input("Write a short review: \n")
        strLabel = input("What is the label?: ")
        while (len(strInput) > 0):
            self.data.append(strInput)
            if (len(strLabel) > 0):
                self.label.append(int(strLabel))
            else:
                while (len(strLabel) == 0):
                    strLabel = input("What is the label? :")
                self.label.append(int(strLabel))
            strInput = input("Write a short review: \n")
            strLabel = input("What is the label? :")

    def __getitem__(self, index):
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        words = parse(self.data[index])
        for word in words:
            if word in self.corpus.dictionary.word2idx:
                if count > self.sen_len - 1:
                    break
                txt[count] = self.corpus.dictionary.word2idx[word]
                count += 1

        print("Wordvector: ", txt)
        label = torch.FloatTensor([self.label[index]])
        return txt, label

    def __len__(self):
        return len(self.data)

### FOR MAKING DICTIONARY ###
def return_all_data_from_file(paths):
    all_words = []
    for path in paths:
        print("Loading from path: ", path)
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if (".txt" in file):
                    new_file = []
                    file = open(path + file)
                    file = file.read()
                    all_words.append(file)
    return all_words


### CHECKING IF A FOLDER IS EMPTY ###
def folder_empty(path):
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if (".txt" in file):
                count +=1 

    if (count < 2):
        return True
    else:
        return False

def write_data(dataset, train):
    filename = polarity_filename + "processed/"
    if not os.path.exists(filename):
        os.makedirs(filename)
    if (not folder_empty(filename)):
        return
    if (train):
        filename += "training.txt"
    else:
        filename += "validation.txt"

    file = open(filename, "w")
    filenames = dataset[0]
    labels = dataset[1]
    for i in range(len(filenames)):
        f = filenames[i]
        label = labels[i]
        file.write(f + ":" + str(label) + "\n")
    file.close()
    print("Data written succesfully to file...")

def load_from_file(train, sen_len, corpus):
    filename = polarity_filename + "processed/"
    if (train):
        filename += "training.txt"
    else:
        filename += "validation.txt"

    dataset = ([], [])
    file = open(filename, "r")
    file = file.read()
    for line in file.split("\n"):
        if (len(line) > 1):
            split_list = line.split(":")
            dataset[0].append(split_list[0])
            dataset[1].append(float(split_list[1]))

    filenames = dataset[0]
    data = []
    labels = []
    for f in range(len(filenames)):
        l = dataset[1][f]
        curr_filename = filenames[f]
        path = polarity_filename
        if (l == 0):
            path += "neg/"
        else:
            path += "pos/"

        file = open(path + curr_filename, "r")
        file = file.read()
        words = []
        for sentence in file.split("."):
            if (len(sentence) <= 1):
                continue
            words += parse(sentence)
        word_vecs = create_word_vectors(words, sen_len, corpus)
        data.append(word_vecs)
        labels.append(int(l))

    return data, labels

def load_filenames(train):
    filename = polarity_filename + "processed/"
    if (train):
        filename += "training.txt"
    else:
        filename += "validation.txt"

    dataset = ([], [])
    file = open(filename, "r")
    file = file.read()
    for line in file.split("\n"):
        if (len(line) > 1):
            split_list = line.split(":")
            dataset[0].append(split_list[0])
            dataset[1].append(float(split_list[1]))

    filenames = dataset[0]
    labels = dataset[1]
    return filenames, labels

def create_word_vectors(words, sen_len, corpus):
    txt = torch.LongTensor(np.zeros(sen_len, dtype=np.int64))
    count = 0
    for word in words:
        if word in corpus.dictionary.word2idx:
            if count > sen_len - 1:
                break
            txt[count] = corpus.dictionary.word2idx[word]
            count += 1
    return txt


def return_all_filenames(path):
    filenames = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if (".txt" in file):
                filenames.append(file)

    return filenames


class TextDataset(Dataset):

    def __init__(self, corpus, train=False, sen_len=16, max_length=256, imdb=False):
        self.sen_len = sen_len
        self.max_length = max_length
        self.corpus = corpus
        self.pos_path = polarity_filename + "/pos/"
        self.neg_path = polarity_filename + "/neg/"
        self.train = train
        self.imdb = imdb

        # Loading the basic polarity dataset:
        if (not imdb):
            if (folder_empty(polarity_filename + "processed/")):
                PARTITION = 0.8
                self.load(PARTITION)

            self.filenames, self.label = load_filenames(self.train)

        # Loading the IMDB dataset:
        else:
            self.filenames, self.label = load_imdb(self.train, imdb_filename, self.train, self.sen_len, self.corpus)



    def load(self, partition):
        partition = partition

        # All positive reviews:
        pos_data = np.array(return_all_filenames(self.pos_path))
        pos_labels = np.ones(len(pos_data))

        # All negative reviews:
        neg_data = np.array(return_all_filenames(self.neg_path))
        neg_labels = np.zeros(len(pos_data))

        all_data = np.concatenate((pos_data, neg_data))
        all_labels = np.concatenate((pos_labels, neg_labels))

        to_shuffle = list(zip(all_data, all_labels))

        random.shuffle(to_shuffle)

        shuffled_data, shuffled_labels = zip(*to_shuffle)

        self.train_data = shuffled_data[0: int(len(shuffled_data)*partition)]
        self.train_label = shuffled_labels[0: int(len(shuffled_labels)*partition)]
        train_set = (self.train_data, self.train_label)
        write_data(train_set, True)


        self.val_data = shuffled_data[int(len(shuffled_data)*partition):]
        self.val_label = shuffled_labels[int(len(shuffled_labels)*partition):]
        val_set = (self.val_data, self.val_label)
        write_data(val_set, False)


    def __getitem__(self, index):

        txt = [torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))]
        count = 0
        done =  False
        label = int(self.label[index])
        if (not self.imdb):
            if (label == 1):
                filename = self.pos_path + self.filenames[index]
            else:
                filename = self.neg_path + self.filenames[index]
        else:
            filename = self.filenames[index]
        with open(filename, "r") as file:
            for words in file:
                for word in parse(words):
                    if word in self.corpus.dictionary.word2idx:
                        if (count > self.max_length -1):
                            done = True
                            break
                        if count > self.sen_len - 1:
                            txt.append(torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64)))
                        txt[-1][count%self.sen_len] = self.corpus.dictionary.word2idx[word]
                        count += 1
                if done:
                    break
        label = torch.FloatTensor([label])
        return txt, label

    def __len__(self):
        return len(self.filenames)




