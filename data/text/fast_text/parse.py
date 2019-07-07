import io
import os
import torch
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word2idx = {"$pad": 0}
        self.idx2word = ["$pad"]

    def add_word(self, word):
        # NEED TO SAVE MAX INDEX FOR "UNKNOWN" WORDS, AND 0 INDEX FOR PADDING:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word) + 1


class FastText:
    directory = "data/text/fast_text"
    file = "wiki-news-300d-1M.vec"
    dictionary_filename = "fast_dict.pt"
    word_vector_filename = "fast_vectors.pt"

    def __init__(self, root):
        self.filename = self.file
        self.root = os.path.expanduser(root)
        self.dictionary = Dictionary()
        self.word_vectors = []
        print("Test path: ", os.path.join(root, self.directory, self.dictionary_filename))
        if not os.path.exists(os.path.join(root, self.directory, self.dictionary_filename)):
            self.load_vectors(os.path.join(root, self.directory, self.filename))
        else:
            self.dictionary = torch.load(os.path.join(root, self.directory, self.dictionary_filename))
            self.word_vectors = torch.load(os.path.join(root, self.directory, self.word_vector_filename))

    def load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        progress = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.dictionary.add_word(tokens[0])
            self.word_vectors.append(np.array(tokens[1:], dtype=np.float64))
            if progress % 50000 == 0:
                print("Reading word vector [" + str(100.0*progress/n)[0:4] + " %]")
            progress += 1
        print("Saving vectors and dictionary...")
        self.save_word_vectors_and_dictionary(self.root)

    def save_word_vectors_and_dictionary(self, path):
        print("Loading vectors and dictionaries...")
        with open(os.path.join(path, self.directory, self.dictionary_filename), 'wb') as f:
            torch.save(self.dictionary, f)
        with open(os.path.join(path, self.directory, self.word_vector_filename), 'wb') as f:
            torch.save(self.word_vectors, f)
