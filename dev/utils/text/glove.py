import numpy as np
import os
import os.path
import random
import errno
import torch

class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []

	def add_word(self, word):
		# NEED TO SAVE MAX INDEX FOR "UNKNOWN" WORDS, AND 0 INDEX FOR PADDING:
		if word not in self.word2idx:
			self.word2idx[word] = len(self.idx2word)
			self.idx2word.append(word)

	def __len__(self):
		return len(self.idx2word)

class GloveLoader():

	directory = "data/text/glove"
	filename = "glove.6B.100d.txt"
	dictionary_filename = "glove_dict.pt"
	word_vector_filename = "glove_vectors.pt"

	def __init__(self, root):
		self.root = os.path.expanduser(root)
		self.dictionary = Dictionary();
		self.word_vectors = []
		if (not os.path.exists(os.path.join(root, self.directory, self.dictionary_filename))):
			self.loadGloveWordVectors(root)
		else:
			self.dictionary = torch.load(
                os.path.join(self.root, self.directory, self.dictionary_filename))
			self.word_vectors = torch.load(
                os.path.join(self.root, self.directory, self.word_vector_filename))
		assert(self.dictionary.idx2word[0] == "the" and self.dictionary.word2idx["the"] == 0)


	def get_dictionary(self):
		return self.dictionary.word2idx

	# Need to ensure a uniformly distributed training/test-set:
	def loadGloveWordVectors(self, path):
		# Read all words in the GloVe file:
		progress = 0
		with open(os.path.join(path, self.directory, self.filename), "r", encoding="utf8") as file:
			for word_vector in file:
				if (progress % 10000 == 0):
					print("Reading word vector [" + str(progress) + "]")

				word_and_vectors = word_vector.split(" ")
				self.dictionary.add_word(word_and_vectors[0])
				self.word_vectors.append(np.array(word_and_vectors[1:], dtype=np.float64))
				progress += 1
		self.saveWordVectorsAndDictionary(path)

	def saveWordVectorsAndDictionary(self, path):

		with open(os.path.join(path, self.directory, self.dictionary_filename), 'wb') as f:
			torch.save(self.dictionary, f)
		with open(os.path.join(path, self.directory, self.word_vector_filename), 'wb') as f:
			torch.save(self.word_vectors, f)

