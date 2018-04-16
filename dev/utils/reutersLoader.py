import numpy as np
import os
import os.path
import random
import errno
import torch
from utils import texthandler

class ReutersLoader():

	raw_folder = 'raw'
	processed_folder = 'processed'
	training_file = 'training.pt'
	test_file = 'test.pt'

	def __init__(self, root, classify=True, partition=0.8, classes=False, dictionary_max_size=5000, sentence_length=16):
		self.root = os.path.expanduser(root)
		self.classify = classify
		if (self.classify):
			self.training_file = "classify_" + self.training_file
			self.test_file = "classify_" + self.test_file
		self.partition = partition
		self.classes = classes
		self.dictionary_max_size = dictionary_max_size
		self.sentence_length = sentence_length
		self.load()


	def _check_exists(self):
		return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
		os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

	def load(self):

		if self._check_exists():
			return

		# Make dirs
		try:
			os.makedirs(os.path.join(self.root, self.processed_folder))
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise

		# process and save as torch files
		print('Processing raw dataset...')
		(training_set, test_set, label_stop), word_dictionary = read_text_file(os.path.join(self.root, self.raw_folder), label_start=0, partition=self.partition, \
																			classes=self.classes, dict_max_size=self.dictionary_max_size, sentence_length=self.sentence_length)
		self.training_set = (
			training_set[0],
			torch.LongTensor(training_set[1])
		)
		self.test_set = (
			test_set[0],
			torch.LongTensor(test_set[1])
		)
		self.dictionary = word_dictionary

		print('Done!')

	def get_training_set(self):
		return self.training_set

	def get_test_set(self):
		return self.test_set

	def get_dictionary(self):
		return self.dictionary

# Need to ensure a uniformly distributed training/test-set:
def read_text_file(path, training_set=None, test_set=None, label_start=0, partition=0.8, classes=False, dict_max_size=5000, sentence_length=16):
	# Create a dictionary first:
	word_dictionary = texthandler.Corpus(dict_max_size)

	# Create the dataset-vectors based on this dictionary:
	uniform_distr = {}
	label_dict = {}
	label = label_start
	progress = 0
	for (root, dirs, files) in os.walk(path):
		for f in files:
			if (progress % 1000 == 0):
				print("Reading file nr. " + str(progress))
			if (f != ".DS_Store"):
				# Reading text file:
				text = open(os.path.join(root, f), "r").read()

				# Parsing text-file, and update Corpus:
				text = texthandler.create_word_vectors(text, sentence_length, word_dictionary)

				if (root not in label_dict):
					label_dict[root] = label
					label += 1

				if (label_dict[root] not in uniform_distr):
					uniform_distr[label_dict[root]] = [text]
				else:
					uniform_distr[label_dict[root]].append(text)
			progress += 1

	return create_datasets(uniform_distr, training_set=training_set, test_set=test_set, partition=partition, shuffle=True, classes=classes), word_dictionary

def create_datasets(text_dictionary, training_set=None, test_set=None, partition=0.8, shuffle=True, classes=False):
	# Splitting the dataset into two parts with completely different EXAMPLES, but same CLASSES:
	if not classes:
		if (training_set == None):
			training_labels = []
			training_text = []
			test_labels = []
			test_text = []
		else:
			training_text, training_labels = training_set
			test_text, test_labels = test_set

		# Create dataset from the collected text:
		for label in text_dictionary.keys():
			txts = text_dictionary[label]
			if shuffle:
				random.shuffle(txt)
			split = int(partition*len(txts))

			#Train set:
			for i in range(split):
				training_text.append(txts[i])
				training_labels.append(int(label))
			#Test set:
			for i in txts[split:]:
				test_text.append(i)
			for i in range(len(txt)-split):
				test_labels.append(int(label))

	# Splitting the dataset into two disjoint parts with completely different CLASSES:
	else:
		all_text = []
		all_labels = []
		if training_set != None:
			training_text, training_labels = training_set
			test_text, test_labels = test_set
		else:
			training_text = []
			training_labels = []
			test_text = []
			test_labels = []
		for label in text_dictionary.keys():
			all_text.append(text_dictionary[label])
			all_labels.append(label)

		split = int(len(all_labels)*partition)

		shuffle_list = list(zip(all_text, all_labels))
		random.shuffle(shuffle_list)
		shuffled_text, shuffled_labels = zip(*shuffle_list)
		
		for i in range(split):
			training_text.append(shuffled_text[i])
			training_labels.append(shuffled_labels[i])

		for i in range(split, len(shuffled_text)):
			test_text.append(shuffled_text[i])
			test_labels.append(shuffled_labels[i])

	print("Length of training set: ", len(training_text), "\nLength of test set: ", len(test_text))
	print("Nof. Classes in training set: ", len(list(set(training_labels))), "\nNof. Classes in test set: ", len(list(set(test_labels))))
	return (training_text, training_labels), (test_text, test_labels), label