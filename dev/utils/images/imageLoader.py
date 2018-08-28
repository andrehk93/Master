import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import os.path
import random
from scipy.ndimage import imread
from PIL import Image
import errno
import torch

class OmniglotLoader():

	raw_folder = 'raw'
	processed_folder = 'processed'
	training_file = 'training.pt'
	test_file = 'test.pt'

	def __init__(self, root, classify=False, partition=0.8, classes=False):
		self.root = os.path.expanduser(root)
		self.classify = classify
		if (self.classify):
			self.training_file = "classify_" + self.training_file
			self.test_file = "classify_" + self.test_file
		self.partition = partition
		self.classes = classes
		self.load()


	def _check_exists(self):
		return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
		os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

	def load(self):

		if self._check_exists():
			return

		# Make dirs
		try:
			os.makedirs(os.path.join(self.root, self.raw_folder))
			os.makedirs(os.path.join(self.root, self.processed_folder))
		except OSError as e:
			if e.errno == errno.EEXIST:
				pass
			else:
				raise

		# process and save as torch files
		print('Processing training set...')
		training_set, test_set, label_stop = read_image_file(os.path.join(self.root, self.raw_folder, 'images_background'), label_start=0, partition=self.partition, classes=self.classes)
		print('Processing evaluation set...')
		training_set, test_set, label_stop = read_image_file(os.path.join(self.root, self.raw_folder, 'images_evaluation'), training_set=training_set, test_set=test_set,
		label_start=label_stop, partition=self.partition, classes=self.classes)

		print("SHAPE: ", torch.ByteTensor(training_set[0]).size())
		self.training_set = (
			torch.ByteTensor(training_set[0]),
			torch.LongTensor(training_set[1])
		)
		self.test_set = (
			torch.ByteTensor(test_set[0]),
			torch.LongTensor(test_set[1])
		)
		#print("LOADING: ", self.training_set[0][0])

		print('Done!')

	def get_training_set(self):
		return self.training_set

	def get_test_set(self):
		return self.test_set

# Need to ensure a uniformly distributed training/test-set:
def read_image_file(path, training_set=None, test_set=None, label_start=0, partition=0.8, classes=False):
	images = []
	size = 105
	uniform_distr = {}
	label_dict = {}
	labels = []
	label = label_start
	curr_dir = ""
	for (root, dirs, files) in os.walk(path):

		for f in files:
			if (f.endswith(".png")):
			# Reading image file:
				if (len(files) < 20):
					print("Length: ", len(files))
					break
				#image = imread(os.path.join(root, f), flatten=True)
				image = np.array(Image.open(os.path.join(root, f)))
				if (root not in label_dict):
					label_dict[root] = label
					label += 1
	
				assert(image.shape == (105, 105))

				if (label_dict[root] not in uniform_distr):
					uniform_distr[label_dict[root]] = [image.astype(int).tolist()]
				else:
					uniform_distr[label_dict[root]].append(image.astype(int).tolist())
	return create_datasets(uniform_distr, training_set=training_set, test_set=test_set, partition=partition, shuffle=True, classes=classes)

def create_datasets(image_dictionary, training_set=None, test_set=None, partition=0.8, shuffle=True, classes=False):
	# Splitting the dataset into two parts with completely different EXAMPLES, but same CLASSES:
	if not classes:
		if (training_set == None):
			training_labels = []
			training_images = []
			test_labels = []
			test_images = []
		else:
			training_images, training_labels = training_set
			test_images, test_labels = test_set

		# Create dataset from the collected images:
		for label in image_dictionary.keys():
			imgs = image_dictionary[label]
			if shuffle:
				random.shuffle(imgs)
			split = int(partition*len(imgs))

			#Train set:
			for i in range(split):
				training_images.append(imgs[i])
				training_labels.append(int(label))
			#Test set:
			for i in imgs[split:]:
				test_images.append(i)
			for i in range(len(imgs)-split):
				test_labels.append(int(label))

	# Splitting the dataset into two parts with completely different CLASSES:
	else:
		all_images = []
		all_labels = []
		if training_set != None:
			training_images, training_labels = training_set
			test_images, test_labels = test_set
		else:
			training_images = []
			training_labels = []
			test_images = []
			test_labels = []
		for label in image_dictionary.keys():
			all_images.append(image_dictionary[label])
			all_labels.append(label)

		split = int(len(all_labels)*partition)

		shuffle_list = list(zip(all_images, all_labels))
		random.shuffle(shuffle_list)
		shuffled_images, shuffled_labels = zip(*shuffle_list)
		
		for i in range(split):
			training_images.append(shuffled_images[i])
			training_labels.append(shuffled_labels[i])

		for i in range(split, len(shuffled_images)):
			test_images.append(shuffled_images[i])
			test_labels.append(shuffled_labels[i])

	print("Length of training set: ", len(training_images), "\nLength of test set: ", len(test_images))
	print("Nof. Classes in training set: ", len(list(set(training_labels))), "\nNof. Classes in test set: ", len(list(set(test_labels))))
	return (training_images, training_labels), (test_images, test_labels), label