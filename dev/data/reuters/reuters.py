from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import random
import os
import os.path
import errno
import torch
import numpy as np
from utils import transforms



class REUTERS(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    dictionary_file = 'dictionary.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''
    def __init__(self, root, train=True, download=False, partition=0.8, reuters_loader=None, classes=3, episode_size=30, tensor_length=18, sentence_length=50):
        self.root = os.path.expanduser(root)
        self.tensor_length = tensor_length
        self.sentence_length = sentence_length
        self.train = train  # training set or test set
        self.classes = classes
        self.episode_size = episode_size
        self.classify = reuters_loader.classify
        if (self.classify):
            self.training_file = "classify_" + self.training_file
            self.test_file = "classify_" + self.test_file
        self.partition = partition

        if download and not self._check_exists():
            self.training_set = reuters_loader.get_training_set()
            self.test_set = reuters_loader.get_test_set()
            self.dictionary = reuters_loader.get_dictionary()
            self.write_datasets()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               'Check instructions at GitHub on where to download!')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
        
        self.dictionary = torch.load(os.path.join(self.root, self.processed_folder, self.dictionary_file))

    def __getitem__(self, index):
        text_list = []
        label_list = []
        if self.train:
            # Collect randomly drawn classes:
            accepted = False
            while not accepted:
                text_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)
                accepted = True
                for i in text_classes:
                    if (len(self.train_data[i]) < int(self.episode_size / self.classes)):
                        accepted = False
                    

            # Give random class-slot in vector:
            ind = 0
            for i in text_classes:
                for j in self.train_data[i]:
                    if (len(j) == 0):
                        continue
                    text_list.append(j)
                    label_list.append(ind)
                ind += 1
        else:
            # Collect randomly drawn classes:
            accepted = False
            while not accepted:
                text_classes = np.random.choice(len(self.test_labels), self.classes, replace=False)
                accepted = True
                for i in text_classes:
                    if (len(self.test_data[i]) < int(self.episode_size / self.classes)):
                        accepted = False

            # Give random class-slot in vector:
            ind = 0
            for i in text_classes:
                for j in self.test_data[i]:
                    text_list.append(j)
                    label_list.append(ind)
                ind += 1

        text_indexes = np.random.choice(len(text_list), self.episode_size, replace=False)

        episode_texts, episode_labels = [], []
        tensor_length = self.tensor_length
        for index in text_indexes:
            episode_texts.append(text_list[index])
            episode_labels.append(label_list[index])

        zero_tensor = torch.LongTensor(torch.cat([torch.LongTensor(torch.cat([torch.zeros(self.sentence_length).type(torch.LongTensor) for i in range(tensor_length)])) for j in range(self.episode_size)]))
        episode_tensor = zero_tensor.view(self.episode_size, tensor_length, self.sentence_length)

        episode_list = list(zip(episode_texts, episode_labels))

        #shuffle_list = list(zip(text_list, label_list))
        random.shuffle(episode_list)
        shuffled_text, shuffled_labels = zip(*episode_list)

        for i in range(self.episode_size):
            for j in range(tensor_length):
                if (j >= len(shuffled_text[i])):
                    break
                episode_tensor[i][j] = shuffled_text[i][j]


        """
        print("labels: ", len(shuffled_labels))
        print("Text: ", len(shuffled_text))
        print("Text[0]: ", len(shuffled_text[0]))
        print("Text[0][0]: ", len(shuffled_text[0][0]))
        """
        return episode_tensor, torch.LongTensor(shuffled_labels)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
        os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def write_datasets(self):

        print("Writing datasets to file...")

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(self.training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(self.test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.dictionary_file), 'wb') as f:
            torch.save(self.dictionary, f)

        print("Data successfully written...")
