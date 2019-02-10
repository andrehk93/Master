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


class TEXT(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training'
    test_file = 'test'
    dictionary_file = 'dictionary'
    word_vector_file = 'word_vectors'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''
    def __init__(self, root, train=True, download=False, partition=0.8, data_loader=None, classes=3, episode_size=30,
                 tensor_length=18, sentence_length=50, cuda=False, scenario=False, embedding_size=200,
                 scenario_size=5, scenario_type=0, class_choice=0, idx2word=[], glove=False):
        self.root = os.path.expanduser(root)
        self.tensor_length = tensor_length
        self.sentence_length = sentence_length
        self.train = train  # training set or test set
        self.classes = classes
        self.episode_size = episode_size
        self.classify = data_loader.classify
        self.idx2word = idx2word
        self.scenario_size = scenario_size
        self.scenario_type = scenario_type
        self.class_choice = class_choice
        self.scenario = scenario
        self.all_margins = []
        self.cuda = cuda
        self.pretrained_vectors = "fast"
        if glove:
            self.pretrained_vectors = "glove"

        # Saving different versions so we can experiment with different setups simultaneously
        self.training_file += "_" + str(sentence_length) + "_" + str(embedding_size) + "_" + str(self.pretrained_vectors) + ".pt"
        self.test_file += "_" + str(sentence_length) + "_" + str(embedding_size) + "_" + str(self.pretrained_vectors) + ".pt"
        self.dictionary_file += "_" + str(sentence_length) + "_" + str(embedding_size) + "_" + str(self.pretrained_vectors) + ".pt"
        self.word_vector_file += "_" + str(sentence_length) + "_" + str(embedding_size) + "_" + str(self.pretrained_vectors) + ".pt"

        self.partition = partition

        if download and not self._check_exists():
            self.training_set = data_loader.get_training_set()
            self.test_set = data_loader.get_test_set()
            self.dictionary = data_loader.get_dictionary()
            self.weight_vector = data_loader.get_word_vectors()
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
        if self.scenario:
            texts = []
            # As in Active One-Shot Learning:
            if self.scenario_type == 0:
                if self.train:
                    txt_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)
                    ind = 0
                    for i in txt_classes:
                        if ind == 0:
                            for j in range(self.scenario_size):
                                texts.append((self.train_data[i][j], ind))
                        else:
                            texts.append((self.train_data[i][random.randint(0, len(self.train_data[i]) - 1)], ind))
                        ind += 1
                else:
                    txt_classes = np.random.choice(len(self.test_labels), self.classes, replace=False)
                    ind = 0
                    for i in txt_classes:
                        if ind == 0:
                            for j in range(self.scenario_size):
                                texts.append((self.test_data[i][j], ind))
                        else:
                            texts.append((self.test_data[i][random.randint(0, len(self.test_data[i]) - 1)], ind))
                        ind += 1

            # Zero-shot scenario:
            elif (self.scenario_type == 1):
                if (self.train):
                    txt_classes = np.random.choice(len(self.train_labels), 3, replace=False)
                    ind = 0
                    for i in txt_classes:
                        if (ind == self.class_choice):
                            for j in range(self.scenario_size):
                                texts.append((self.train_data[i][j], ind))
                        else:
                            texts.append((self.train_data[i][random.randint(0, len(self.train_data[i]) - 1)], ind))
                        ind += 1
                else:
                    txt_classes = np.random.choice(len(self.test_labels), 3, replace=False)
                    ind = 0
                    for i in txt_classes:
                        if (ind == self.class_choice):
                            for j in range(self.scenario_size):
                                texts.append((self.test_data[i][j], ind))
                        else:
                            texts.append((self.test_data[i][random.randint(0, len(self.test_data[i]) - 1)], ind))
                        ind += 1
            # K-shot scenario:
            elif (self.scenario_type == 2):
                if (self.train):
                    txt_classes = np.random.choice(len(self.train_labels), 3, replace=False)
                    ind = 0
                    for i in txt_classes:
                        txt_samples = np.random.choice(len(self.train_data[i]), self.scenario_size, replace=False)
                        for j in txt_samples:
                            texts.append((self.train_data[i][j], ind))
                        ind += 1
                else:
                    txt_classes = np.random.choice(len(self.test_labels), 3, replace=False)
                    ind = 0
                    for i in txt_classes:
                        txt_samples = np.random.choice(len(self.test_data[i]), self.scenario_size, replace=False)
                        for j in txt_samples:
                            texts.append((self.test_data[i][j], ind))
                        ind += 1
            # One-shot scenario:
            elif (self.scenario_type == 3):
                if (self.train):
                    txt_classes = np.random.choice(len(self.train_labels), 3, replace=False)
                    appended_texts = []
                    ind = 0
                    k = 0
                    for i in txt_classes:
                        if (ind == self.class_choice):
                            txt_samples = np.random.choice(len(self.train_data[i]), self.scenario_size, replace=False)
                        else:
                            txt_samples = np.random.choice(len(self.train_data[i]), 1, replace=False)
                        for j in txt_samples:
                            if (ind == self.class_choice):
                                if (k == 0):
                                    texts.append((self.train_data[i][j], ind))
                                    k += 1
                                else:
                                    appended_texts.append((self.train_data[i][j], ind))
                            else:
                                texts.append((self.train_data[i][j], ind))
                            
                        ind += 1
                    for txt in appended_texts:
                        texts.append(txt)
                else:
                    txt_classes = np.random.choice(len(self.test_labels), 3, replace=False)
                    appended_texts = []
                    ind = 0
                    k = 0
                    for i in txt_classes:
                        if (ind == self.class_choice):
                            txt_samples = np.random.choice(len(self.test_data[i]), self.scenario_size, replace=False)
                        else:
                            txt_samples = np.random.choice(len(self.test_data[i]), 1, replace=False)
                        for j in txt_samples:
                            if (ind == self.class_choice):
                                if (k == 0):
                                    texts.append((self.test_data[i][j], ind))
                                    k += 1
                                else:
                                    appended_texts.append((self.test_data[i][j], ind))
                            else:
                                texts.append((self.test_data[i][j], ind))
                            
                        ind += 1
                    for txt in appended_texts:
                        texts.append(txt)

            episode_texts, episode_labels = [], []
            tensor_length = self.tensor_length
            for index in range(len(texts)):
                txt, lbl = texts[index]
                episode_texts.append(txt)
                episode_labels.append(lbl)

            zero_tensor = torch.LongTensor(torch.cat([torch.LongTensor(torch.cat([torch.zeros(self.sentence_length).type(torch.LongTensor)])) for j in range(len(texts))]))
            episode_tensor = zero_tensor.view(len(texts), 1, self.sentence_length)

            for i in range(len(episode_texts)):
                episode_tensor[i][0] = episode_texts[i][0]

            return episode_tensor, torch.LongTensor(episode_labels)

        # Normal drawing (training):
        else:
            if self.train:
                # Collect randomly drawn classes:
                text_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)

                # Give random class-slot in vector:
                ind = 0
                for i in text_classes:
                    text_samples = np.random.choice(len(self.train_data[i]), 15, replace=False)
                    for j in text_samples:
                        text_list.append(self.train_data[i][j])
                        label_list.append(ind)
                    ind += 1
            else:
                # Collect randomly drawn classes:
                text_classes = np.random.choice(len(self.test_labels), self.classes, replace=False)

                # Give random class-slot in vector:
                ind = 0
                for i in text_classes:
                    text_samples = np.random.choice(len(self.test_data[i]), 15, replace=False)
                    for j in text_samples:
                        text_list.append(self.test_data[i][j])
                        label_list.append(ind)
                    ind += 1
            # Select the random texts from all collected texts:
            text_indexes = np.random.choice(len(text_list), self.episode_size, replace=False)

            episode_texts, episode_labels = [], []
            for index in text_indexes:
                episode_texts.append(text_list[index])
                episode_labels.append(label_list[index])

            # Create empty tensor of desired size (including batch):
            episode_tensor = torch.zeros(self.episode_size, 1, self.sentence_length).type(torch.LongTensor)

            # Zip and shuffle:
            episode_list = list(zip(episode_texts, episode_labels))
            random.shuffle(episode_list)
            # Unzip:
            shuffled_text, shuffled_labels = zip(*episode_list)

            # Insert all texts into final episode_tensor:
            for i in range(self.episode_size):
                episode_tensor[i][0] = shuffled_text[i][0]
            return episode_tensor, torch.LongTensor(shuffled_labels)

    def __len__(self):
        return 256

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
        with open(os.path.join(self.root, self.processed_folder, self.word_vector_file), 'wb') as f:
            torch.save(self.weight_vector, f)

        print("Data successfully written...")
