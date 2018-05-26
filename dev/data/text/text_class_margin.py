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



class TextMargin(data.Dataset):
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
    def __init__(self, root, train=True, download=False, partition=0.8, data_loader=None, classes=3, episode_size=30, tensor_length=18, sentence_length=50, cuda=False, scenario=False, scenario_size=5, margin_time=4, CMS=2, q_network=None):
        self.root = os.path.expanduser(root)
        self.tensor_length = tensor_length
        self.sentence_length = sentence_length
        self.train = train  # training set or test set
        self.classes = classes
        self.episode_size = episode_size
        self.classify = data_loader.classify
        self.scenario_size = scenario_size
        self.margin_time = margin_time
        self.CMS = CMS
        self.q_network = q_network
        self.scenario = scenario
        self.all_margins = []
        self.cuda = cuda
        if (self.classify):
            self.training_file = "classify_" + self.training_file
            self.test_file = "classify_" + self.test_file
        self.partition = partition

        if download and not self._check_exists():
            self.training_set = data_loader.get_training_set()
            self.test_set = data_loader.get_test_set()
            self.dictionary = data_loader.get_dictionary()
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
            # Collect randomly drawn classes:
            accepted = False
            while not accepted:
                text_classes = np.random.choice(len(self.test_labels), 2, replace=False)
                accepted = True
                for i in text_classes:
                    if (len(self.test_data[i]) < self.scenario_size):
                        accepted = False

            # Give random class-slot in vector:
            ind = 0
            for i in text_classes:
                if (ind == 0):
                    for j in range(self.scenario_size):
                        text_list.append(self.test_data[i][j])
                        label_list.append(ind)
                else:
                    text_list.append(self.test_data[i][random.randint(0, len(self.test_data[i]) -1)])
                    label_list.append(ind)
                ind += 1


            episode_texts, episode_labels = [], []
            tensor_length = self.tensor_length
            for index in range(len(text_list)):
                episode_texts.append(text_list[index])
                episode_labels.append(label_list[index])

            if (self.cuda):
                zero_tensor = torch.LongTensor(torch.cat([torch.LongTensor(torch.cat([torch.zeros(self.sentence_length).type(torch.LongTensor) for i in range(tensor_length)])) for j in range(self.scenario_size + 1)]))
            else:
                zero_tensor = torch.LongTensor(torch.cat([torch.LongTensor(torch.cat([torch.zeros(self.sentence_length).type(torch.LongTensor) for i in range(tensor_length)])) for j in range(self.scenario_size + 1)]))
            
            episode_tensor = zero_tensor.view(self.scenario_size + 1, tensor_length, self.sentence_length)

            for i in range(len(episode_texts)):
                for j in range(tensor_length):
                    if (j >= len(episode_texts[i])):
                        break
                    episode_tensor[i][j] = episode_texts[i][j]

            if (self.cuda):
                return episode_tensor, torch.LongTensor(episode_labels).cuda()
            else:
                return episode_tensor, torch.LongTensor(episode_labels)


        else:
            if self.train:
                # Collect randomly drawn classes:
                text_classes = np.random.choice(len(self.train_labels), self.classes*self.CMS, replace=False)
                    
                # Give random class-slot in vector:
                ind = 0
                for i in text_classes:
                    n = 0
                    for j in self.train_data[i]:
                        if (n >= int(self.episode_size/self.classes)):
                            break

                        text_list.append(j)
                        label_list.append(ind)
                        n += 1
                    ind += 1
            else:
                # Collect randomly drawn classes:
                text_classes = np.random.choice(len(self.test_labels), self.classes*self.CMS, replace=False)
                    
                # Give random class-slot in vector:
                ind = 0
                for i in text_classes:
                    n = 0
                    for j in self.test_data[i]:
                        if (n >= int(self.episode_size/self.classes)):
                            break
                        text_list.append(j)
                        label_list.append(ind)
                        n += 1
                    ind += 1

            tensor_length = self.tensor_length

            zero_tensor = torch.LongTensor(torch.cat([torch.LongTensor(torch.cat([torch.zeros(self.sentence_length).type(torch.LongTensor) for i in range(tensor_length)])) for j in range(len(text_list))]))
            
            episode_tensor = zero_tensor.view(len(text_list), tensor_length, self.sentence_length)

            for i in range(len(text_list)):
                for j in range(tensor_length):
                    if (j >= len(text_list[i])):
                        break
                    episode_tensor[i][j] = text_list[i][j]

            return episode_tensor, torch.LongTensor(label_list)

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
