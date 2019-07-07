from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import random
import os
import operator
import os.path
import errno
import torch
from torch.autograd import Variable
import numpy as np
from utils import transforms



class OMNIGLOT_MARGIN(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, partition=0.8, omniglot_loader=None, classes=3, episode_size=30, scenario=False, scenario_size=5, margin_time=2, MARGIN_SIZE=2, q_network=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = classes
        self.MARGIN_SIZE = MARGIN_SIZE
        self.big_diff = 0
        self.all_margins = []
        self.episode_size = episode_size
        self.scenario = scenario
        self.scenario_size = scenario_size
        self.classify = omniglot_loader.classify
        self.q_network = q_network
        self.margin_time = margin_time
        if self.classify:
            self.training_file = "classify_" + self.training_file
            self.test_file = "classify_" + self.test_file
        self.partition = partition

        if download and not self._check_exists():
            self.training_set = omniglot_loader.get_training_set()
            self.test_set = omniglot_loader.get_test_set()
            self.write_datasets()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self.scenario:
            images = []
            if self.train:
                img_classes = np.random.choice(len(self.train_labels), 2, replace=False)
                ind = 0
                for i in img_classes:
                    if ind == 0:
                        for j in range(self.scenario_size):
                            images.append((self.train_data[i][j], ind))
                    else:
                        images.append((self.train_data[i][random.randint(0, 19)], ind))
                    ind += 1
            else:
                img_classes = np.random.choice(len(self.test_labels), 2, replace=False)
                ind = 0
                for i in img_classes:
                    if ind == 0:
                        for j in range(self.scenario_size):
                            images.append((self.test_data[i][j], ind))
                    else:
                        images.append((self.test_data[i][random.randint(0, 19)], ind))
                    ind += 1
            img_list, target_list = [], [] 

            for i in range(len(images)):
                img, label = images[i]
                img = Image.fromarray(img.numpy())

                if self.transform is not None:
                    img = self.transform(img)

                # Normalizing (pixels are binary):
                for row in range(len(img[0])):
                    for i in range(len(img[0][row])):
                        if img[0][row][i] > 0:
                            img[0][row][i] = 0.0
                        else:
                            img[0][row][i] = 1.0
                
                img_list.append(img)
                target_list.append(label)

            return img_list, target_list

        else:
            img_list, target_list = [], []
            if self.train:
                # Train-dataset:
                img_classes = np.random.choice(len(self.train_labels), int(self.classes*self.MARGIN_SIZE), replace=False)

                ind = 0
                for i in img_classes:
                    for j in self.train_data[i]:
                        img_list.append(j)
                        target_list.append(ind)
                    ind += 1

            else:
                # Test-dataset:
                img_classes = np.random.choice(len(self.test_labels), int(self.classes*self.MARGIN_SIZE), replace=False)
                ind = 0
                for i in img_classes:
                    for j in self.test_data[i]:
                        img_list.append(j)
                        target_list.append(ind)
                    ind += 1

            return img_list, target_list

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

        print("Data successfully written...")
