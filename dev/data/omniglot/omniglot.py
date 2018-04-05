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



class OMNIGLOT(data.Dataset):
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
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, partition=0.8, omniglot_loader=None, classes=3, batch_size=50):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = classes
        self.batch_size = batch_size
        self.classify = omniglot_loader.classify
        if (self.classify):
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
        images = []
        if self.train:
            # Trains on whole dataset
            img_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)

            # Trains on 3 classes:
            #img_classes = np.random.choice(3, self.classes, replace=False)
            #print("IMG CLASSES = ", img_classes)
            
            #img_classes = [0, 1, 2]
            ind = 0
            for i in img_classes:
                for j in self.train_data[i]:
                    images.append((j, ind))
                ind += 1
        else:
            img_classes = np.random.choice(len(self.test_labels), self.classes, replace=False)
            ind = 0
            for i in img_classes:
                for j in self.test_data[i]:
                    images.append((j, ind))
                ind += 1

        images_indexes = np.random.choice(len(images), self.batch_size, replace=False)
        img_list = []
        target_list = []
        rotations = [0, 90, 180, 270]

        image_rotations = [rotations[random.randint(0, len(rotations)-1)] for i in range(len(img_classes))]
        for i in images_indexes:
            img, label = images[i]
            img = Image.fromarray(img.numpy())

            if self.transform is not None:

                # Applying class specific rotations:
                if (image_rotations[label] == 90):
                    img = transforms.vflip(img)
                elif (image_rotations[label] == 180):
                    img = transforms.hflip(img)
                elif (image_rotations[label] == 270):
                    img = transforms.hflip(transforms.vflip(img))
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # Normalizing (pixels are binary):
            for row in range(len(img[0])):
                for i in range(len(img[0][row])):
                    if (img[0][row][i] > 0):
                        img[0][row][i] = 0.0
                    else:
                        img[0][row][i] = 1.0
            
            img_list.append(img)
            target_list.append(label)

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
