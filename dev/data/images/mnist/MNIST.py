from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import random
import codecs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, transform=None, target_transform=None, download=False, scenario_size=5, episode_size=30, classes=3, scenario=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.scenario_size = scenario_size
        self.episode_size = episode_size
        self.classes = classes
        self.scenario = scenario

        if download:
            self.download()

        if not self._check_exists():
            self.download()

        self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if (self.scenario):
            images = []
            img_classes = np.random.choice(len(self.test_labels), 2, replace=False)
            ind = 0
            for i in img_classes:
                if (ind == 0):
                    for j in range(self.scenario_size):
                        images.append((self.test_data[i][j], ind))
                else:
                    images.append((self.test_data[i][random.randint(0, len(self.test_data[i]) - 1)], ind))
                ind += 1
            img_list, target_list = [], [] 

            for i in range(len(images)):
                img, label = images[i]
                img = Image.fromarray(img.numpy())

                if self.transform is not None:
                    img = self.transform(img)

                # Normalizing (pixels are binary):
                threshold = torch.Tensor([0.0])
                img = (img > threshold).float() * 1

                img_list.append(img)
                target_list.append(label)
        else:
            images = []
            img_classes = np.random.choice(len(self.test_labels), self.classes, replace=False)
            ind = 0
            for i in img_classes:
                for j in self.test_data[i]:
                    images.append((j, ind))
                ind += 1
            images_indexes = np.random.choice(len(images), self.episode_size, replace=False)
            img_list = []
            target_list = []

            for i in images_indexes:
                img, label = images[i]
                img = Image.fromarray(img.numpy())

                if self.transform is not None:
                    img = self.transform(img)

                # Normalizing (pixels are binary):
                threshold = torch.Tensor([0.0])
                img = (img > threshold).float() * 1
                
                img_list.append(img)
                target_list.append(label)

        return img_list, target_list

    def __len__(self):
        return 5000

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        
        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        data_set = create_dataset(training_set, test_set)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(data_set, f)

        print('Done!')


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


class EMNIST(MNIST):
    """`EMNIST <https://www.nist.gov/itl/iad/image-group/emnist-dataset/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    def __init__(self, root, split, **kwargs):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)

    def _training_file(self, split):
        return 'training_{}.pt'.format(split)

    def _test_file(self, split):
        return 'test_{}.pt'.format(split)

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip
        import shutil
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        raw_folder = os.path.join(self.root, self.raw_folder)
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                print('Extracting ' + gzip_file)
                with open(os.path.join(raw_folder, gzip_file.replace('.gz', '')), 'wb') as out_f, \
                        gzip.GzipFile(os.path.join(gzip_folder, gzip_file)) as zip_f:
                    out_f.write(zip_f.read())
        shutil.rmtree(gzip_folder)

        # process and save as torch files
        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(raw_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(raw_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.root, self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.root, self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def create_dataset(training_set, test_set):
    print(training_set[0].size())
    print(training_set[1].size())
    print(test_set[0].size())
    print(test_set[1].size())
    input("SIZES GOOD?")

    images_sorted_by_label = []
    labels = []

    for l in training_set[1]:
        if (l not in labels):
            labels.append(l)
    labels.sort()
    print(labels)
    input("OK")
    for l in labels:
        images_sorted_by_label.append([])

    for i in range(len(training_set[0])):
        img, label = training_set[0][i], training_set[1][i]

        images_sorted_by_label[label].append(img)

    return images_sorted_by_label, torch.LongTensor(labels)















