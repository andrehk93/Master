from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import errno
import torch
import numpy as np


class TextMargin(data.Dataset):
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
    def __init__(self, root, train=True, download=False, partition=0.8, data_loader=None, classes=3,
                 episode_size=30, tensor_length=18, sentence_length=50, cuda=False, scenario=False,
                 scenario_size=5, embedding_size=200, margin_time=4, MARGIN_SIZE=2, q_network=None, glove=False):
        self.root = os.path.expanduser(root)
        self.tensor_length = tensor_length
        self.sentence_length = sentence_length
        self.train = train  # training set or test set
        self.classes = classes
        self.episode_size = episode_size
        self.classify = data_loader.classify
        self.scenario_size = scenario_size
        self.margin_time = margin_time
        self.MARGIN_SIZE = MARGIN_SIZE
        self.print = True
        self.q_network = q_network
        self.scenario = scenario
        self.all_margins = []
        self.cuda = cuda
        self.pretrained_vectors = "fast"
        if glove:
            self.pretrained_vectors = "glove"

        # Saving different versions so we can experiment with different setups simultaneously
        self.training_file += "_" + str(sentence_length) + "_" + str(embedding_size)\
                              + "_" + str(self.pretrained_vectors) + ".pt"
        self.test_file += "_" + str(sentence_length) + "_" + str(embedding_size)\
                          + "_" + str(self.pretrained_vectors) + ".pt"
        self.dictionary_file += "_" + str(sentence_length) + "_" + str(embedding_size)\
                                + "_" + str(self.pretrained_vectors) + ".pt"
        self.word_vector_file += "_" + str(sentence_length) + "_" + str(embedding_size)\
                                 + "_" + str(self.pretrained_vectors) + ".pt"

        self.partition = partition

        if download and not self._check_exists():
            print("Loading pre-parsed dataset!")
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
        
        if self.train:
            # Collect randomly drawn classes:
            text_classes = np.random.choice(len(self.train_labels), self.classes*self.MARGIN_SIZE, replace=False)
                
            # Give random class-slot in vector:
            ind = 0
            for i in text_classes:
                text_samples = np.random.choice(len(self.train_data[i]), int(self.episode_size/self.classes),
                                                replace=False)
                for j in text_samples:
                    text_list.append(self.train_data[i][j])
                    label_list.append(ind)
                ind += 1

        episode_tensor = torch.zeros(len(text_list), 1, self.sentence_length).type(torch.LongTensor)

        # Iterating over all texts collected:
        for i in range(len(text_list)):
                episode_tensor[i][0] = text_list[i][0]

        return episode_tensor, torch.LongTensor(label_list)

    # Need to fake this so the batch-collector doesn't collect to few batches
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
