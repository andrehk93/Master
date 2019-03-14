from utils.text import textLoader as loader, glove as gloveLoader
from data.text.fast_text.parse import FastText
from models import reinforcement_models
import torch

from data.text.text_dataset import TEXT
from data.text.text_class_margin import TextMargin


class TextModelSetup:

    def __init__(self, margin_sampling, margin_size, margin_time, embedding_size, sentence_length):
        # PARAMETERS
        # TEXT AND MODEL DETAILS:
        self.EMBEDDING_SIZE = embedding_size

        # Need to remake dataset if change ANY of these:
        self.SENTENCE_LENGTH = sentence_length
        self.NUMBER_OF_SENTENCES = 1
        self.DICTIONARY_MAX_SIZE = 20000

        # TRUE = REMOVE self.STOPWORDS:
        self.STOPWORDS = False

        self.CMS = margin_sampling
        self.MARGIN_SIZE = margin_size
        self.MARGIN_TIME = margin_time


class TextNetworkSetup:

    def __init__(self, setup, dataset, args):
        self.setup = setup
        self.dataset = dataset
        self.args = args

        self.text_loader = self.setup_utility_loaders(self.setup, self.dataset)

        self.q_network = self.setup_network(self.setup, args)

        self.train_loader, self.test_loader, self.idx2word = \
            self.setup_loaders(self.setup, self.dataset, self.q_network, args, self.text_loader)

    def setup_utility_loaders(self, setup, dataset):
        print("Setting up WordVectors...")

        if self.args.GLOVE:
            data_loader = gloveLoader.GloveLoader("", setup.EMBEDDING_SIZE)
        else:
            data_loader = FastText("")
            setup.EMBEDDING_SIZE = 300

        text_loader = loader.TextLoader(data_loader, dataset, classify=False, partition=0.8, classes=True,
                                        dictionary_max_size=setup.DICTIONARY_MAX_SIZE,
                                        sentence_length=setup.SENTENCE_LENGTH, stopwords=setup.STOPWORDS,
                                        embedding_size=setup.EMBEDDING_SIZE, glove=self.args.GLOVE)

        return text_loader

    def setup_network(self, setup, args):
        print("Setting up Q Network...")
        if args.LSTM:
            q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda, args.class_vector_size,
                                                           setup.EMBEDDING_SIZE,
                                                           embedding_weight_matrix=
                                                           self.text_loader.embedding_weight_matrix,
                                                           embedding=True, dict_size=setup.DICTIONARY_MAX_SIZE)
        elif args.NTM:
            q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda, args.class_vector_size,
                                                           setup.EMBEDDING_SIZE,
                                                           embedding_weight_matrix=
                                                           self.text_loader.embedding_weight_matrix,
                                                           embedding=True, dict_size=setup.DICTIONARY_MAX_SIZE)
        else:
            q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda, args.class_vector_size,
                                                            setup.EMBEDDING_SIZE,
                                                            embedding_weight_matrix=
                                                            self.text_loader.embedding_weight_matrix,
                                                            embedding=True, dict_size=setup.DICTIONARY_MAX_SIZE)
        return q_network

    def setup_loaders(self, setup, dataset, q_network, args, text_loader):
        print("Setting up Dataloaders...")
        idx2word = []
        # MARGIN SAMPLING:
        if setup.CMS:
            train_loader = torch.utils.data.DataLoader(
                TextMargin(dataset, train=True, download=True, data_loader=text_loader, classes=args.class_vector_size,
                           episode_size=args.episode_size, tensor_length=setup.NUMBER_OF_SENTENCES,
                           sentence_length=setup.SENTENCE_LENGTH, embedding_size=setup.EMBEDDING_SIZE, margin_time=setup.MARGIN_TIME,
                           MARGIN_SIZE=setup.MARGIN_SIZE, q_network=q_network, glove=self.args.GLOVE),
                batch_size=args.batch_size, shuffle=False)

        # NO MARGIN:
        else:
            text_class = TEXT(dataset, train=True, download=True, data_loader=text_loader,
                              classes=args.class_vector_size, episode_size=args.episode_size,
                              tensor_length=setup.NUMBER_OF_SENTENCES, sentence_length=setup.SENTENCE_LENGTH,
                              embedding_size=setup.EMBEDDING_SIZE, glove=self.args.GLOVE)
            idx2word = text_class.dictionary.dictionary.idx2word
            train_loader = torch.utils.data.DataLoader(
                text_class,
                batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            TEXT(dataset, train=False, data_loader=text_loader, classes=args.class_vector_size,
                 episode_size=args.episode_size, tensor_length=setup.NUMBER_OF_SENTENCES,
                 sentence_length=setup.SENTENCE_LENGTH, embedding_size=setup.EMBEDDING_SIZE, glove=self.args.GLOVE),
            batch_size=args.test_batch_size, shuffle=True)

        return train_loader, test_loader, idx2word
