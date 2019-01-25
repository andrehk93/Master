from utils.text import textLoader as loader, glove as gloveLoader
from models import reinforcement_models
import torch

from data.text.text_dataset import TEXT
from data.text.text_class_margin import TextMargin


class TextModelSetup:

    def __init__(self, margin_sampling, margin_time):
        # PARAMETERS
        # TEXT AND MODEL DETAILS:
        self.EMBEDDING_SIZE = 200

        # Need to remake dataset if change ANY of these:
        self.SENTENCE_LENGTH = 12
        self.NUMBER_OF_SENTENCES = 1
        self.DICTIONARY_MAX_SIZE = 400000

        # TRUE = REMOVE self.STOPWORDS:
        self.STOPWORDS = True

        self.CMS = margin_sampling
        self.MARGIN_TIME = margin_time


class TextNetworkSetup:

    def __init__(self, setup, dataset, args):
        self.setup = setup
        self.dataset = dataset

        self.glove_loader, self.text_loader = self.setup_utility_loaders(self.setup, self.dataset)

        self.q_network = self.setup_network(self.setup, args)

        self.train_loader, self.test_loader = self.setup_loaders(self.setup, self.dataset, self.q_network,
                                                                 args, self.text_loader)

    def setup_utility_loaders(self, setup, dataset):
        print("Setting up GloVe...")

        glove_loader = gloveLoader.GloveLoader("", self.setup.EMBEDDING_SIZE)

        text_loader = loader.TextLoader(glove_loader, dataset, classify=False, partition=0.8, classes=True,
                                        dictionary_max_size=setup.DICTIONARY_MAX_SIZE,
                                        sentence_length=setup.SENTENCE_LENGTH,
                                        stopwords=setup.STOPWORDS, embedding_size=setup.EMBEDDING_SIZE)

        return gloveLoader, text_loader

    def setup_network(self, setup, args):
        print("Setting up Q Network...")
        if args.LSTM:
            q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda, args.class_vector_size,
                                                           setup.EMBEDDING_SIZE,
                                                           weights_matrix=self.text_loader.weights_matrix,
                                                           embedding=True, dict_size=setup.DICTIONARY_MAX_SIZE)
        elif args.NTM:
            q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda, args.class_vector_size,
                                                           setup.EMBEDDING_SIZE, embedding=True,
                                                           dict_size=setup.DICTIONARY_MAX_SIZE)
        else:
            q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda, args.class_vector_size,
                                                            setup.EMBEDDING_SIZE, embedding=True,
                                                            dict_size=setup.DICTIONARY_MAX_SIZE)
        return q_network

    def setup_loaders(self, setup, dataset, q_network, args, text_loader):
        print("Setting up Dataloaders...")

        # MARGIN SAMPLING:
        if setup.CMS:
            train_loader = torch.utils.data.DataLoader(
                TextMargin(dataset, train=True, download=True, data_loader=text_loader, classes=args.class_vector_size,
                           episode_size=args.episode_size, tensor_length=setup.NUMBER_OF_SENTENCES,
                           sentence_length=setup.SENTENCE_LENGTH, margin_time=setup.MARGIN_TIME, CMS=setup.CMS,
                           q_network=q_network), batch_size=args.batch_size, shuffle=False)

        # NO MARGIN:
        else:
            train_loader = torch.utils.data.DataLoader(
                TEXT(dataset, train=True, download=True, data_loader=text_loader, classes=args.class_vector_size,
                     episode_size=args.episode_size, tensor_length=setup.NUMBER_OF_SENTENCES,
                     sentence_length=setup.SENTENCE_LENGTH),
                batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            TEXT(dataset, train=False, data_loader=text_loader, classes=args.class_vector_size,
                 episode_size=args.episode_size, tensor_length=setup.NUMBER_OF_SENTENCES,
                 sentence_length=setup.SENTENCE_LENGTH), batch_size=args.test_batch_size, shuffle=True)

        return train_loader, test_loader
