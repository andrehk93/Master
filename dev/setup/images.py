from utils import transforms
from models import reinforcement_models
import torch
from utils.images import imageLoader as loader
from data.images.omniglot.omniglot_class_margin import OMNIGLOT_MARGIN
from data.images.omniglot.omniglot import OMNIGLOT


class ImageModelSetup:

    def __init__(self, cms, margin_size, margin_time):
        self.IMAGE_SCALE = 20
        self.IMAGE_SIZE = self.IMAGE_SCALE * self.IMAGE_SCALE

        self.train_transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SCALE, self.IMAGE_SCALE)),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SCALE, self.IMAGE_SCALE)),
            transforms.ToTensor()
        ])

        self.CMS = cms
        self.MARGIN_SIZE = margin_size
        self.MARGIN_TIME = margin_time


class ImageNetworkSetup:

    def __init__(self, setup, dataset, args):
        self.setup = setup
        self.dataset = dataset

        self.q_network = self.setup_network(self.setup, args)

        self.train_loader, self.test_loader = self.setup_loaders(self.setup, self.dataset, self.q_network, args)

    def setup_network(self, setup, args):
        print("Setting up Q Network...")
        if args.LSTM:
            q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda,
                                                           args.class_vector_size, setup.IMAGE_SIZE)
        elif args.NTM:
            q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda,
                                                           args.class_vector_size, setup.IMAGE_SIZE)
        else:
            q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda,
                                                            args.class_vector_size, setup.IMAGE_SIZE)

        return q_network

    def setup_loaders(self, setup, dataset, q_network, args):
        print("Setting up Dataloaders...")

        omniglot_loader = loader.OmniglotLoader(dataset, classify=False, partition=0.8, classes=True)
        if setup.CMS:
            train_loader = torch.utils.data.DataLoader(
                OMNIGLOT_MARGIN(dataset, train=True, transform=setup.train_transform, download=True,
                                omniglot_loader=omniglot_loader, classes=args.class_vector_size,
                                episode_size=args.episode_size, margin_time=setup.MARGIN_TIME,
                                MARGIN_SIZE=setup.MARGIN_SIZE, q_network=q_network),
                batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                OMNIGLOT(dataset, train=True, transform=setup.test_transform, download=True,
                         omniglot_loader=omniglot_loader, classes=args.class_vector_size,
                         episode_size=args.episode_size), batch_size=args.batch_size, shuffle=True)

        print("Loading testset...")
        test_loader = torch.utils.data.DataLoader(
            OMNIGLOT(dataset, train=False, transform=setup.test_transform, omniglot_loader=omniglot_loader,
                     classes=args.class_vector_size, episode_size=args.episode_size, test=True),
            batch_size=args.test_batch_size, shuffle=True)

        return train_loader, test_loader
