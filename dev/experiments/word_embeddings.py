from setup.text import TextModelSetup, TextNetworkSetup
import argparse
from utils.arguments import parse_arguments


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning NTM',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_arguments(parser)

dataset_to_test = "../dataset/text/questions"


def invert_word_vectors(arg_parser, dataset):
    args = arg_parser.parse_args()
    # Collecting static text setup
    text_setup = TextModelSetup(False, 0)

    setup = TextNetworkSetup(text_setup, dataset, args)
    train_loader, test_loader, q_network = \
        setup.train_loader, setup.test_loader, setup.q_network

    sample_batch = test_loader.__iter__().__next__()
    print("Sample batch: ", sample_batch)


invert_word_vectors(parser, dataset_to_test)
