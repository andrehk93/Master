from setup.text import TextModelSetup, TextNetworkSetup
from setup.images import ImageNetworkSetup, ImageModelSetup
import argparse
import numpy as np
from utils.arguments import parse_arguments
import torch
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning NTM',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_arguments(parser)


def invert_word_vectors(loader, dictionary):
    sample_batch, label_batch = loader.__iter__().__next__()
    print(sample_batch.size())
    for b in range(sample_batch.size()[0]):
        for e in range(sample_batch.size()[1]):
            sentence = ""
            for w in sample_batch[b][e][0]:
                if w >= len(dictionary):
                    sentence += "OOV "
                else:
                    sentence += dictionary[w] + " "
            print("Sentence: ", sentence)


def visualize_episode(loader, dictionary):
    sample_batch, label_batch = loader.__iter__().__next__()
    for b in range(sample_batch.size()[0]):
        for e in range(sample_batch.size()[1]):
            sentence = ""
            for w in sample_batch[b][e][0]:
                if w >= len(dictionary):
                    sentence += "OOV "
                else:
                    sentence += dictionary[w] + " "
            print("t: ", e)
            print("Question: ", sentence)
            print("Label: ", label_batch[b][e].item())


def test_padding(network, args):
    # Initialize q_network between each episode
    hidden = network.reset_hidden(args.batch_size)

    # Generating mock-state
    state = []
    for i in range(args.batch_size):
        state.append([0 for i in range(args.class_vector_size)])

    # Tensoring the state:
    state = Variable(torch.FloatTensor(state))

    pad_vector = torch.LongTensor([[[np.zeros(9)] for x in range(args.episode_size)] for y in range(args.batch_size)])
    print("Pad-vector size: ", pad_vector.size())
    input("OK")
    res = network(Variable(pad_vector).type(torch.LongTensor), hidden, class_vector=state, seq=pad_vector.size()[1],
                  display_embeddings=True)
    print("Result: ")
    print(res)
    input("OK")


def test_activated_gradients(network, text):
    if text:
        print("Embedding Gradient: ", network.embedding_layer.requires_grad)
    print("LSTM Gradient: ", network.lstm.requires_grad)
    print("Output Gradient: ", network.hidden2probs.requires_grad)


if __name__ == '__main__':
    # Parsing arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Deciding on data-set for experiment
    TEXT = False
    dataset_to_test = "data/"
    if args.MNIST:
        dataset_to_test += '/images/mnist'
        IMAGES = True
    elif args.OMNIGLOT:
        dataset_to_test += '/images/omniglot'
        IMAGES = True
    elif args.INH:
        dataset_to_test += '/text/headlines'
        TEXT = True
    elif args.REUTERS:
        dataset_to_test += '/text/reuters'
        TEXT = True
    else:
        dataset_to_test += '/text/questions'
        TEXT = True

    # Collecting setup
    if TEXT:
        # Collecting static text setup
        text_setup = TextModelSetup(False, 0)
        setup = TextNetworkSetup(text_setup, dataset_to_test, args)
    else:
        image_setup = ImageModelSetup(False, 0)
        setup = ImageNetworkSetup(image_setup, dataset_to_test, args)

    train_loader, test_loader, q_network, idx2word = \
        setup.train_loader, setup.test_loader, setup.q_network, setup.idx2word

    # Testing the sentences from word-vectors
    invert_word_vectors(train_loader, idx2word)

    # Visually testing that the sentences in the same class cohere
    visualize_episode(train_loader, idx2word)

    # Test that the padding-vector is a 0-vector
    test_padding(q_network, args)

    # Test that gradients are enabled in the correct layers
    test_activated_gradients(q_network.q_network, TEXT)
