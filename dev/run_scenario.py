import os
import matplotlib.pyplot as plt
import numpy as np

from utils.images import imageLoader as imageLoader
from utils.text import textLoader as textLoader
from data.images.omniglot.omniglot import OMNIGLOT
from data.images.mnist.MNIST import MNIST
from data.text.text_dataset import TEXT
from reinforcement_utils.reinforcement import ReinforcementLearning as rl
from utils import transforms
import torch
from torch.utils.data import DataLoader
from models import reinforcement_models


def load_scenario(size, batch_size, dataset):

    scenario_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])
    
    
    # OMNIGLOT:
    if (dataset == 0):
        root = 'data/images/omniglot'
        print("Loading OMNIGLOT scenario...")
        omniglot_loader = imageLoader.OmniglotLoader(root, classify=False, partition=0.8, classes=True)
        scenario_loader = torch.utils.data.DataLoader(
            OMNIGLOT(root, train=False, transform=scenario_transform, download=True, omniglot_loader=omniglot_loader, episode_size=0, scenario=True, scenario_size=size, test=True),
            batch_size=batch_size, shuffle=True)

    # INH:
    elif (dataset == 2):
        root = 'data/text/headlines'
        print("Loading INH scenario...")
        text_loader = textLoader.TextLoader(root, classify=False, partition=0.8, classes=True,\
                                        dictionary_max_size=DICTIONARY_MAX_SIZE, sentence_length=SENTENCE_LENGTH, stopwords=True)
        scenario_loader = torch.utils.data.DataLoader(
        TEXT(root, train=False, download=True, data_loader=text_loader, classes=classes, episode_size=0, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH, scenario=True, scenario_size=size),
                batch_size=batch_size, shuffle=True)
    else:
        root = 'data/images/mnist'

        print("Loading MNIST scenario...")
        scenario_loader = torch.utils.data.DataLoader(
            MNIST(root, transform=scenario_transform, download=True, scenario_size=scenario_size, scenario=True),
            batch_size=batch_size, shuffle=True)

    return scenario_loader


def add_list(list1, list2, dim=1):
    if (dim == 1):
        for l in range(len(list2)):
            list1[l] += list2[l]
    elif (dim == 2):
        for l in range(len(list2)):
            for i in range(len(list2[l])):
                list1[l][i] += list2[l][i]

def divide_list(list1, iterations, dim=1):
    if (dim == 1):
        for l in range(len(list1)):
            list1[l] = float(list1[l]/iterations)
    elif (dim == 2):
        for l in range(len(list1)):
            for i in range(len(list1[l])):
                list1[l][i] = float(list1[l][i]/iterations)



def bar_plot(lists, bar_type, name, labels, size):
    plot_list = []
    for i in range(len(lists)):
        if (type(lists[i]) != type(0.2)):
            plot_list.append([])
            for j in range(len(lists[i])):
                plot_list[i].append(abs(lists[i][j]))
        else:
            plot_list.append(abs(lists[i]))

    fig, ax = plt.subplots()
    ax.yaxis.grid(True)


    lab = 0
    colors = ["red", "green", "blue", "yellow", "magenta", "white", "grey"]
    if ("Request" in bar_type):
        x1 = np.arange(1, len(plot_list))
        x2 = len(plot_list)
        plt.bar(x1, plot_list[0:len(plot_list)-1], color=colors[0], label=labels[0], edgecolor="black")
        plt.bar(x2, plot_list[-1], color=colors[1], label=labels[1], edgecolor="black")

        plt.ylabel("% Label Requests")

    else:
        np_lists = np.array(plot_list).transpose()
        x = np.arange(1, len(plot_list) + 1)
        bottom_list = []
        for i in range(len(np_lists)):

            if (len(bottom_list) == 0):
                plt.bar(x, np_lists[i], color=colors[i], label=labels[i], edgecolor="black")
                bottom_list = np_lists[i]
            else:
                plt.bar(x, np_lists[i], bottom=bottom_list, color=colors[i], label=labels[i], edgecolor="black")
                bottom_list += np_lists[i]
        plt.ylabel("Class Q-value")

    plt.legend(loc=9)
    plt.title("RL Scenario")
    plt.xlabel("Time step")
    
    plt.ylim((0, 1))
    directory = "results/plots/"
    if not os.path.exists(directory + name):
        os.makedirs(directory + name)
    plt.savefig(directory + name + bar_type + "_" + str(size) + ".png")
    plt.show()


if __name__ == '__main__':

    name = 'reinforced_lstm3/'
    checkpoint = 'pretrained/' + name + 'best.pth.tar'

    batch_size = 64
    scenario_size = 5
    classes = 3
    cuda = False

    OMNIGLOT_DATASET = 0
    MNIST_DATASET = 1
    INH_DATASET = 2
    REUTERS_DATASET = 3
    

    dataset = OMNIGLOT_DATASET

    # LSTM & Q Learning
    IMAGE_SCALE = 20
    IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
    OMNIGLOT_DATASET = True
    ##################

    # TEXT AND MODEL DETAILS:
    EMBEDDING_SIZE = 128
    SENTENCE_LENGTH = 12
    NUMBER_OF_SENTENCES = 1
    DICTIONARY_MAX_SIZE = 10000

    scenario_loader = load_scenario(scenario_size, batch_size, dataset)

    LSTM = True
    NTM = False
    LRUA = False

    if (dataset == REUTERS_DATASET):
        name = "REUTERS_" + name
    elif (dataset == MNIST_DATASET):
        name = "MNIST_" + name

    if (dataset < 2):
        if LSTM:
            q_network = reinforcement_models.ReinforcedRNN(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)
        elif NTM:
            q_network = reinforcement_models.ReinforcedNTM(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)
        elif LRUA:
            q_network = reinforcement_models.ReinforcedLRUA(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)

        from reinforcement_utils.images import scenario
    else:
        if LSTM:
            q_network = reinforcement_models.ReinforcedRNN(batch_size, cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)
        elif NTM:
            q_network = reinforcement_models.ReinforcedNTM(batch_size, cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)
        elif LRUA:
            q_network = reinforcement_models.ReinforcedLRUA(batch_size, cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)
        from reinforcement_utils.text import scenario

    if os.path.isfile(checkpoint):
        checkpoint = torch.load(checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))

    rl = rl(classes)

    iterations = 10

    # Scenario 1:
    requests, accuracies, total_percentages = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
    for t in range(iterations - 1):
        r, a, r_p = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
        add_list(requests, r, dim=1)
        add_list(accuracies, a, dim=2)
        add_list(total_percentages, r_p, dim=1)

    divide_list(requests, iterations, dim=1)
    divide_list(accuracies, iterations, dim=2)
    divide_list(total_percentages, iterations, dim=1)

    bar_plot(requests, "Request", name, ["First Class", "Second Class"], scenario_size)
    bar_plot(accuracies, "Accuracy", name, ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], scenario_size)
    bar_plot(total_percentages, "Request Percentage", name, ["First Class", "Second Class"], scenario_size)








