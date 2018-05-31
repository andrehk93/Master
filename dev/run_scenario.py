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


def load_scenario(size, batch_size, dataset, scenario_type):

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
            OMNIGLOT(root, train=TRAIN, transform=scenario_transform, download=True, omniglot_loader=omniglot_loader, episode_size=0, scenario=True, scenario_size=size, test=True, scenario_type=scenario_type, class_choice=class_choice),
            batch_size=batch_size, shuffle=True)

    # MNIST:
    elif (dataset == 1):
        root = 'data/images/mnist'

        print("Loading MNIST scenario...")
        scenario_loader = torch.utils.data.DataLoader(
            MNIST(root, transform=scenario_transform, download=True, scenario_size=scenario_size, scenario=True, scenario_type=scenario_type, class_choice=class_choice),
            batch_size=batch_size, shuffle=True)

    # INH:
    elif (dataset == 2):
        root = 'data/text/headlines'
        print("Loading INH scenario...")
        text_loader = textLoader.TextLoader(root, classify=False, partition=0.8, classes=True,\
                                        dictionary_max_size=DICTIONARY_MAX_SIZE, sentence_length=SENTENCE_LENGTH, stopwords=True)
        scenario_loader = torch.utils.data.DataLoader(
        TEXT(root, train=TRAIN, download=True, data_loader=text_loader, classes=classes, episode_size=0, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH, scenario=True, scenario_size=size, scenario_type=scenario_type, class_choice=class_choice),
                batch_size=batch_size, shuffle=True)
    
    # REUTERS:
    else:
        root = 'data/text/reuters'
        print("Loading REUTERS scenario...")
        text_loader = textLoader.TextLoader(root, classify=False, partition=0.8, classes=True,\
                                        dictionary_max_size=DICTIONARY_MAX_SIZE, sentence_length=SENTENCE_LENGTH, stopwords=True)
        scenario_loader = torch.utils.data.DataLoader(
        TEXT(root, train=TRAIN, download=True, data_loader=text_loader, classes=classes, episode_size=0, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH, scenario=True, scenario_size=size, scenario_type=scenario_type, class_choice=class_choice),
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



def bar_plot(lists, bar_type, name, labels, size, ylabel=""):
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
    if ("Percentage" in bar_type):
        if (scenario_type == 0):
            x1 = np.arange(1, scenario_size + 1)
            plt.bar(x1, plot_list[0:scenario_size], color=colors[0], label=labels[0], edgecolor="black")
            for i in range(0, len(plot_list) - scenario_size):
                plt.bar(scenario_size + i + 1, plot_list[scenario_size + i], color=colors[i+1], label=labels[i+1], edgecolor="black")
        elif (scenario_type == 1):
            x_curr = 0
            for i in range(0, classes):
                if (i == class_choice):
                    x = np.arange(i + 1, i + 1 + scenario_size)
                    x_curr += scenario_size
                    y = plot_list[i : i + scenario_size]
                else:
                    x = x_curr + 1
                    y = plot_list[x_curr]
                    x_curr += 1
                plt.bar(x, y, color=colors[i], label=labels[i], edgecolor="black")
        elif (scenario_type == 2):
            for i in range(classes):
                x = np.arange(int(i*scenario_size), int((i+1)*scenario_size))
                plt.bar(x, plot_list[int(i*scenario_size) : int((i+1)*scenario_size)], color=colors[i], label=labels[i], edgecolor="black")
        else:

            for i in range(0, classes):
                plt.bar(i + 1, plot_list[i], color=colors[i], label=labels[i], edgecolor="black")
            x1 = np.arange(classes + 1, classes + scenario_size)
            print(x1, plot_list[classes:])
            plt.bar(x1, plot_list[classes:], color=colors[class_choice], edgecolor="black")

        if (len(ylabel) > 0):
            plt.ylabel(ylabel)
        else:
            plt.ylabel("% Label Requests")

    else:
        print(bar_type)
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

    name = 'reinforced_lstm_r2/'
    checkpoint = 'pretrained/' + name + 'best.pth.tar'

    LSTM = True
    NTM = False
    LRUA = False

    TRAIN = False

    batch_size = 256
    scenario_size = 10

    # Different scenario-types:
    META_SCENARIO = 0
    K_SHOT_SCENARIO = 2

    # scenario_size > 1:
    class_choice = 0
    ZERO_SHOT_SCENARIO = 1
    ONE_SHOT_SCENARIO = 3

    # CHOOSE SCENARIO:
    scenario_type = META_SCENARIO

    if (scenario_type == 0):
        name = "meta/" + name
    elif (scenario_type == 1):
        name = "zero_shot/c" + str(class_choice) + "_" + name
    elif (scenario_type == 2):
        name = "k_shot/" + name
    elif (scenario_type == 3):
        name = "one_shot/c" + str(class_choice) + "_" + name 

    classes = 3
    cuda = False

    OMNIGLOT_DATASET = 0
    MNIST_DATASET = 1
    INH_DATASET = 2
    REUTERS_DATASET = 3
    

    dataset = MNIST_DATASET

    # LSTM & Q Learning
    IMAGE_SCALE = 20
    IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
    ##################

    # TEXT AND MODEL DETAILS:
    EMBEDDING_SIZE = 128
    SENTENCE_LENGTH = 12
    NUMBER_OF_SENTENCES = 1
    DICTIONARY_MAX_SIZE = 10000

    scenario_loader = load_scenario(scenario_size, batch_size, dataset, scenario_type)

    if (TRAIN):
        name = "TRAIN/" + name
    else:
        name = "TEST/" + name

    if (dataset == REUTERS_DATASET):
        name = "REUTERS/" + name
    elif (dataset == MNIST_DATASET):
        name = "MNIST/" + name
    elif (dataset == INH_DATASET):
        name = "INH/" + name
    else:
        name = "OMNIGLOT/" + name

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
    requests, accuracies, total_percentages, total_prediction_accuracies, total_accuracies = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
    for t in range(iterations - 1):
        r, a, r_p, a_p, a_c = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
        add_list(requests, r, dim=1)
        add_list(accuracies, a, dim=2)
        add_list(total_percentages, r_p, dim=1)
        add_list(total_prediction_accuracies, a_p, dim=1)
        add_list(total_accuracies, a_c, dim=1)

    divide_list(requests, iterations, dim=1)
    divide_list(accuracies, iterations, dim=2)
    divide_list(total_percentages, iterations, dim=1)
    divide_list(total_prediction_accuracies, iterations, dim=1)
    divide_list(total_accuracies, iterations, dim=1)


    bar_plot(requests, "Request Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size)
    bar_plot(accuracies, "Accuracy", name, ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], scenario_size)
    bar_plot(total_percentages, "Request Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size)
    bar_plot(total_accuracies, "Accuracy Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size, ylabel="% Accuracy")
    bar_plot(total_prediction_accuracies, "Prediction Accuracy Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size, ylabel="% Prediction Accuracy")








