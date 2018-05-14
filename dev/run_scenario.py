import os
import matplotlib.pyplot as plt
import numpy as np
from reinforcement_utils.images import scenario, scenario2
from utils.images import imageLoader as loader
from data.images.omniglot.omniglot import OMNIGLOT
from data.images.mnist.MNIST import MNIST
from reinforcement_utils.reinforcement import ReinforcementLearning as rl
from utils import transforms
import torch
from torch.utils.data import DataLoader
from models import reinforcement_models


def load_scenario(size, batch_size):

    scenario_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])
    
    OMNIGLOT_DATASET = False

    if (OMNIGLOT_DATASET):
        root = 'data/images/omniglot'
        print("Loading scenario...")
        omniglot_loader = loader.OmniglotLoader(root, classify=False, partition=0.8, classes=True)
        scenario_loader = torch.utils.data.DataLoader(
            OMNIGLOT(root, train=True, transform=scenario_transform, download=True, omniglot_loader=omniglot_loader, episode_size=0, scenario=True, scenario_size=size, test=True),
            batch_size=batch_size, shuffle=True)
    else:
        root = 'data/images/mnist'

        print("Loading scenario...")
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

    plt.legend(loc=0)
    plt.title("RL Scenario")
    plt.xlabel("Time step")
    
    plt.ylim((0, 1))
    directory = "results/plots/"
    if not os.path.exists(directory + name):
        os.makedirs(directory + name)
    plt.savefig(directory + name + bar_type + "_" + str(size) + ".png")
    plt.show()


if __name__ == '__main__':

    name = 'reinforced_ntm_margin/'
    checkpoint = 'pretrained/' + name + 'best.pth.tar'

    batch_size = 32
    scenario_size = 5
    classes = 3
    cuda = False

    # LSTM & Q Learning
    IMAGE_SCALE = 20
    IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
    ##################

    scenario_loader = load_scenario(scenario_size, batch_size)

    LSTM = False
    NTM = True
    LRUA = False


    if LSTM:
        q_network = reinforcement_models.ReinforcedRNN(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)
    elif NTM:
        q_network = reinforcement_models.ReinforcedNTM(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)
    elif LRUA:
        q_network = reinforcement_models.ReinforcedLRUA(batch_size, cuda, classes, IMAGE_SIZE, output_classes=classes)

    if os.path.isfile(checkpoint):
        checkpoint = torch.load(checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))

    rl = rl(classes)

    iterations = 10

    # Scenario 1:
    requests, accuracies = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
    for t in range(iterations - 1):
        r, a = scenario.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
        add_list(requests, r, dim=1)
        add_list(accuracies, a, dim=2)

    divide_list(requests, iterations, dim=1)
    divide_list(accuracies, iterations, dim=2)


    bar_plot(requests, "Request", name, ["First Class", "Second Class"], scenario_size)
    bar_plot(accuracies, "Accuracy", name, ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], scenario_size)

    # Scenario 2:
    total_percentages = scenario2.run(q_network, scenario_loader, batch_size, rl, classes, cuda)

    for t in range(iterations - 1):
        request_percentage = scenario2.run(q_network, scenario_loader, batch_size, rl, classes, cuda)
        add_list(total_percentages, request_percentage, dim=1)

    divide_list(total_percentages, iterations, dim=1)

    bar_plot(total_percentages, "Request Percentage", name, ["First Class", "Second Class"], scenario_size)








