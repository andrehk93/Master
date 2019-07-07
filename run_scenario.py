import os
import matplotlib.pyplot as plt
import numpy as np

from reinforcement_utils.reinforcement import ReinforcementLearning as rl
import torch
from scenario import scenario

# Setup
from setup.text import TextModelSetup, TextNetworkSetup
from setup.images import ImageModelSetup, ImageNetworkSetup


def load_scenario():

    # Collecting correct data-set
    dataset_folder = 'data'

    # Collecting static image setup
    image_setup = ImageModelSetup(False, 0, 0)

    # Collecting static text setup
    text_dataset = False
    text_setup = TextModelSetup(False, 0, 0, args.embedding_size, args.sentence_length)

    # Creating setup based on data-set
    if dataset == 0:
        dataset_folder += '/images/omniglot'
    elif dataset == 1:
        dataset_folder += '/images/mnist'
    elif dataset == 2:
        dataset_folder += '/text/headlines'
        text_dataset = True
    elif dataset == 3:
        dataset_folder += '/text/reuters'
        text_dataset = True
    else:
        dataset_folder += '/text/questions'
        text_dataset = True

    if text_dataset:
        setup = TextNetworkSetup(text_setup, dataset_folder, args, scenario=True, scenario_setup=scenario_setup)
        return setup.scenario_loader, setup.q_network

    else:
        setup = ImageNetworkSetup(image_setup, dataset_folder, args, scenario=True, scenario_setup=scenario_setup)
        return setup.scenario_loader, setup.q_network


def add_list(list1, list2, dim=1):
    if dim == 1:
        for l in range(len(list2)):
            list1[l] += list2[l]
    elif dim == 2:
        for l in range(len(list2)):
            for i in range(len(list2[l])):
                list1[l][i] += list2[l][i]


def divide_list(list1, iterations, dim=1):
    if dim == 1:
        for l in range(len(list1)):
            list1[l] = float(list1[l]/iterations)
    elif dim == 2:
        for l in range(len(list1)):
            for i in range(len(list1[l])):
                list1[l][i] = float(list1[l][i]/iterations)


def bar_plot(lists, bar_type, name, labels, size, ylabel="", display=True):
    plot_list = []
    for i in range(len(lists)):
        if type(lists[i]) != type(0.2):
            plot_list.append([])
            for j in range(len(lists[i])):
                plot_list[i].append(abs(lists[i][j]))
        else:
            plot_list.append(abs(lists[i]))

    fig, ax = plt.subplots()
    ax.yaxis.grid(True)

    colors = ["red", "green", "blue", "yellow", "magenta", "white", "grey"]
    if "Percentage" in bar_type:
        if scenario_type == 0:
            x1 = np.arange(1, scenario_size + 1)
            plt.bar(x1, plot_list[0:scenario_size], color=colors[0], label=labels[0], edgecolor="black")
            for i in range(0, len(plot_list) - scenario_size):
                plt.bar(scenario_size + i + 1, plot_list[scenario_size + i], color=colors[i+1],
                        label=labels[i+1], edgecolor="black")
        elif scenario_type == 1:
            x_curr = 0
            for i in range(0, classes):
                if i == class_choice:
                    x = np.arange(i + 1, i + 1 + scenario_size)
                    x_curr += scenario_size
                    y = plot_list[i : i + scenario_size]
                else:
                    x = x_curr + 1
                    y = plot_list[x_curr]
                    x_curr += 1
                plt.bar(x, y, color=colors[i], label=labels[i], edgecolor="black")
        elif scenario_type == 2:
            for i in range(classes):
                x = np.arange(int(i*scenario_size), int((i+1)*scenario_size))
                plt.bar(x, plot_list[int(i*scenario_size) : int((i+1)*scenario_size)], color=colors[i], label=labels[i],
                        edgecolor="black")
        else:
            for i in range(0, classes):
                plt.bar(i + 1, plot_list[i], color=colors[i], label=labels[i], edgecolor="black")
            x1 = np.arange(classes + 1, classes + scenario_size)
            plt.bar(x1, plot_list[classes:], color=colors[class_choice], edgecolor="black")

        if len(ylabel) > 0:
            plt.ylabel(ylabel)
        else:
            plt.ylabel("% Label Requests")

    else:
        np_lists = np.array(plot_list).transpose()
        x = np.arange(1, len(plot_list) + 1)
        bottom_list = []
        for i in range(len(np_lists)):

            if len(bottom_list) == 0:
                plt.bar(x, np_lists[i], color=colors[i], label=labels[i], edgecolor="black")
                bottom_list = np_lists[i]
            else:
                plt.bar(x, np_lists[i], bottom=bottom_list, color=colors[i], label=labels[i], edgecolor="black")
                bottom_list += np_lists[i]
        plt.ylabel("Class Q-value")

    plt.legend(loc=9)
    plt.title("ReinforcementLearning Scenario")
    plt.xlabel("Time step")
    
    plt.ylim((0, 1))
    if not os.path.exists(directory + name):
        os.makedirs(directory + name)
    plt.savefig(directory + name + bar_type + "_" + str(size) + ".png")
    if display:
        plt.show()


def get_pretrained_models():
    result_folder = 'pretrained/'
    pretrained_models = []
    if os.path.exists(result_folder):
        for root, dirs, files in os.walk(result_folder):
            for dir in dirs:
                pretrained_models.append(dir)

    return pretrained_models


def get_selected_model():
    n = 25
    print('Models:\n')
    for i in range(n):
        print(str(i) + ': ' + pretrained_models[i] + '\n')
    selected_model = input('Select model to run scenario for [0-N]:\n')
    while True:
        try:
            selected_model_index = int(selected_model)
            if selected_model_index < len(pretrained_models):
                choice = pretrained_models[selected_model_index] + '/'
                model = 'lstm'
                if 'ntm' in choice:
                    model = 'ntm'
                elif 'lrua' in choice:
                    model = 'lrua'
                return choice, model
            else:
                print('Selected model was not in list!')
                selected_model = input('Select model to run scenario for [0-N]:\n')
        except:
            print('Selected model must be an integer!')
            selected_model = input('Select model to run scenario for [0-N]:\n')


def get_scenario():
    scenarios = ['Meta Scenario', 'Zero Shot Scenario', 'K Shot Scenario', 'One Shot Scenario', 'All scenarios']
    for i in range(len(scenarios)):
        print(str(i) + ': ' + scenarios[i] + '\n')
    return get_integer_input('Select scenario to run [0-N]:\n', 'scenario', len(scenarios))


def get_data_set():
    for i in range(len(data_sets)):
        print(str(i) + ': ' + data_sets[i] + '\n')
    return get_integer_input('Select data set to test on:\n', 'data set', len(data_sets))


def get_integer_input(msg, object, limit):
    selected_class = input(msg)
    while True:
        try:
            selected_class_int = int(selected_class)
            if selected_class_int < limit:
                return selected_class_int
            else:
                print('Selected ' + object + ' doesn\'t exist!')
                selected_class = input(msg)
        except:
            print('Selected ' + object + ' must be an integer!')
            selected_class = input(msg)


class Args:
    def __init__(self, setup):
        self.class_vector_size = setup['class_vector_size']
        self.episode_size = setup['episode_size']
        self.scenario_size = setup['scenario_size']
        self.GLOVE = setup['GLOVE']
        self.scenario_batch_size = setup['scenario_batch_size']
        self.batch_size = setup['batch_size']
        self.cuda = setup['cuda']
        self.train = setup['train']
        self.embedding_size = setup['embedding_size']
        self.sentence_length = setup['sentence_length']
        self.number_of_sentences = setup['number_of_sentences']
        self.LSTM = setup['LSTM']
        self.NTM = setup['NTM']
        self.LRUA = setup['LRUA']


if __name__ == '__main__':
    data_sets = ['OMNIGLOT', 'MNIST', 'INH', 'REUTERS', 'QA']
    directory = "results/plots/"
    nof_scenarios = 1

    pretrained_models = get_pretrained_models()
    chosen_model_to_train, model_type = get_selected_model()
    scenario_type = get_scenario()
    if scenario_type == 4:
        nof_scenarios = 4
    classes = get_integer_input('Set number of classes to train on (Can not be more than what the model is trained on)'
                                ' [2-5]:\n', 'number of classes', 5)
    class_choice = get_integer_input('Select class to test on [0-' + str(classes-1) + ']:\n', 'class', classes)
    batch_size = get_integer_input('Set batch size:\n', 'batch size', 256)
    scenario_size = get_integer_input('Set scenario size:\n', 'scenario', 30)
    dataset = get_data_set()

    model_choice = [0, 0, 0]
    if model_type == 'lstm':
        model_choice[0] = 1
    elif model_type == 'ntm':
        model_choice[1] = 1
    else:
        model_choice[2] = 1

    checkpoint = 'pretrained/' + chosen_model_to_train + 'best.pth.tar'

    # Training or test dataset
    TRAIN = True

    # TEXT AND MODEL DETAILS:
    EMBEDDING_SIZE = 100
    SENTENCE_LENGTH = 6
    NUMBER_OF_SENTENCES = 1
    DICTIONARY_MAX_SIZE = 10000

    argument_setup = {
        'class_vector_size': 3,
        'episode_size': 0,
        'scenario_size': scenario_size,
        'GLOVE': True,
        'scenario_batch_size': batch_size,
        'batch_size': batch_size,
        'cuda': False,
        'train': TRAIN,
        'embedding_size': EMBEDDING_SIZE,
        'sentence_length': SENTENCE_LENGTH,
        'number_of_sentences': NUMBER_OF_SENTENCES,
        'LSTM': model_choice[0],
        'NTM': model_choice[1],
        'LRUA': model_choice[2],
    }
    args = Args(argument_setup)
    iterations = 10

    scenario_setup = [scenario_size, scenario_type, class_choice, classes]

    scenario_loader, q_network = load_scenario()
    if os.path.isfile(checkpoint):
        checkpoint = torch.load(checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint))

    rl = rl(3)

    for i in range(nof_scenarios):

        if nof_scenarios > 1:
            print('\n--- Running scenario ' + str(i) + '---')
            scenario_loader.dataset.scenario_type = i
            scenario_type = i

        if scenario_type == 0:
            name = "meta/" + chosen_model_to_train
        elif scenario_type == 1:
            name = "zero_shot/c" + str(class_choice) + "_" + chosen_model_to_train
        elif scenario_type == 2:
            name = "k_shot/" + chosen_model_to_train
        elif scenario_type == 3:
            name = "one_shot/c" + str(class_choice) + "_" + chosen_model_to_train

        if TRAIN:
            name = "TRAIN/" + chosen_model_to_train
        else:
            name = "TEST/" + chosen_model_to_train

        name = data_sets[dataset] + '/' + name

        # Scenario t = 1:
        stats = scenario.run(q_network, scenario_loader, args, rl, dataset > 1)

        # Record stats
        for t in range(iterations - 1):
            new_stats = scenario.run(q_network, scenario_loader, args, rl, dataset > 1)
            add_list(stats[0], new_stats[0], dim=1)
            add_list(stats[1], new_stats[1], dim=2)
            add_list(stats[2], new_stats[2], dim=1)
            add_list(stats[3], new_stats[3], dim=1)
            add_list(stats[4], new_stats[4], dim=1)

        divide_list(stats[0], iterations, dim=1)
        divide_list(stats[1], iterations, dim=2)
        divide_list(stats[2], iterations, dim=1)
        divide_list(stats[3], iterations, dim=1)
        divide_list(stats[4], iterations, dim=1)

        bar_plot(stats[0], "Request Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size,
                 display=nof_scenarios == 1)
        bar_plot(stats[1], "Accuracy", name, ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], scenario_size,
                 display=nof_scenarios == 1)
        bar_plot(stats[2], "Request Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size,
                 display=nof_scenarios == 1)
        bar_plot(stats[3], "Accuracy Percentage", name, ["First Class", "Second Class", "Third Class"], scenario_size,
                 ylabel="% Accuracy", display=nof_scenarios == 1)
        bar_plot(stats[4], "Prediction Accuracy Percentage", name, ["First Class", "Second Class", "Third Class"],
                 scenario_size, ylabel="% Prediction Accuracy", display=nof_scenarios == 1)
    print('\nSuccessfully completed scenario(s)\nSaved to ' + directory + name)








