# Python Libraries
from __future__ import print_function
import argparse
import os

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optimizers

# Utilities
from utils import tablewriter, resultwriter
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.arguments import parse_arguments
from utils.status import print_best_stats, StatusHandler
from utils.plot_handler import PlotHandler

# ML
import train
import test
import validation

# Setup
from setup.text import TextModelSetup, TextNetworkSetup
from setup.images import ImageModelSetup, ImageNetworkSetup

# ReinforcementLearning and Data-sets:
from reinforcement_utils.reinforcement import ReinforcementLearning
from reinforcement_utils.class_margin_sampling import ClassMarginSampler


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning NTM',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_arguments(parser)


if __name__ == '__main__':

    # Set up result directory
    result_directory = 'results/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Parse arguments from CLI
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.LSTM or args.NTM or args.LRUA, \
        "You need to chose a network architecture! type python main_text.py -h for help."

    assert args.INH or args.REUTERS or args.QA or args.MNIST or args.OMNIGLOT, \
        "You need to chose data-set type python main.py -h for help."

    # Seed the RNG for consistent results
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Collecting correct data-set
    dataset = 'data'

    # Collecting static image setup
    IMAGES = False
    image_setup = ImageModelSetup(args.margin_sampling, args.margin_size, args.margin_time)

    # Collecting static text setup
    TEXT = False
    text_setup = TextModelSetup(args.margin_sampling, args.margin_size, args.margin_time)

    # Setting up Class Margin Sampler
    class_margin_sampler = ClassMarginSampler(args, image_setup.train_transform,
                                              text_setup.SENTENCE_LENGTH)

    # Creating setup based on data-set
    if args.MNIST:
        dataset += '/images/mnist'
        IMAGES = True
    elif args.OMNIGLOT:
        dataset += '/images/omniglot'
        IMAGES = True
    elif args.INH:
        dataset += '/text/headlines'
        TEXT = True
    elif args.REUTERS:
        dataset += '/text/reuters'
        TEXT = True
    else:
        dataset += '/text/questions'
        TEXT = True

    if TEXT:
        setup = TextNetworkSetup(text_setup, dataset, args)
        train_loader, test_loader, q_network = \
            setup.train_loader, setup.test_loader, setup.q_network

    else:
        setup = ImageNetworkSetup(image_setup, dataset, args)
        train_loader, test_loader, q_network = \
            setup.train_loader, setup.test_loader, setup.q_network

    # Activating training on GPU
    if args.cuda:
        print("\n---Activating GPU Training---\n")
        q_network.cuda()

    # Loading Reinforcement module
    ReinforcementLearning = ReinforcementLearning(args.class_vector_size)

    # Initialize/Load Q Network & Statistics
    q_network, statistics = load_checkpoint(q_network, args)

    # Initialize Optimizer & Loss Function
    optimizer = optimizers.Adam(q_network.parameters())
    criterion = nn.MSELoss()

    training_status_handler = StatusHandler(args)

    # Training loop
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Train for one epoch
        train.train(q_network, epoch, optimizer, train_loader, args, ReinforcementLearning,
                    training_status_handler.episode, criterion, statistics, TEXT,
                    margin=args.margin_sampling, class_margin_sampler=class_margin_sampler)

        # Status update
        print("\n\n--- " + args.name + ": Training epoch " + str(epoch) + " ---\n\n")
        print_best_stats(statistics.statistics)
        training_status_handler.update_status(epoch, statistics)

        # Write results to file
        resultwriter.write_to_result_file(args.name, resultwriter.to_result_array(statistics.statistics))

        # Test for one epoch
        if epoch % 10 == 0:
            print(args)
            test.validate(q_network, epoch, test_loader, args,
                          ReinforcementLearning, statistics, TEXT)
            # Save best checkpoint
            if training_status_handler.update_best(statistics.statistics['total_test_reward']):
                statistics.update_state(q_network.state_dict())
                save_checkpoint(statistics.statistics, args.name, filename="best.pth.tar")

        # Save checkpoint
        if epoch % training_status_handler.SAVE == 0:
            statistics.update_state(q_network.state_dict())
            save_checkpoint(statistics.statistics, args.name)

        # Save backup checkpoint
        if epoch % training_status_handler.BACKUP == 0:
            statistics.update_state(q_network.state_dict())
            save_checkpoint(statistics.statistics, args.name, filename="backup.pth.tar")

    # Final checkpoint
    statistics.update_state(q_network.state_dict())
    save_checkpoint(statistics.statistics, args.name)

    # Update time elapsed and write meta-stats to file
    training_status_handler.finish(statistics.statistics['epoch'])

    # Initialize plot-handler & plot
    plot_handler = PlotHandler(statistics.statistics, args)
    plot_handler.plot()
    if args.margin_sampling:
        plot_handler.margin_plot()

    # Proceed to testing
    choice = input("\nChoices: "
                   "\n[1]: Test current model"
                   "\n[2]: Plot existing test results"
                   "\n[Any other key]: cancel\n\n")

    # Validate model
    if choice == "1":
        validation.validate_model(args, statistics, q_network, test_loader,
                                  train_loader, TEXT, ReinforcementLearning)
        statistics.update_state(q_network.state_dict())
        save_checkpoint(statistics.statistics, args.name, filename="testpoint.pth.tar")

    # Load previous validation
    elif choice == "2":
        _, statistics = load_checkpoint(q_network, args, best=True)

    if choice == "1" or choice == "2":
        # Scatter-plots
        plot_handler.scatter_plot(statistics.statistics)

        # Write result in tables
        tablewriter.write_stats(statistics.statistics['total_requests'],
                                statistics.statistics['total_prediction_accuracy'],
                                ReinforcementLearning.prediction_penalty, args.name + "/")
        tablewriter.write_stats(statistics.statistics['total_test_requests'],
                                statistics.statistics['total_test_prediction_accuracy'],
                                ReinforcementLearning.prediction_penalty, args.name + "/", test=True)

        # Write K-shot tables:
        tablewriter.print_k_shot_tables(statistics.statistics['test_pred_dict'],
                                        statistics.statistics['test_acc_dict'],
                                        statistics.statistics['test_req_dict'],
                                        "test", args.name + "/")
