### MISC ###
from __future__ import print_function
import argparse
import time
import os
import shutil
import copy
import math
import numpy as np

### PYTORCH STUFF ###
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### Datasets and UTILS ###
from utils.images import imageLoader as loader
#from utils.plot import loss_plot, percent_scatterplot as scatterplot
from utils import transforms, tablewriter
#from data.images.omniglot.omniglot_margin import OMNIGLOT_MARGIN
from data.images.omniglot.omniglot import OMNIGLOT
from data.images.mnist.MNIST import MNIST

# RL:
from models import reinforcement_models
from reinforcement_utils.images import train_nonsequence as train, test_nonsequence as test



### IMPORTANT NOTICE ###
"""
If train on 3 classes (or more):
    omniglot.py, line 64: img_classes = np.random.choice(3, self.classes, replace=False)

If train on whole dataset:
    omniglot.py, line 64: img_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)
"""


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning NTM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Batch size:
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 50)')

# Mini-batch size:
parser.add_argument('--mini-batch-size', type=int, default=16, metavar='N',
                    help='How many episodes to train on at a time (default: 1)')

# Mini-batch size:
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='How many episodes to test on at a time (default: 1)')

# Episode size:
parser.add_argument('--episode-size', type=int, default=30, metavar='N',
                    help='input episode size for training (default: 30)')

# Epochs:
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 2000)')

# Starting Epoch:
parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                    help='starting epoch (default: 1)')

# Nof Classes:
parser.add_argument('--class-vector-size', type=int, default=3, metavar='N',
                    help='Number of classes per episode (default: 3)')

# CUDA:
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')

# Checkpoint Loader:
parser.add_argument('--load-checkpoint', default='pretrained/baseline_lrua/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')

# Checkpoint Loader:
parser.add_argument('--load-best-checkpoint', default='pretrained/baseline_lrua/best.pth.tar', type=str,
                    help='path to best checkpoint (default: none)')

# Checkpoint Loader:
parser.add_argument('--load-test-checkpoint', default='pretrained/baseline_lrua/testpoint.pth.tar', type=str,
                    help='path to best checkpoint (default: none)')

# Network Name:
parser.add_argument('--name', default='baseline_lrua', type=str,
                    help='name of file')

# Seed:
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# Logging interval:
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')


# Saves checkpoint to disk
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    directory = "pretrained/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print("Checkpoint successfully saved!")

def update_dicts(accuracy_train_dict, acc_dict):
    for key in acc_dict.keys():
        acc_dict[key].append(accuracy_train_dict[key])


def print_best_stats(stats):
    # Static strings:
    stat_string = "\nBest Training Stats"
    table_string = "|\tAcc\t|"
    str_length = 16

    # Printing:
    print(stat_string)
    print("-"*str_length)
    print(table_string)
    print("-"*str_length)
    print("|\t" + str(stats[0])[0:4] + " %\t|")
    print("-"*str_length + "\n\n")



if __name__ == '__main__':

    ### SETTING UP RESULTS DIRECTORY ###
    result_directory = 'results/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    ### PARSING ARGUMENTS ###
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)
    #if args.cuda:
        #torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    ### PARAMETERS ###

    # LSTM & Q Learning
    IMAGE_SCALE = 20
    IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
    MARGIN_TIME = 10
    ##################

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])

    # abcde-vectors for classes (3125 different classes):
    multi_state = False
    state_size = 5

    if (multi_state):
        nof_classes = int(state_size*state_size)
        output_classes = nof_classes
    else:
        nof_classes = args.class_vector_size
        output_classes = nof_classes

    LSTM = False
    NTM = False
    LRUA = True

    if LSTM:
        q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda, nof_classes, IMAGE_SIZE, output_classes=output_classes, baseline=True)
    elif NTM:
        q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda, nof_classes, IMAGE_SIZE, output_classes=output_classes, baseline=True)
    elif LRUA:
        q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda, nof_classes, IMAGE_SIZE, output_classes=output_classes, baseline=True)

    root = 'data/images/omniglot'

    MNIST_TEST = False

    print("Loading trainingsets...")
    omniglot_loader = loader.OmniglotLoader(root, classify=False, partition=0.8, classes=True)
    train_loader = torch.utils.data.DataLoader(
        OMNIGLOT(root, train=True, transform=train_transform, download=True, omniglot_loader=omniglot_loader, classes=args.class_vector_size, episode_size=args.episode_size),
        batch_size=args.mini_batch_size, shuffle=True, **kwargs)
    print("Loading testset...")
    if (not MNIST_TEST):
        test_loader = torch.utils.data.DataLoader(
        OMNIGLOT(root, train=False, transform=test_transform, omniglot_loader=omniglot_loader, classes=args.class_vector_size, episode_size=args.episode_size, test=True),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_train_loader = torch.utils.data.DataLoader(
        OMNIGLOT(root, train=True, transform=test_transform, download=True, omniglot_loader=omniglot_loader, classes=args.class_vector_size, episode_size=args.episode_size, test=True),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        root = 'data/images/mnist'
        test_loader = torch.utils.data.DataLoader(
        MNIST(root, transform=test_transform, download=True, episode_size=args.episode_size, classes=args.class_vector_size),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    print("Done loading datasets!")

    if args.cuda:
        print("\n---Activating GPU Training---\n")
        q_network.cuda()

    ### PRINTING AMOUNT OF PARAMETERS ###
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in q_network.parameters()])))

    best_accuracy = 0.0

    acc_dict = {1: [], 2: [], 5: [], 10: []}
    total_accuracy = []
    total_loss = []
    best = -30


    ### LOADING PREVIOUS NETWORK ###
    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("=> loading checkpoint '{}'".format(args.load_checkpoint))
            checkpoint = torch.load(args.load_checkpoint)
            args.start_epoch = checkpoint['epoch']
            episode = checkpoint['episode']
            acc_dict = checkpoint['accuracy']
            total_accuracy = checkpoint['tot_accuracy']
            total_loss = checkpoint['tot_loss']
            best = checkpoint['best']
            q_network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_checkpoint))

    print("Current best: ", best)

    ### WEIGHT OPTIMIZER ###
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Init train stuff:
    epoch = 0
    episode = (args.start_epoch-1)*args.batch_size
    done = False
    start_time = time.time()

    # Constants:
    SAVE = 10
    BACKUP = 50

    while not done:
        ### TRAINING AND TESTING LOOP ###
        for epoch in range(args.start_epoch, args.epochs + 1):

            ### TRAINING ###
            print("\n\n--- Training epoch " + str(epoch) + " ---\n\n")

            if (len(total_accuracy) > 0):
                best_index = np.argmax(total_accuracy)
            
                print_best_stats([best, total_accuracy[best_index]])


            stats, accuracy_train_dict = train.train(q_network, epoch, optimizer, train_loader, args, episode, criterion, batch_size=args.batch_size)
            
            episode += args.batch_size

            update_dicts(accuracy_train_dict, acc_dict)

            # STATS:
            total_accuracy.append(stats[0])
            total_loss.append(stats[1])

            if (epoch % 50 == 0):
                test.validate(q_network, epoch, optimizer, test_loader, args, episode, criterion, batch_size=args.test_batch_size)


            ### SAVING THE BEST ALWAYS ###
            if (stats[0] >= best):
                best = stats[0]
                print("\n--- NEW BEST: ", best, "---\n")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_loss': total_loss,
                    'best': best
                }, filename="best.pth.tar")

            ### SAVING CHECKPOINT ###
            if (epoch % SAVE == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_loss': total_loss,
                    'best': best
                })

            ### ALSO SAVING BACKUP-CHECKPOINT ###
            if (epoch % BACKUP == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_loss': total_loss,
                    'best': best
                }, filename="backup.pth.tar")

        elapsed_time = time.time() - start_time
        print("ELAPSED TIME = " + str(elapsed_time) + " seconds")
        answer = input("How many more epochs to train: ")
        try:
            if int(answer) == 0:
                done = True
            else:
                args.start_epoch = args.epochs + 1
                args.epochs += int(answer)
        except:
            done = True

    
    # Plotting training accuracy:
    #loss_plot.plot([total_accuracy], ["Training Accuracy Percentage"], "training_stats", args.name + "/", "Percentage")
    #loss_plot.plot([total_loss], ["Training Loss"], "training_loss", args.name + "/", "Average Loss")

    print("\n\n--- Training Done ---\n")
    val = input("\nProceed to testing? \n[Y/N]: ")

    test_network = False
    if (val.lower() == "y"):
        print("=> loading checkpoint '{}'".format(args.load_best_checkpoint))
        checkpoint = torch.load(args.load_best_checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_best_checkpoint, checkpoint['epoch']))
        test_accuracy = 0.0
        test_network = True

        parsed = False
        while not parsed:
            test_epochs = input("\nHow many epochs to test? \n[0, N]: ")
            try:
                test_epochs = int(test_epochs)
                parsed = True
            except:
                parsed = False

        # Stats for tables:
        test_stats = [[], []]
        training_stats = [[], []]
        test_acc_dict = {1: [], 2: [], 5: [], 10: []}
        train_acc_dict = {1: [], 2: [], 5: [], 10: []}

        print("\n--- Testing for", int(test_epochs), "epochs ---\n")

        # Validating on the test set for 100 epochs (5000 episodes):
        for epoch in range(args.epochs + 1, args.epochs + 1 + test_epochs):

            # Validate the model:
            stats, accuracy_test_dict = test.validate(q_network, epoch, optimizer, test_loader, args, episode, criterion, batch_size=args.test_batch_size)
            if not MNIST_TEST:
                train_stats, train_accs = test.validate(q_network, epoch, optimizer, test_train_loader, args, episode, criterion, batch_size=args.test_batch_size)

            update_dicts(accuracy_test_dict, acc_dict)
            update_dicts(accuracy_test_dict, test_acc_dict)
            if not MNIST_TEST:
                update_dicts(train_accs, train_acc_dict)

            # Increment episode count:
            episode += args.batch_size

            # Statistics:
            test_accuracy += stats[0]

            # For stat file:
            test_stats[0].append(stats[0])
            if not MNIST_TEST:
                training_stats[0].append(train_stats[0])

            # Statistics:
            total_accuracy.append(stats[0])

        test_accuracy = float(test_accuracy/test_epochs)

        # Printing:
        print("\nTesting Average Accuracy = ", str(test_accuracy) + " %")
        #loss_plot.plot([total_accuracy[args.epochs + 1:]], ["Accuracy Percentage"], "testing_stats", args.name + "/", "Percentage")

    else:
        checkpoint = torch.load(args.load_test_checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_checkpoint, checkpoint['epoch']))
        if not MNIST_TEST:
            training_stats = checkpoint['training_stats']
        test_stats = checkpoint['test_stats']
        test_acc_dict = checkpoint['test_acc_dict']
        if not MNIST_TEST:
            train_acc_dict = checkpoint['train_acc_dict']

    #scatterplot.plot(acc_dict, args.name + "/", args.batch_size, title="Prediction Accuracy")

    if (test_network):
        if not MNIST_TEST:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'training_stats': training_stats,
                    'test_stats': test_stats,
                    'test_acc_dict': test_acc_dict,
                    'train_acc_dict': train_acc_dict,
                    'tot_loss': total_loss,
                    'best': best
                }, filename="testpoint.pth.tar")
        else:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'test_stats': test_stats,
                    'test_acc_dict': test_acc_dict,
                    'tot_loss': total_loss,
                    'best': best
                }, filename="testpoint_mnist.pth.tar")
    if not MNIST_TEST:
        tablewriter.write_baseline_stats(training_stats[0], args.name + "/")
    tablewriter.write_baseline_stats(test_stats[0], args.name + "/", test=True)
    tablewriter.print_k_shot_baseline_tables(test_acc_dict, "test", args.name + "/")
    if not MNIST_TEST:
        tablewriter.print_k_shot_baseline_tables(train_acc_dict, "train", args.name + "/")


