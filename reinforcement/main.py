### MISC ###
from __future__ import print_function
import argparse
import time
import os
import shutil

### PYTORCH STUFF ###
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### CLASSES ###
from utils import loss_plot, batch_scatterplot as scatterplot, transforms, loader, logger
from reinforcement import ReinforcementLearning as rl
from omniglot import OMNIGLOT

import model
import validate
import train_batch_sum as train


### IMPORTANT NOTICE ###
"""
If train on 3 classes (or more):
    omniglot.py, line 64: img_classes = np.random.choice(3, self.classes, replace=False)

If train on whole dataset:
    omniglot.py, line 64: img_classes = np.random.choice(len(self.train_labels), self.classes, replace=False)
"""


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning LSTM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Batch size:
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')

# Mini-batch size:
parser.add_argument('--mini-batch-size', type=int, default=50, metavar='N',
                    help='How many episodes to train on at a time (default: 1)')

# Episode size:
parser.add_argument('--episode-size', type=int, default=30, metavar='N',
                    help='input episode size for training (default: 30)')

# Nof. classes in an episode:
parser.add_argument('--class-vector-size', type=int, default=3, metavar='N',
                    help='input class vector size for training (default: 3)')

# Epochs:
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 2000)')

# Starting Epoch:
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='starting epoch (default: 1)')

# CUDA:
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')

# Checkpoint Loader:
parser.add_argument('--load-checkpoint', default='pretrained/truncated_singlesum/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')

# Network Name:
parser.add_argument('--name', default='truncated_singlesum', type=str,
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


# Writes a table to file:
def write_stats(requests, accuracy, penalty, folder, test=False):
    filename = "results/plots/" + str(folder) + "table_file.txt"
    stat_filename = "results/plots/" + str(folder) + "stat_file.txt"
    dimensions = [50, 20, 20]
    headers = ["Method", "Accuracy (%)", "Requests (%)"]
    method = "RL Prediction"
    if penalty == 0:
        method = "Supervised"
    else:
        if test:
            method += "(Rinc = " + str(penalty) + ") - Test"
        else:
            method += "(Rinc = " + str(penalty) + ") - Training"
    specs = [accuracy, requests]
    if (os.path.isfile(stat_filename)):
        stat_file = open(stat_filename, "a")
    else:
        stat_file = open(stat_filename, "w")
    stat_file.write(method + "\n")

    # Averaging over 20 episodes:
    for s in specs:
        length = min(20, len(s))
        average = float(sum(s[len(s) - length:])/length)
        stat_file.write(str(average)[0:4] + "\n")
    stat_file.close()

    # Reading from stat_file:
    stats = {}
    length = 3
    with open(stat_filename, "r") as statistics:
        i = 0
        current_key = ""
        for line in statistics:
            if (i == 0):
                if (line.rstrip() not in stats):
                    stats[line.rstrip()] = [[], []]
                current_key = line.rstrip()
            elif (i < length):
                stats[current_key][i-1].append(float(line.rstrip()))
            i += 1
            if (i == length):
                i = 0

    stat_list = []
    for k in stats.keys():
        stat_list.append([])
        stat_list[-1].append(k)
        for v in stats[k]:
            stat_list[-1].append(sum(v)/len(v))

    print(stats)
    print(stat_list)

    # Creating Line:
    table = ""
    line = "\n+"
    for d in dimensions:
        line += "-"*d + "+"
    line += "\n"

    if (os.path.isfile(filename)):
        file = open(filename, "a")

    else:
        file = open(filename, "w")

        # HEADER
        table += line
        header = "|"
        for i, d in enumerate(dimensions):
            header += int((d/2) - int(len(headers[i])/2))*" " + headers[i] + int((d/2) - int(len(headers[i])/2))*" " + "|"
        table += header
        table += line

    # BODY
    for stat in stat_list:
        table += "|"
        for i, d in enumerate(dimensions):
            table +=  int((d/2) - int(len(str(stat[i]))/2))*" " + str(stat[i]) + int((d/2) - int(len(str(stat[i]))/2))*" " + "|"
        # END
        table += line

    
    print(table)
    file.write(table)
    print("Table successfully written!")
    file.close()



if __name__ == '__main__':

    ### SETTING UP TENSORBOARD LOGGER ###
    result_directory = 'results/'
    log_directory = 'results/logs/'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = logger.Logger('results/logs/')

    ### PARSING ARGUMENTS ###
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)
    #if args.cuda:
        #torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    ### PARAMETERS ###

    # LSTM & Q Learning
    IMAGE_SCALE = 28
    IMAGE_SIZE = IMAGE_SCALE*IMAGE_SCALE
    HIDDEN_LAYERS = 1
    HIDDEN_NODES = 200
    OUTPUT_CLASSES = args.class_vector_size
    ##################

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])

    print("Loading trainingsets...")
    omniglot_loader = loader.OmniglotLoader('data/omniglot', classify=False, partition=0.8, classes=True)
    train_loader = torch.utils.data.DataLoader(
        OMNIGLOT('data/omniglot', train=True, transform=train_transform, download=True, omniglot_loader=omniglot_loader, batch_size=args.episode_size),
        batch_size=args.mini_batch_size, shuffle=True, **kwargs)
    print("Loading testset...")
    test_loader = torch.utils.data.DataLoader(
        OMNIGLOT('data/omniglot', train=False, transform=test_transform, omniglot_loader=omniglot_loader, batch_size=args.episode_size),
        batch_size=args.mini_batch_size, shuffle=True, **kwargs)
    print("Done loading datasets!")


    # LSTM Model:
    model = model.ReinforcedLSTM(IMAGE_SIZE, HIDDEN_NODES, HIDDEN_LAYERS, OUTPUT_CLASSES,
                                  args.batch_size, args.cuda)

    # Modules:
    rl = rl(OUTPUT_CLASSES)

    if args.cuda:
        print("\n---Activating GPU Training---\n")
        model.cuda()

    ### PRINTING AMOUNT OF PARAMETERS ###
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    best_accuracy = 0.0

    req_dict = {1: [], 2: [], 5: [], 10: []}
    acc_dict = {1: [], 2: [], 5: [], 10: []}
    test_req_dict = {1: [], 2: [], 5: [], 10: []}
    test_acc_dict = {1: [], 2: [], 5: [], 10: []}
    total_requests = []
    total_accuracy = []
    total_prediction_accuracy= []
    total_loss = []
    total_reward = []


    ### LOADING PREVIOUS NETWORK ###
    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("=> loading checkpoint '{}'".format(args.load_checkpoint))
            checkpoint = torch.load(args.load_checkpoint)
            args.start_epoch = checkpoint['epoch']
            episode = checkpoint['episode']
            req_dict = checkpoint['requests']
            acc_dict = checkpoint['accuracy']
            total_requests = checkpoint['tot_requests']
            total_accuracy = checkpoint['tot_accuracy']
            total_prediction_accuracy = checkpoint['tot_pred_acc']
            total_loss = checkpoint['tot_loss']
            total_reward = checkpoint['tot_reward']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_checkpoint))

    ### WEIGHT OPTIMIZER ###
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduce=True)
    epoch = 0
    episode = (args.start_epoch-1)*args.batch_size
    done = False
    start_time = time.time()
    while not done:
        ### TRAINING AND TESTING LOOP ###
        for epoch in range(args.start_epoch, args.epochs + 1):

            ### TRAINING ###
            print("\n\n--- Training epoch " + str(epoch) + " ---\n\n")
            prediction_accuracy, requests, accuracy, loss, reward, req_dict, acc_dict = train.train(model, epoch, optimizer, train_loader, args, logger, rl, req_dict, acc_dict, episode, criterion)
            episode += args.batch_size

            # STATS:
            total_prediction_accuracy.append(prediction_accuracy)
            total_accuracy.append(accuracy)
            total_requests.append(requests)
            total_loss.append(loss)
            total_reward.append(reward)

            if (epoch % 10000 == 0):
                validate.validate(model, epoch, optimizer, test_loader, args, logger, rl, test_req_dict, test_acc_dict, episode)

            ### SAVING CHECKPOINT ###
            save_checkpoint({
                'epoch': epoch + 1,
                'episode': episode,
                'state_dict': model.state_dict(),
                'requests': req_dict,
                'accuracy': acc_dict,
                'tot_accuracy': total_accuracy,
                'tot_requests': total_requests,
                'tot_pred_acc': total_prediction_accuracy,
                'tot_loss': total_loss,
                'tot_reward': total_reward
            })

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
    loss_plot.plot([total_accuracy, total_prediction_accuracy, total_requests], ["Training Accuracy Percentage", "Training Prediction Accuracy",  "Training Requests Percentage"], "training_stats", args.name + "/", "Percentage")
    loss_plot.plot([total_loss], ["Training Loss"], "training_loss", args.name + "/", "Average Loss")
    loss_plot.plot([total_reward], ["Training Average Reward"], "training_reward", args.name + "/", "Average Reward")

    print("\n\n--- Training Done ---\n")
    val = input("\nProceed to testing? \n[Y/N]: ")

    if (val.lower() == "y"):
        test_accuracy = 0.0
        test_request = 0.0
        test_reward = 0.0
        test_epochs = 20
        test_stats = [[], []]
        training_stats = [[], []]

        # Validating on the test set for 100 epochs (5000 episodes):
        print_stats = False
        first = True
        for epoch in range(args.epochs + 1, args.epochs + 1 + test_epochs):

            # Validate the model:
            prediction, requests, accuracy, reward, req_dict, acc_dict = validate.validate(model, epoch, optimizer, test_loader, args, logger, rl, req_dict, acc_dict, episode)
            prediction, train_requests, train_accuracy, _, _, _ = validate.validate(model, epoch, optimizer, train_loader, args, logger, rl, req_dict, acc_dict, episode)

            # Increment episode count:
            episode += args.batch_size

            # Statistics:
            test_accuracy += accuracy
            test_request += requests
            test_reward += reward

            # For stat file:
            test_stats[0].append(accuracy)
            test_stats[1].append(requests)
            training_stats[0].append(train_accuracy)
            training_stats[1].append(train_requests)

            # Statistics:
            total_accuracy.append(accuracy)
            total_requests.append(requests)
            total_reward.append(reward)

        test_accuracy = float(test_accuracy/test_epochs)
        test_request = float(test_request/test_epochs)
        test_reward = float(test_reward/test_epochs)

        # Printing:
        print("\nTesting Average Accuracy = ", str(test_accuracy) + " %")
        print("Testing Average Requests = ", str(test_request) + " %")
        print("Testing Average Reward = ", str(test_reward) + " %")
        loss_plot.plot([total_accuracy[args.epochs + 1:], total_requests[args.epochs + 1:]], ["Accuracy Percentage", "Requests Percentage"], "testing_stats", args.name + "/", "Percentage")
        loss_plot.plot([total_reward[args.epochs + 1:]], ["Average Reward"], "test_reward", args.name + "/", "Average Reward")


    scatterplot.plot(acc_dict, args.name + "/", title="Prediction Accuracy")
    scatterplot.plot(req_dict, args.name + "/", title="Total Requests")

    write_stats(training_stats[1], training_stats[0], rl.prediction_penalty, args.name + "/")
    write_stats(test_stats[1], test_stats[0], rl.prediction_penalty, args.name + "/", test=True)

