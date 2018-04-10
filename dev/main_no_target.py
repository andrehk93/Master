### MISC ###
from __future__ import print_function
import argparse
import time
import os
import shutil
import copy

### PYTORCH STUFF ###
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### CLASSES ###
from utils import loss_plot, percent_scatterplot as scatterplot, transforms, loader, tablewriter
from reinforcement_utils.reinforcement import ReinforcementLearning as rl
from data.omniglot import OMNIGLOT

from models import reinforcement_models
from reinforcement_utils import train, test



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
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')

# Mini-batch size:
parser.add_argument('--mini-batch-size', type=int, default=50, metavar='N',
                    help='How many episodes to train on at a time (default: 1)')

# Episode size:
parser.add_argument('--episode-size', type=int, default=30, metavar='N',
                    help='input episode size for training (default: 30)')

# Epochs:
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
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
parser.add_argument('--load-checkpoint', default='pretrained/reinforced_ntm_notarget/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')

# Checkpoint Loader:
parser.add_argument('--load-best-checkpoint', default='pretrained/reinforced_ntm_notarget/best.pth.tar', type=str,
                    help='path to best checkpoint (default: none)')

# Network Name:
parser.add_argument('--name', default='reinforced_ntm_notarget', type=str,
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

def update_dicts(request_train_dict, accuracy_train_dict, req_dict, acc_dict):
    for key in acc_dict.keys():
        acc_dict[key].append(accuracy_train_dict[key])
        req_dict[key].append(request_train_dict[key])



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


    # Different Models:
    classes = args.class_vector_size

    LSTM = False
    NTM = True
    LRUA = False


    if LSTM:
        q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda, classes, IMAGE_SIZE)
    elif NTM:
        q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda, classes, IMAGE_SIZE)
    elif LRUA:
        q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda, classes, IMAGE_SIZE)

    # Modules:
    rl = rl(classes)

    if args.cuda:
        print("\n---Activating GPU Training---\n")
        q_network.cuda()

    ### PRINTING AMOUNT OF PARAMETERS ###
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in q_network.parameters()])))

    best_accuracy = 0.0

    req_dict = {1: [], 2: [], 5: [], 10: []}
    acc_dict = {1: [], 2: [], 5: [], 10: []}
    total_requests = []
    total_accuracy = []
    total_prediction_accuracy= []
    total_loss = []
    total_reward = []
    best = -30


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
            best = checkpoint['best']
            q_network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_checkpoint))

    print("Current best: ", best)

    ### WEIGHT OPTIMIZER ###
    optimizer = optim.Adam(q_network.parameters())
    criterion = nn.MSELoss()

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

            stats, request_train_dict, accuracy_train_dict = train.train(q_network, epoch, optimizer, train_loader, args, rl, episode, criterion)
            
            episode += args.batch_size

            update_dicts(request_train_dict, accuracy_train_dict, req_dict, acc_dict)

            # STATS:
            total_prediction_accuracy.append(stats[0])
            total_requests.append(stats[1])
            total_accuracy.append(stats[2])
            total_loss.append(stats[3])
            total_reward.append(stats[4])

            if (epoch % 1000 == 0):
                test.validate(q_network, epoch, optimizer, test_loader, args, rl, episode, criterion)


            ### SAVING THE BEST ALWAYS ###
            if (stats[4] >= best):
                best = stats[4]
                print("\n--- NEW BEST: ", best, "---\n")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'requests': req_dict,
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_requests': total_requests,
                    'tot_pred_acc': total_prediction_accuracy,
                    'tot_loss': total_loss,
                    'tot_reward': total_reward,
                    'best': best
                }, filename="best.pth.tar")

            ### SAVING CHECKPOINT ###
            if (epoch % SAVE == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'requests': req_dict,
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_requests': total_requests,
                    'tot_pred_acc': total_prediction_accuracy,
                    'tot_loss': total_loss,
                    'tot_reward': total_reward,
                    'best': best
                })

            ### ALSO SAVING BACKUP-CHECKPOINT ###
            if (epoch % BACKUP == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'episode': episode,
                    'state_dict': q_network.state_dict(),
                    'requests': req_dict,
                    'accuracy': acc_dict,
                    'tot_accuracy': total_accuracy,
                    'tot_requests': total_requests,
                    'tot_pred_acc': total_prediction_accuracy,
                    'tot_loss': total_loss,
                    'tot_reward': total_reward,
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
    loss_plot.plot([total_accuracy, total_prediction_accuracy, total_requests], ["Training Accuracy Percentage", "Training Prediction Accuracy",  "Training Requests Percentage"], "training_stats", args.name + "/", "Percentage")
    loss_plot.plot([total_loss], ["Training Loss"], "training_loss", args.name + "/", "Average Loss")
    loss_plot.plot([total_reward], ["Training Average Reward"], "training_reward", args.name + "/", "Average Reward")

    print("\n\n--- Training Done ---\n")
    val = input("\nProceed to testing? \n[Y/N]: ")

    test_network = False
    if (val.lower() == "y"):
        print("=> loading checkpoint '{}'".format(args.load_best_checkpoint))
        checkpoint = torch.load(args.load_best_checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_checkpoint, checkpoint['epoch']))
        test_accuracy = 0.0
        test_request = 0.0
        test_reward = 0.0
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

        print("\n--- Testing for", int(test_epochs), "epochs ---\n")

        # Validating on the test set for 100 epochs (5000 episodes):
        for epoch in range(args.epochs + 1, args.epochs + 1 + test_epochs):

            # Validate the model:
            prediction, accuracy, requests, reward, request_test_dict, accuracy_test_dict = test.validate(q_network, epoch, optimizer, test_loader, args, rl, episode, criterion)
            train_prediction, train_accuracy, train_requests, _, _, _ = test.validate(q_network, epoch, optimizer, train_loader, args, rl, episode, criterion)

            update_dicts(request_test_dict, accuracy_test_dict, req_dict, acc_dict)

            # Increment episode count:
            episode += args.batch_size

            # Statistics:
            test_accuracy += prediction
            test_request += requests
            test_reward += reward

            # For stat file:
            test_stats[0].append(prediction)
            test_stats[1].append(requests)
            training_stats[0].append(train_prediction)
            training_stats[1].append(train_requests)

            # Statistics:
            total_accuracy.append(prediction)
            total_requests.append(requests)
            total_reward.append(reward)

        test_accuracy = float(test_accuracy/test_epochs)
        test_request = float(test_request/test_epochs)
        test_reward = float(test_reward/test_epochs)

        # Printing:
        print("\nTesting Average Accuracy = ", str(test_accuracy) + " %")
        print("Testing Average Requests = ", str(test_request) + " %")
        print("Testing Average Reward = ", str(test_reward))
        loss_plot.plot([total_accuracy[args.epochs + 1:], total_requests[args.epochs + 1:]], ["Accuracy Percentage", "Requests Percentage"], "testing_stats", args.name + "/", "Percentage")
        loss_plot.plot([total_reward[args.epochs + 1:]], ["Average Reward"], "test_reward", args.name + "/", "Average Reward")


    scatterplot.plot(acc_dict, args.name + "/", args.batch_size, title="Prediction Accuracy")
    scatterplot.plot(req_dict, args.name + "/", args.batch_size, title="Total Requests")

    if (test_network):
        tablewriter.write_stats(training_stats[1], training_stats[0], rl.prediction_penalty, args.name + "/")
        tablewriter.write_stats(test_stats[1], test_stats[0], rl.prediction_penalty, args.name + "/", test=True)

