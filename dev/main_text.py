### MISC ###
from __future__ import print_function
import argparse
import time
import os
import shutil
import copy
import numpy as np

### PYTORCH STUFF ###
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### CLASSES ###
from utils.plot import loss_plot, percent_scatterplot as scatterplot
from utils.text import textLoader as loader
from utils import transforms, tablewriter

# RL and DATASETS:
from reinforcement_utils.reinforcement import ReinforcementLearning as rl
from reinforcement_utils.class_margin_sampling import ClassMarginSampler
from data.text.text_dataset import TEXT
from data.text.text_class_margin import TextMargin

from models import reinforcement_models
from reinforcement_utils.text import train_text as train, test_text as test



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning NTM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Batch size:
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 50)')

# Batch size:
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 50)')

# Episode size:
parser.add_argument('--episode-size', type=int, default=30, metavar='N',
                    help='input episode size for training (default: 30)')

# Epochs:
parser.add_argument('--epochs', type=int, default=30000, metavar='N',
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
parser.add_argument('--load-checkpoint', default='pretrained/headlines_lstm_standard/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')

# Checkpoint Loader:
parser.add_argument('--load-best-checkpoint', default='pretrained/headlines_lstm_standard/best.pth.tar', type=str,
                    help='path to best checkpoint (default: none)')

# Checkpoint Loader:
parser.add_argument('--load-test-checkpoint', default='pretrained/headlines_lstm_standard/testpoint.pth.tar', type=str,
                    help='path to best checkpoint (default: none)')

# Network Name:
parser.add_argument('--name', default='headlines_lstm_standard', type=str,
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

def print_best_stats(stats):
    # Static strings:
    stat_string = "\n\t\tBest Training Stats"
    table_string = "|\tReward\t|\tPred. Acc.\t|\t[Acc / Req]\t\t|"
    str_length = 72

    # Printing:
    print(stat_string)
    print("-"*str_length)
    print(table_string)
    print("-"*str_length)
    print("|\t" + str(stats[0])[0:4] + "\t|\t" + str(stats[1])[0:4] + " %\t\t|\t" + str(stats[2])[0:4] + " % / " + str(stats[3])[0:4] + " %\t\t|\t")
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

    # CLASS MARGIN SAMPLING:
    MARGIN_TIME = 4
    CMS = 3

    # TEXT AND MODEL DETAILS:
    EMBEDDING_SIZE = 128
    SENTENCE_LENGTH = 12
    NUMBER_OF_SENTENCES = 1
    DICTIONARY_MAX_SIZE = 10000

    class_margin_sampler = ClassMarginSampler(int(CMS*args.class_vector_size), args.class_vector_size, MARGIN_TIME, None, episode_size=args.episode_size, sentence_length=NUMBER_OF_SENTENCES, tensor_length=SENTENCE_LENGTH)

    # TRUE = REMOVE STOPWORDS:
    STOPWORDS = True

    # Different text-datasets:
    reuters = 'reuters'
    headlines = 'headlines'
    dataset = 'data/text/' + headlines
    ##################


    # Different Models:
    classes = args.class_vector_size

    LSTM = True
    NTM = False
    LRUA = False


    if LSTM:
        q_network = reinforcement_models.ReinforcedRNN(args.batch_size, args.cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)
    elif NTM:
        q_network = reinforcement_models.ReinforcedNTM(args.batch_size, args.cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)
    elif LRUA:
        q_network = reinforcement_models.ReinforcedLRUA(args.batch_size, args.cuda, classes, EMBEDDING_SIZE, embedding=True, dict_size=DICTIONARY_MAX_SIZE)

    print("Loading datasets...")
    text_loader = loader.TextLoader(dataset, classify=False, partition=0.8, classes=True,\
                                        dictionary_max_size=DICTIONARY_MAX_SIZE, sentence_length=SENTENCE_LENGTH, stopwords=STOPWORDS)

    # MARGIN SAMPLING:
    """
    train_loader = torch.utils.data.DataLoader(
        TextMargin(dataset, train=True, download=True, data_loader=text_loader, classes=args.class_vector_size, episode_size=args.episode_size, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH, margin_time=MARGIN_TIME, CMS=CMS, q_network=q_network),
                batch_size=args.batch_size, shuffle=True, **kwargs)

    # NO MARGIN:
    """
    train_loader = torch.utils.data.DataLoader(
        TEXT(dataset, train=True, download=True, data_loader=text_loader, classes=args.class_vector_size, episode_size=args.episode_size, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    
    
    print("Loading testset...")
    test_loader = torch.utils.data.DataLoader(
        TEXT(dataset, train=False, data_loader=text_loader, classes=args.class_vector_size, episode_size=args.episode_size, tensor_length=NUMBER_OF_SENTENCES, sentence_length=SENTENCE_LENGTH),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print("Done loading datasets!")

    # Modules:
    rl = rl(classes)

    if args.cuda:
        print("\n---Activating GPU Training---\n")
        q_network.cuda()

    ### PRINTING AMOUNT OF PARAMETERS ###
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in q_network.parameters()])))


    # Parameters that potentially can be collected from checkpoint:
    best_accuracy = 0.0
    req_dict = {1: [], 2: [], 5: [], 10: []}
    acc_dict = {1: [], 2: [], 5: [], 10: []}
    total_requests = []
    total_accuracy = []
    total_prediction_accuracy= []
    total_loss = []
    total_reward = []
    all_margins = []
    best = -30
    start_time = time.time()


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
            start_time -= checkpoint['time']
            all_margins = checkpoint['all_margins']
            best = checkpoint['best']
            q_network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_checkpoint))

    print("Current best: ", best)

    ### WEIGHT OPTIMIZER & CRITERION FOR LOSS ###
    optimizer = optim.Adam(q_network.parameters())
    criterion = nn.MSELoss()

    # Init train stats:
    epoch = 0
    episode = (args.start_epoch-1)*args.batch_size
    done = False
    
    # Constants for checkpoint saving:
    SAVE = 10
    BACKUP = 50

    while not done:
        ### TRAINING AND TESTING LOOP ###
        for epoch in range(args.start_epoch, args.epochs + 1):

            ### TRAINING ###
            print("\n\n--- " + args.name + ": Training epoch " + str(epoch) + " ---\n")
            
            if (len(total_reward) > 0):
                best_index = np.argmax(total_reward)
            
                print_best_stats([best, total_prediction_accuracy[best_index], total_accuracy[best_index], total_requests[best_index]])

            stats, request_train_dict, accuracy_train_dict = train.train(q_network, epoch, optimizer, train_loader, args, rl, episode, criterion, NUMBER_OF_SENTENCES, class_margin_sampler)
            
            episode += args.batch_size

            update_dicts(request_train_dict, accuracy_train_dict, req_dict, acc_dict)

            # STATS:
            total_prediction_accuracy.append(stats[0])
            total_requests.append(stats[1])
            total_accuracy.append(stats[2])
            total_loss.append(stats[3])
            total_reward.append(stats[4])

            if (epoch % 50 == 0):
                test.validate(q_network, epoch, optimizer, test_loader, args, rl, episode, criterion, NUMBER_OF_SENTENCES)


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
                    'all_margins': class_margin_sampler.all_margins,
                    'best': best,
                    'time': time.time() - start_time
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
                    'all_margins': class_margin_sampler.all_margins,
                    'best': best,
                    'time': time.time() - start_time
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
                    'all_margins': class_margin_sampler.all_margins,
                    'best': best,
                    'time': time.time() - start_time
                }, filename="backup.pth.tar")

        elapsed_time = time.time() - start_time
        print("\n--- Total Training Time ---\nT = " + str(elapsed_time) + " seconds\n")
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
    loss_plot.plot([all_margins], ["Avg. Sample Margin"], "sample_margin", args.name + "/", "Avg. Sample Margin", avg=5, batch_size=args.batch_size)

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
        test_acc_dict = {1: [], 2: [], 5: [], 10: []}
        test_req_dict = {1: [], 2: [], 5: [], 10: []}
        test_pred_dict = {1: [], 2: [], 5: [], 10: []}
        train_acc_dict = {1: [], 2: [], 5: [], 10: []}
        train_req_dict = {1: [], 2: [], 5: [], 10: []}
        train_pred_dict = {1: [], 2: [], 5: [], 10: []}

        print("\n--- Testing for", int(test_epochs), "epochs ---\n")

        # Validating on the test set for 100 epochs (5000 episodes):
        for epoch in range(args.epochs + 1, args.epochs + 1 + test_epochs):

            # Validate the model:
            stats, request_test_dict, accuracy_test_dict, prediction_accuracy_test_dict = test.validate(q_network, epoch, optimizer, test_loader, args, rl, episode, criterion, NUMBER_OF_SENTENCES)
            train_stats, train_reqs, train_accs, prediction_accuracy_train_dict = test.validate(q_network, epoch, optimizer, train_loader, args, rl, episode, criterion, NUMBER_OF_SENTENCES)

            update_dicts(request_test_dict, accuracy_test_dict, req_dict, acc_dict)
            update_dicts(request_test_dict, accuracy_test_dict, test_req_dict, test_acc_dict)
            update_dicts(train_reqs, train_accs, train_req_dict, train_acc_dict)
            update_dicts(prediction_accuracy_test_dict, prediction_accuracy_train_dict, test_pred_dict, train_pred_dict)

            # Increment episode count:
            episode += args.batch_size

            # Statistics:
            test_accuracy += stats[0]
            test_request += stats[1]
            test_reward += stats[3]

            # For stat file:
            test_stats[0].append(stats[0])
            test_stats[1].append(stats[1])
            training_stats[0].append(train_stats[0])
            training_stats[1].append(train_stats[1])

            # Statistics:
            total_accuracy.append(stats[0])
            total_requests.append(stats[1])
            total_reward.append(stats[3])

        test_accuracy = float(test_accuracy/test_epochs)
        test_request = float(test_request/test_epochs)
        test_reward = float(test_reward/test_epochs)

        # Printing:
        print("\nTesting Average Accuracy = ", str(test_accuracy) + " %")
        print("Testing Average Requests = ", str(test_request) + " %")
        print("Testing Average Reward = ", str(test_reward))

        # Saving:
        save_checkpoint({
            'epoch': epoch + 1,
            'episode': episode,
            'state_dict': q_network.state_dict(),
            'requests': req_dict,
            'accuracy': acc_dict,
            'tot_accuracy': total_accuracy,
            'tot_requests': total_requests,
            'tot_pred_acc': total_prediction_accuracy,
            'training_stats': training_stats,
            'test_stats': test_stats,
            'test_acc_dict': test_acc_dict,
            'test_req_dict': test_req_dict,
            'test_pred_dict': test_pred_dict,
            'train_acc_dict': train_acc_dict,
            'train_req_dict': train_req_dict,
            'train_pred_dict': train_pred_dict,
            'tot_loss': total_loss,
            'tot_reward': total_reward,
            'all_margins': class_margin_sampler.all_margins,
            'best': best
        }, filename="testpoint.pth.tar")
        
        loss_plot.plot([total_accuracy[args.epochs + 1:], total_requests[args.epochs + 1:]], ["Accuracy Percentage", "Requests Percentage"], "testing_stats", args.name + "/", "Percentage")
        loss_plot.plot([total_reward[args.epochs + 1:]], ["Average Reward"], "test_reward", args.name + "/", "Average Reward")
        
    else:
        checkpoint = torch.load(args.load_test_checkpoint)
        q_network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_checkpoint, checkpoint['epoch']))
        training_stats = checkpoint['training_stats']
        test_stats = checkpoint['test_stats']
        test_acc_dict = checkpoint['test_acc_dict']
        test_req_dict = checkpoint['test_req_dict']
        train_acc_dict = checkpoint['train_acc_dict']
        train_req_dict = checkpoint['train_req_dict']
        test_pred_dict = checkpoint['test_pred_dict']
        train_pred_dict = checkpoint['train_pred_dict']
        acc_dict = checkpoint['accuracy']
        req_dict = checkpoint['requests']

    scatterplot.plot(acc_dict, args.name + "/", args.batch_size, title="Prediction Accuracy", max_epochs=args.epochs)
    scatterplot.plot(req_dict, args.name + "/", args.batch_size, title="Total Requests", max_epochs=args.epochs)
    scatterplot.plot(acc_dict, args.name + "/", args.batch_size, title="Prediction Accuracy", zoom=True, max_epochs=args.epochs)
    scatterplot.plot(req_dict, args.name + "/", args.batch_size, title="Total Requests", zoom=True, max_epochs=args.epochs)

        

    tablewriter.write_stats(training_stats[1], training_stats[0], rl.prediction_penalty, args.name + "/")
    tablewriter.write_stats(test_stats[1], test_stats[0], rl.prediction_penalty, args.name + "/", test=True)
    tablewriter.print_k_shot_tables(test_pred_dict, test_acc_dict, test_req_dict, "test", args.name + "/")
    tablewriter.print_k_shot_tables(train_pred_dict, train_acc_dict, train_req_dict, "train", args.name + "/")

