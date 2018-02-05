#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import os
import re
from utils import transforms, loader, loss_plot, batch_scatterplot as scatterplot, matrix_plot
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import attr
from attr import attrs, attrib, Factory
import argcomplete


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch import optim
import numpy as np


from ntm.aio import EncapsulatedNTM
from utils.omniglot import OMNIGLOT


LOGGER = logging.getLogger(__name__)


# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 1
CHECKPOINT_INTERVAL = 10


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss, last_accuracy):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.2f}, Accuracy: {:.2f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss, last_accuracy), end='')


# Saves checkpoint to disk
def save_checkpoint(state, name, filename='checkpoint.pth.tar'):
    directory = "pretrained/%s/" % (name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    print("Checkpoint successfully saved!")


def clip_grads(model):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def show_image(true_image, copy_image):
    copy_image = copy_image.view(28, 28)
    true_image = true_image.view(28, 28)
    fig = plt.figure()
    left = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(true_image.numpy(), cmap='gray')
    left.set_title("True Image")
    right = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(copy_image.numpy(), cmap='gray')
    right.set_title("Copied Image")
    plt.show()
    input("OK")


def evaluate(model, test_loader, criterion, params, args, iterations):
    batch_size = params.batch_size

    LOGGER.info("\Testing model for %d batches (batch_size=%d)...\n",
                len(test_loader), batch_size)

    test_accuracies, test_losses = [], []
    test_accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    start_ms = get_ms()
    for epoch in range(iterations):
        LOGGER.info("\nStarting training epoch %d...\n", epoch)

        # Collecting batch of images and labels:
        image_batch, label_batch = test_loader.__iter__().__next__()

        # Stats:
        episode_loss = 0.0
        episode_correct = 0.0

        # New sequence w/wiped memory:
        model.init_sequence(batch_size)

        # For matrix plot:
        predictions = []
        test_labels = []

        # Creating initial state:
        initial_state = []
        label_dict = []
        for i in range(params.batch_size):
            initial_state.append([0 for c in range(params.classes)])
            label_dict.append({})

        # To collect k-shot prediction accuracies:
        for v in test_accuracy_dict.values():
            v.append([])

        # Placeholder for loss variable:
        accum_loss = Variable(torch.zeros(params.batch_size).type(torch.FloatTensor), requires_grad=False)

        for i_e in range(params.episode_length):
            # Colelcting timestep images + labels:
            images_t, true_labels = image_batch[i_e], Variable(torch.LongTensor(label_batch[i_e]), volatile=True)

            # Flattening images:
            flattened_images_t = images_t.view(params.batch_size, -1)

            # Tensoring initial state:
            initial_state = torch.FloatTensor(initial_state)

            # Creating state:
            state = torch.cat((flattened_images_t, initial_state), 1)

            # Feed the batch:
            predicted_labels, previous_state = model(x=Variable(state, volatile=True))

            predictions.append([pred for pred in F.softmax(predicted_labels[0]).data])
            #print("TRUE LABEL = ", true_labels.data[0])
            #print("PREDICTION = ", predictions[-1])
            test_labels.append(true_labels.data[0])


            # Summing accuracies:
            predicted_indexes = predicted_labels.data.max(1)[1].view(params.batch_size)
            timestep_correct = sum(predicted_indexes.eq(true_labels.data))
            episode_correct += timestep_correct

            # Create one hot labels for next state:
            initial_state = []
            for b in range(params.batch_size):
                true_label = true_labels.data[b]
                initial_state.append([1 if c == true_label else 0 for c in range(params.classes)])

                # Logging statistics:
                if (true_label not in label_dict[b]):
                    label_dict[b][true_label] = 1
                else:
                    label_dict[b][true_label] += 1

            # Just some statistics logging:
            for b in range(params.batch_size):

                true_label = true_labels.data[b]
                prediction = predicted_indexes[b]

                if (label_dict[b][true_label] in test_accuracy_dict):
                    if (true_label == prediction):
                        test_accuracy_dict[label_dict[b][true_label]][-1].append(1)
                    else:
                        test_accuracy_dict[label_dict[b][true_label]][-1].append(0)

            # Collect the loss:
            loss = criterion(predicted_labels, true_labels)

            # Episode loss:
            episode_loss += torch.mean(loss).data[0]

            # Printing progress bar:
            progress_bar(i_e, 30, loss.data[0], float((timestep_correct*100.0)/(params.batch_size)))

        progress_bar(epoch, int(params.total_episodes/params.batch_size), episode_loss,
        float((episode_correct*100.0)/(params.batch_size*params.episode_length)))

        # Update stats:
        test_accuracies.append(float((100.0*episode_correct)/(params.batch_size*params.episode_length)))
        test_losses.append(episode_loss)

        print("\n\n--- Epoch " + str(epoch) + ", Episode " + str((epoch + 1)*params.batch_size) + " Statistics ---")
        print("Instance\tAccuracy")    
        for key in test_accuracy_dict.keys():
            predictions = test_accuracy_dict[key][-1]
            
            accuracy = float(sum(predictions)/len(predictions))
            print("Instance " + str(key) + ":\t" + str(100.0*accuracy)[0:4] + " %")
        print("---------------------------------------------\n")
    

    return predictions, test_labels


def train_model(model, params, train_loader, criterion, optimizer, args, accuracy_dict, losses, accuracies, start_epoch):
    batch_size = params.batch_size

    LOGGER.info("\nTraining model for %d batches (batch_size=%d)...\n",
                len(train_loader), batch_size)

    start_ms = get_ms()
    for epoch in range(start_epoch, int(params.total_episodes/params.batch_size)):
        LOGGER.info("\nStarting training epoch %d...\n", epoch)

        # Collecting batch of images and labels:
        image_batch, label_batch = train_loader.__iter__().__next__()

        # Stats:
        episode_loss = 0.0
        episode_correct = 0.0

        # New sequence w/wiped memory:
        model.init_sequence(batch_size)

        # Creating initial state:
        initial_state = []
        label_dict = []
        for i in range(params.batch_size):
            initial_state.append([0 for c in range(params.classes)])
            label_dict.append({})

        # To collect k-shot prediction accuracies:
        for v in accuracy_dict.values():
            v.append([])

        # Placeholder for loss variable:
        accum_loss = Variable(torch.zeros(params.batch_size).type(torch.FloatTensor))

        for i_e in range(params.episode_length):
            # Colelcting timestep images + labels:
            images_t, true_labels = image_batch[i_e], Variable(torch.LongTensor(label_batch[i_e]))

            # Flattening images:
            flattened_images_t = images_t.view(params.batch_size, -1)

            # Tensoring initial state:
            initial_state = torch.FloatTensor(initial_state)

            # Creating state:
            state = torch.cat((flattened_images_t, initial_state), 1)

            # Feed the sequence + delimiter
            predicted_labels, previous_state = model(x=Variable(state))

            # Summing accuracies:
            predicted_indexes = predicted_labels.data.max(1)[1].view(params.batch_size)
            timestep_correct = sum(predicted_indexes.eq(true_labels.data))
            episode_correct += timestep_correct

            # Create one hot labels for next state:
            initial_state = []
            for b in range(params.batch_size):
                true_label = true_labels.data[b]
                initial_state.append([1 if c == true_label else 0 for c in range(params.classes)])

                # Logging statistics:
                if (true_label not in label_dict[b]):
                    label_dict[b][true_label] = 1
                else:
                    label_dict[b][true_label] += 1

            # Just some statistics logging:
            for b in range(params.batch_size):

                true_label = true_labels.data[b]
                prediction = predicted_indexes[b]

                if (label_dict[b][true_label] in accuracy_dict):
                    if (true_label == prediction):
                        accuracy_dict[label_dict[b][true_label]][-1].append(1)
                    else:
                        accuracy_dict[label_dict[b][true_label]][-1].append(0)

            # Collect the loss:
            loss = criterion(predicted_labels, true_labels)

            # Adding to existing loss:
            accum_loss += loss

            #print("time = ", i_e, "\tAccum loss = ", accum_loss)
            #input("OK")

            # Episode loss:
            episode_loss += torch.mean(loss).data[0]

            # Printing progress bar:
            progress_bar(i_e, 30, loss.data[0], float((timestep_correct*100.0)/(params.batch_size)))

        progress_bar(epoch, int(params.total_episodes/params.batch_size), episode_loss,
        float((episode_correct*100.0)/(params.batch_size*params.episode_length)))

        # Get the mean loss over the whole batch of episodes:
        mean_loss = torch.mean(accum_loss)
        #print("Mean loss = ", mean_loss)

        # Zeroing gradients:
        optimizer.zero_grad()

        # Backpropagation:
        mean_loss.backward()

        # SGD Step:
        optimizer.step()

        # Update stats:
        accuracies.append(float((100.0*episode_correct)/(params.batch_size*params.episode_length)))
        losses.append(episode_loss)

        print("\n\n--- Epoch " + str(epoch) + ", Episode " + str((epoch + 1)*params.batch_size) + " Statistics ---")
        print("Instance\tAccuracy")    
        for key in accuracy_dict.keys():
            predictions = accuracy_dict[key][-1]
            
            accuracy = float(sum(predictions)/len(predictions))
            print("Instance " + str(key) + ":\t" + str(100.0*accuracy)[0:4] + " %")
        print("---------------------------------------------\n")

        # Report
        if epoch % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            prediction_accuracy = np.array(accuracies[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            LOGGER.info("Batch %d Loss: %.6f Accuracy: %.2f Time: %d ms/sequence",
                        epoch, mean_loss, prediction_accuracy, mean_time)
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (epoch % args.checkpoint_interval == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'episode': epoch*params.batch_size,
                'state_dict': model.state_dict(),
                'accuracy': accuracy_dict,
                'accuracies': accuracies,
                'losses': losses,
                'parameters': params
            }, args.name)
    """
    save_checkpoint({
        'epoch': epoch + 1,
        'episode': epoch*params.batch_size,
        'state_dict': model.state_dict(),
        'accuracy': accuracy_dict,
        'accuracies': accuracies,
        'losses': losses,
        'parameters': params
    }, args.name)
    """

    LOGGER.info("Training epoch done.")

    return accuracy_dict, losses, accuracies


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")

    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')

    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))

    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")

    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")

    # Checkpoint Loader:
    parser.add_argument('--load-checkpoint', default='pretrained/mann_batch/checkpoint.pth.tar', type=str,
                        help='path to latest checkpoint (default: none)')

    # Network Name:
    parser.add_argument('--name', default='mann_batch', type=str,
                        help='name of file')

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

@attrs
class Parameters(object):
    # PARAMS:
    name = attrib(default="copy-task")
    controller_size = attrib(default=200, convert=int)
    controller_layers = attrib(default=1, convert=int)
    classes = attrib(default=3, convert=int)
    total_episodes = attrib(default=100000, convert=int)
    episode_length = attrib(default=30, convert=int)
    image_size = attrib(default=400, convert=int)
    num_read_heads = attrib(default=4, convert=int)
    num_write_heads = attrib(default=1, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=40, convert=int)
    batch_size = attrib(default=16, convert=int)

def init_model(args):

    params = Parameters()
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    model = EncapsulatedNTM(params.image_size + params.classes, params.classes, params.controller_size, params.controller_layers, params.num_read_heads,
                            params.num_write_heads, params.memory_n, params.memory_m)

    # Can be collected by checkpoint:
    losses, accuracies = [], []
    accuracy_dict = {1: [], 2: [], 5: [], 10: []}
    start_epoch = 0

    ### LOADING PREVIOUS NETWORK ###
    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("=> loading checkpoint '{}'".format(args.load_checkpoint))
            checkpoint = torch.load(args.load_checkpoint)
            start_epoch = checkpoint['epoch']
            episode = checkpoint['episode']
            accuracy_dict = checkpoint['accuracy']
            accuracies = checkpoint['accuracies']
            losses = checkpoint['losses']
            params = checkpoint['parameters']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.load_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_checkpoint))


    return model, params, losses, accuracy_dict, accuracies, start_epoch


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():


    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model, parameters, losses, accuracy_dict, accuracies, start_epoch = init_model(args)

    IMAGE_SCALE = 20

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SCALE, IMAGE_SCALE)),
        transforms.ToTensor()
    ])

    omniglot_loader = loader.OmniglotLoader('data/omniglot', classify=False, partition=0.8, classes=True)
    train_loader = torch.utils.data.DataLoader(
        OMNIGLOT('data/omniglot', train=True, transform=transform, download=True, omniglot_loader=omniglot_loader, batch_size=parameters.episode_length),
        batch_size=parameters.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        OMNIGLOT('data/omniglot', train=False, transform=transform, download=False, omniglot_loader=omniglot_loader, batch_size=parameters.episode_length),
        batch_size=parameters.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters())

    init_logging()
    accuracy_dict, losses, accuracies = train_model(model, parameters, train_loader, criterion, optimizer, args, accuracy_dict, losses, accuracies, start_epoch)
    loss_plot.plot([accuracies], ["Training Accuracy Percentage"], "training_stats", args.name + "/", "Percentage")
    loss_plot.plot([losses], ["Training Loss"], "training_loss", args.name + "/", "Average Loss")
    predictions, labels = evaluate(model, test_loader, criterion, parameters, args, 10)
    matrix_plot.plot_matrix(predictions, labels, args.name + "/", title="matrix_plot_test")
    scatterplot.plot(accuracy_dict, args.name + "/", parameters.batch_size, title="Prediction Accuracy after Testing")



if __name__ == '__main__':
    main()