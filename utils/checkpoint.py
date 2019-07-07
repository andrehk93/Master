import os
import torch
from utils.statistics import Statistics


# Saves checkpoint to disk
def save_checkpoint(state, name, filename='checkpoint.pth.tar'):
    directory = "pretrained/" + name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    print("Checkpoint successfully saved!")


def load_checkpoint(q_network, args, best=False, test=False):
    statistics = Statistics()
    checkpoint_name = "pretrained/" + args.name + "/"
    if test:
        checkpoint_name += "testpoint.pth.tar"
    else:
        if best:
            checkpoint_name += "best.pth.tar"
        else:
            checkpoint_name += "checkpoint.pth.tar"

    # Loading checkpoint
    if checkpoint_name:
        if os.path.isfile(checkpoint_name):
            print("=> loading checkpoint '{}'".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_name)

            # Dictionaries
            statistics.statistics['test_pred_dict'] = checkpoint['test_pred_dict']
            statistics.statistics['test_train_pred_dict'] = checkpoint['test_train_pred_dict']

            statistics.statistics['acc_dict'] = checkpoint['acc_dict']
            statistics.statistics['test_acc_dict'] = checkpoint['test_acc_dict']
            statistics.statistics['test_train_acc_dict'] = checkpoint['test_train_acc_dict']

            statistics.statistics['req_dict'] = checkpoint['req_dict']
            statistics.statistics['test_req_dict'] = checkpoint['test_req_dict']
            statistics.statistics['test_train_req_dict'] = checkpoint['test_train_req_dict']

            # Requests
            statistics.statistics['requests'] = checkpoint['requests']
            statistics.statistics['test_requests'] = checkpoint['test_requests']
            statistics.statistics['test_train_requests'] = checkpoint['test_train_requests']
            statistics.statistics['training_test_requests'] = checkpoint['training_test_requests']

            # Accuracy
            statistics.statistics['accuracy'] = checkpoint['accuracy']
            statistics.statistics['test_accuracy'] = checkpoint['test_accuracy']
            statistics.statistics['test_train_accuracy'] = checkpoint['test_train_accuracy']
            statistics.statistics['training_test_accuracy'] = checkpoint['training_test_accuracy']

            # Prediction accuracy
            statistics.statistics['prediction_accuracy'] = checkpoint['prediction_accuracy']
            statistics.statistics['test_prediction_accuracy'] = checkpoint['test_prediction_accuracy']
            statistics.statistics['test_train_prediction_accuracy'] = checkpoint['test_train_prediction_accuracy']
            statistics.statistics['training_test_prediction_accuracy'] = checkpoint['training_test_prediction_accuracy']

            # Reward
            statistics.statistics['reward'] = checkpoint['reward']
            statistics.statistics['test_reward'] = checkpoint['test_reward']
            statistics.statistics['test_train_reward'] = checkpoint['test_train_reward']
            statistics.statistics['training_test_reward'] = checkpoint['training_test_reward']

            statistics.statistics['loss'] = checkpoint['loss']

            if args.margin_sampling:
                statistics.statistics['all_margins'] = checkpoint['all_margins']
                statistics.statistics['low_margins'] = checkpoint['low_margins']
                statistics.statistics['all_choices'] = checkpoint['all_choices']

            statistics.statistics['best'] = checkpoint['best']
            args.start_epoch = checkpoint['epoch']

            q_network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_name, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_name))

    return q_network, statistics
