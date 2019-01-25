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


def load_checkpoint(q_network, args, best=False):
    statistics = Statistics()
    checkpoint_name = "pretrained/" + args.name + "/"
    if best:
        checkpoint_name += "best.pth.tar"
    else:
        checkpoint_name += "checkpoint.pth.tar"

    # Loading checkpoint
    if checkpoint_name:
        if os.path.isfile(checkpoint_name):
            print("=> loading checkpoint '{}'".format(checkpoint_name))
            checkpoint = torch.load(checkpoint_name)
            args.start_epoch = checkpoint['epoch']
            statistics.statistics['req_dict'] = checkpoint['req_dict']
            statistics.statistics['acc_dict'] = checkpoint['acc_dict']
            statistics.statistics['total_requests'] = checkpoint['total_requests']
            statistics.statistics['total_accuracy'] = checkpoint['total_accuracy']
            statistics.statistics['total_prediction_accuracy'] = checkpoint['total_prediction_accuracy']
            statistics.statistics['total_loss'] = checkpoint['total_loss']
            statistics.statistics['total_reward'] = checkpoint['total_reward']

            # Test stats:
            statistics.statistics['total_test_requests'] = checkpoint['total_test_requests']
            statistics.statistics['total_test_accuracy'] = checkpoint['total_test_accuracy']
            statistics.statistics['total_test_prediction_accuracy'] = checkpoint['total_test_prediction_accuracy']
            statistics.statistics['total_test_reward'] = checkpoint['total_test_reward']
            statistics.statistics['test_pred_dict'] = checkpoint['test_pred_dict']
            statistics.statistics['test_acc_dict'] = checkpoint['test_acc_dict']
            statistics.statistics['test_req_dict'] = checkpoint['test_req_dict']

            if args.margin_sampling:
                statistics.statistics['all_margins'] = checkpoint['all_margins']
                statistics.statistics['low_margins'] = checkpoint['low_margins']
                statistics.statistics['all_choices'] = checkpoint['all_choices']
                statistics.statistics['best'] = checkpoint['best']
            q_network.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_name, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_name))

    return q_network, statistics
