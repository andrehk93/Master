import os
import torch
from operator import add

from utils.plot import percent_scatterplot as scatterplot
from utils import tablewriter

model = "reinforced_lstm"
name = "avg_results" + "_" + model

checkpoints = ["reinforced_lstm_r2", "reinforced_lstm_r2_margin"]

#Scatterplots:
total_acc_dict = {}
total_req_dict = {}

# K-shot tables:
total_test_acc_dict = {}
total_test_req_dict = {}
total_train_acc_dict = {}
total_train_req_dict = {}

# Avg. accuracies:
total_training_stats = []
total_test_stats = []


# Iterating over all wanted checkpoints:
for c_point in checkpoints:
    checkpoint = torch.load("pretrained/" + c_point + "/testpoint.pth.tar")
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(c_point, checkpoint['epoch']))

    # For tables:
    training_stats = checkpoint['training_stats']
    test_stats = checkpoint['test_stats']

    if (len(total_training_stats) == 0):
        total_training_stats = training_stats
        total_test_stats = test_stats
    else:
        total_training_stats += training_stats
        total_test_stats += test_stats




    # For scatterplot:
    acc_dict = checkpoint['accuracy']
    req_dict = checkpoint['requests']

    for key in acc_dict.keys():
        if (key not in total_acc_dict):
            total_acc_dict[key] = acc_dict[key]
            total_req_dict[key] = req_dict[key]
        else:
            total_acc_dict[key] = list(map(add, total_acc_dict[key], acc_dict[key]))
            total_req_dict[key] = list(map(add, total_req_dict[key], req_dict[key]))




    # For K-shot tables:
    test_acc_dict = checkpoint['test_acc_dict']
    test_req_dict = checkpoint['test_req_dict']
    train_acc_dict = checkpoint['train_acc_dict']
    train_req_dict = checkpoint['train_req_dict']

    for key in test_acc_dict.keys():
        if (key not in total_test_acc_dict):
            total_test_acc_dict[key] = test_acc_dict[key]
            total_test_req_dict[key] = test_req_dict[key]
            total_train_acc_dict[key] = train_acc_dict[key]
            total_train_req_dict[key] = train_req_dict[key]
        else:
            total_test_acc_dict[key] = list(map(add, total_test_acc_dict[key], test_acc_dict[key]))
            total_test_req_dict[key] = list(map(add, total_test_req_dict[key], test_req_dict[key]))
            total_train_acc_dict[key] = list(map(add, total_train_acc_dict[key], train_acc_dict[key]))
            total_train_req_dict[key] = list(map(add, total_train_req_dict[key], train_req_dict[key]))


# Averaging accuracies:
for stat in range(len(total_training_stats)):
    for s in total_training_stats[stat]:
        s = float(s/len(checkpoints))
    for s in total_test_stats[stat]:
        s = float(s/len(checkpoints))


tablewriter.write_stats(total_training_stats[1], total_training_stats[0], -1.0, name + "/")
tablewriter.write_stats(total_test_stats[1], total_test_stats[0], -1.0, name + "/", test=True)



# Averaging scatterplot:
for key in total_acc_dict.keys():
    for a in range(len(total_acc_dict[key])):
        total_acc_dict[key][a] = float(total_acc_dict[key][a]/len(checkpoints))
    for r in range(len(total_req_dict[key])):
        total_req_dict[key][r] = float(total_req_dict[key][r]/len(checkpoints))

scatterplot.plot(total_acc_dict, name + "/", 50, title=model + "Prediction Accuracy")
scatterplot.plot(total_req_dict, name + "/", 50, title=model + "Total Requests")




# Averaging K-shot accuracies:
for key in total_test_acc_dict.keys():
    for a in range(len(total_test_acc_dict[key])):
        total_test_acc_dict[key][a] = float(total_test_acc_dict[key][a]/len(checkpoints))
    for a in range(len(total_train_acc_dict[key])):
        total_train_acc_dict[key][a] = float(total_train_acc_dict[key][a]/len(checkpoints))
    for r in range(len(total_test_req_dict[key])):
        total_test_req_dict[key][r] = float(total_test_req_dict[key][r]/len(checkpoints))
    for r in range(len(total_train_req_dict[key])):
        total_train_req_dict[key][r] = float(total_train_req_dict[key][r]/len(checkpoints))

tablewriter.print_k_shot_tables(total_test_acc_dict, total_test_req_dict, "test", name + "/")
tablewriter.print_k_shot_tables(total_train_acc_dict, total_train_req_dict, "train", name + "/")

