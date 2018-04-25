
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy
import math
import copy


#Discount:
GAMMA = 0.5

def validate(q_network, epoch, optimizer, test_loader, args, reinforcement_learner, episode, criterion, nof_sentences):

    # Initialize training:
    q_network.eval()

    # Collect a random batch:
    text_batch, label_batch = test_loader.__iter__().__next__()

    # Episode Statistics:
    episode_correct = 0.0
    episode_predict = 0.0
    episode_request = 0.0
    episode_reward = 0.0

    # Create initial state:
    state = []
    label_dict = []
    for i in range(text_batch.size()[0]):
        state.append([0 for i in range(args.class_vector_size)])
        label_dict.append({})

    # Initialize q_network between each episode:
    hidden = q_network.reset_hidden(text_batch.size()[0])
    
    # Statistics again:    
    request_dict = {1: [], 2: [], 5: [], 10: []}
    accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    # EPISODE LOOP:
    for i_e in range(len(label_batch[0])):

        # Collecting timestep image/label batch:
        episode_labels, episode_texts = label_batch[:, i_e], text_batch[:, i_e]

        episode_texts = episode_texts.squeeze()

        # Tensoring the state:
        if (args.cuda):
            state = Variable(torch.FloatTensor(state)).cuda()
        else:
            state = Variable(torch.FloatTensor(state))

        # Create possible next states and update stats:
        one_hot_labels = []
        for i in range(text_batch.size()[0]):
            true_label = episode_labels[i]

            # Creating one hot labels:
            one_hot_labels.append([1 if j == true_label else 0 for j in range(args.class_vector_size)])

            # Logging statistics:
            if (true_label not in label_dict[i]):
                label_dict[i][true_label] = 1
            else:
                label_dict[i][true_label] += 1

        # Selecting an action to perform (Epsilon Greedy):
        
        if (args.cuda):
            q_values, hidden = q_network(Variable(episode_texts).type(torch.LongTensor).cuda(), hidden, class_vector=state, seq=text_batch.size()[0])
        else:
            q_values, hidden = q_network(Variable(episode_texts).type(torch.LongTensor), hidden, class_vector=state, seq=text_batch.size()[0])

        # Choosing the largest Q-values:
        q_network_actions = q_values.data.max(1)[1].view(text_batch.size()[0])

        # Collect Epsilon-Greedy actions:
        agent_actions = reinforcement_learner.select_actions(epoch, q_network_actions, text_batch.size()[0], args.class_vector_size, episode_labels)
        
        # Collect rewards:
        rewards = reinforcement_learner.collect_reward_batch(agent_actions, one_hot_labels, text_batch.size()[0])

        # Collecting average reward at time t over the batch:
        episode_reward += float(sum(rewards)/text_batch.size()[0])

        # Just some statistics logging:
        stats = update_dicts(text_batch.size()[0], episode_labels, rewards, reinforcement_learner, label_dict, request_dict, accuracy_dict)
        episode_predict += stats[0]
        episode_correct += stats[1]
        episode_request += stats[2]

        # Observe next state and images:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, one_hot_labels, text_batch.size()[0])

        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###

    for key in request_dict.keys():
        request_dict[key] = sum(request_dict[key])/len(request_dict[key]) 
        accuracy_dict[key] = sum(accuracy_dict[key])/len(accuracy_dict[key])


    ### VALIDATION BATCH DONE ###
    print("\n---Validation Statistics---\n")

    print("\n--- Epoch " + str(epoch) + ", Episode " + str(episode + i + 1) + " Statistics ---")
    print("Instance\tAccuracy\tRequests")       
    for key in accuracy_dict.keys():
        accuracy = accuracy_dict[key]
        request_percentage = request_dict[key]
        
        print("Instance " + str(key) + ":\t" + str(100.0*accuracy)[0:4] + " %" + "\t\t" + str(100.0*request_percentage)[0:4] + " %")
    

    # Even more status update:
    print("\n+------------------STATISTICS----------------------+")
    total_prediction_accuracy = float((100.0 * episode_correct) / max(1, episode_predict-episode_request))
    print("Batch Average Prediction Accuracy = " + str(total_prediction_accuracy)[:5] +  " %")
    total_accuracy = float((100.0 * episode_correct) / episode_predict)
    print("Batch Average Accuracy = " + str(total_accuracy)[:5] +  " %")
    total_requests = float((100.0 * episode_request) / (text_batch.size()[0]*args.episode_size))
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(episode_reward)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    return [total_prediction_accuracy, total_requests, total_accuracy, total_reward], request_dict, accuracy_dict


def update_dicts(batch_size, episode_labels, rewards, reinforcement_learner, label_dict, request_dict, accuracy_dict):
    predict = 0.0
    request = 0.0
    correct = 0.0
    for i in range(batch_size):
        true_label = episode_labels[i]

        # Statistics:
        reward = rewards[i]
        if (reward == reinforcement_learner.request_reward):
            request += 1.0
            predict += 1.0
            if (label_dict[i][true_label] in request_dict):
                request_dict[label_dict[i][true_label]].append(1)
            if (label_dict[i][true_label] in accuracy_dict):
                accuracy_dict[label_dict[i][true_label]].append(0)
        elif (reward == reinforcement_learner.prediction_reward):
            correct += 1.0
            predict += 1.0
            if (label_dict[i][true_label] in request_dict):
                request_dict[label_dict[i][true_label]].append(0)
            if (label_dict[i][true_label] in accuracy_dict):
                accuracy_dict[label_dict[i][true_label]].append(1)
        else:
            predict += 1.0
            if (label_dict[i][true_label] in request_dict):
                request_dict[label_dict[i][true_label]].append(0)
            if (label_dict[i][true_label] in accuracy_dict):
                accuracy_dict[label_dict[i][true_label]].append(0)
    
    

    return predict, correct, request




