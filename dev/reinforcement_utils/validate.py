import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy
import math


GAMMA = 0.5
def validate(model, epoch, optimizer, test_loader, args, reinforcement_learner, episode):

    # Initialize training:
    model.eval()

    # Collect a random batch:
    image_batch, label_batch = test_loader.__iter__().__next__()

    # Episode Statistics:
    episode_correct = 0.0
    episode_predict = 0.0
    episode_request = 0.0
    episode_reward = 0.0
    episode_loss = 0.0

    # Create initial state:
    state = []
    label_dict = []
    for i in range(args.batch_size):
        state.append([0 for i in range(args.class_vector_size)])
        label_dict.append({})

    # Initialize model between each episode:
    hidden = model.reset_hidden()

    # Statistics again:    
    request_dict = {1: [], 2: [], 5: [], 10: []}
    accuracy_dict = {1: [], 2: [], 5: [], 10: []}

    # Placeholder for loss Variable:
    if (args.cuda):
        loss = Variable(torch.zeros(1).type(torch.Tensor)).cuda()
    else:
        loss = Variable(torch.zeros(1).type(torch.Tensor))

    # EPISODE LOOP:
    for i_e in range(len(label_batch)):
        episode_labels = label_batch[i_e]
        episode_images = image_batch[i_e]

        # Tensoring the state:
        state = torch.FloatTensor(state)
        
        # Need to add image to the state vector:
        flat_images = episode_images.squeeze().view(args.batch_size, -1)

        # Concatenating possible labels/zero vector with image, to create the environment state:
        state = torch.cat((state, flat_images), 1)
        
        one_hot_labels = []
        for i in range(args.batch_size):
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
            q_values, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor).cuda(), hidden)
        else:
            q_values, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor), hidden)

        # Choosing the largest Q-values:
        model_actions = q_values.data.max(1)[1].view(args.batch_size)

        # Performing Epsilon Greedy Exploration:
        agent_actions = model_actions
        
        # Collect rewards:
        rewards = reinforcement_learner.collect_reward_batch(agent_actions, one_hot_labels, args.batch_size)

        # Collecting average reward at time t:
        episode_reward += float(sum(rewards)/args.batch_size)

        # Update dicts and stats:
        stats = update_dicts(args.batch_size, episode_labels, rewards, reinforcement_learner, label_dict, request_dict, accuracy_dict)
        episode_predict += stats[0]
        episode_correct += stats[1]
        episode_request += stats[2]
        
        # Observe next state and images:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, one_hot_labels, args.batch_size)

        # Tensoring the reward:
        rewards = Variable(torch.Tensor([rewards]))

        # Need to collect the representative Q-values:
        agent_actions = Variable(torch.LongTensor(agent_actions)).unsqueeze(1)
        current_q_values = q_values.gather(1, agent_actions)

        # Non-final state:
        if (i_e < args.episode_size - 1):
            # Collect next image:
            next_flat_images = image_batch[i_e + 1].squeeze().view(args.batch_size, -1)

            # Create next state:
            next_state = torch.cat((torch.FloatTensor(next_state_start), next_flat_images), 1)

            # Get target value for next state:
            target_value = model(Variable(next_state, volatile=True), hidden)[0].max(1)[0]

            # Make it un-volatile again:
            target_value.volatile = False

            # Discounting the next state + reward collected in this state:
            discounted_target_value = (GAMMA*target_value) + rewards

        # Final state:
        else:
            # As there is no next state, we only have the rewards:
            discounted_target_value = rewards

        discounted_target_value = discounted_target_value.view(args.batch_size, -1)

        # Calculating Bellman error:
        bellman_loss = F.mse_loss(current_q_values, discounted_target_value)

        # Backprop:
        loss = loss.add(bellman_loss)
        
        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###

    print("\n---Validation Statistics---\n")

    # Turning stats into percentages:
    for key in accuracy_dict.keys():
        accuracy_dict[key] = float(sum(accuracy_dict[key])/max(1.0, len(accuracy_dict[key])))
        request_dict[key] = float(sum(request_dict[key])/max(1.0, len(request_dict[key])))


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
    total_loss = loss.data[0]
    print("Batch Average Loss = " + str(total_loss)[:5])
    total_requests = float((100.0 * episode_request) / (args.batch_size*args.episode_size))
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(episode_reward)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    return total_prediction_accuracy, total_requests, total_accuracy, total_reward, request_dict, accuracy_dict



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
