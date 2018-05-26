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

def train(q_network, epoch, optimizer, train_loader, args, reinforcement_learner, episode, criterion, nof_sentences, class_margin_sampler):

    # Set up CMS:
    q_network.eval()    

    # Collect a random batch:
    margin_batch, margin_label_batch = train_loader.__iter__().__next__()

    # Get margin classes:
    text_batch, label_batch = class_margin_sampler.sample_text(margin_batch, margin_label_batch, q_network, args.batch_size)

    # Initialize training:
    q_network.train()

    # Episode Statistics:
    episode_correct = 0.0
    episode_predict = 0.0
    episode_request = 0.0
    episode_reward = 0.0
    episode_loss = 0.0
    total_loss = 0.0

    # Create initial state:
    state = []
    label_dict = []
    for i in range(args.batch_size):
        state.append([0 for i in range(args.class_vector_size)])
        label_dict.append({})

    # Initialize q_network between each episode:
    hidden = q_network.reset_hidden()
    
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

        # Collecting timestep image/label batch:
        episode_labels, episode_texts = label_batch[i_e], text_batch[i_e]

        episode_texts = episode_texts.squeeze()
        
        # Tensoring the state:
        if (args.cuda):
            state = Variable(torch.FloatTensor(state)).cuda()
        else:
            state = Variable(torch.FloatTensor(state))

        # Create possible next states and update stats:
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
            q_values, hidden = q_network(Variable(episode_texts).type(torch.LongTensor).cuda(), hidden, class_vector=state, seq=episode_texts.size()[1])
        else:
            q_values, hidden = q_network(Variable(episode_texts).type(torch.LongTensor), hidden, class_vector=state, seq=episode_texts.size()[1])

        # Choosing the largest Q-values:
        q_network_actions = q_values.data.max(1)[1].view(args.batch_size)

        # Collect Epsilon-Greedy actions:
        agent_actions = reinforcement_learner.select_actions(epoch, q_network_actions, args.batch_size, args.class_vector_size, episode_labels)
        
        # Collect rewards:
        rewards = reinforcement_learner.collect_reward_batch(agent_actions, one_hot_labels, args.batch_size)

        # Collecting average reward at time t over the batch:
        episode_reward += float(sum(rewards)/args.batch_size)

        # Just some statistics logging:
        stats = update_dicts(args.batch_size, episode_labels, rewards, reinforcement_learner, label_dict, request_dict, accuracy_dict)
        episode_predict += stats[0]
        episode_correct += stats[1]
        episode_request += stats[2]

        # Observe next state and images:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, one_hot_labels, args.batch_size)

        # Tensoring the reward:
        if (args.cuda):
            rewards = Variable(torch.Tensor([rewards])).cuda()
        else:
            rewards = Variable(torch.Tensor([rewards]))

        # Need to collect the representative Q-values:
        if (args.cuda):
            agent_actions = Variable(torch.LongTensor(agent_actions)).cuda().unsqueeze(1)
        else:
            agent_actions = Variable(torch.LongTensor(agent_actions)).unsqueeze(1)
        current_q_values = q_values.gather(1, agent_actions)

        # Non-final state, collected by TARGET NETWORK:
        if (i_e < args.episode_size - 1):
            # Collect next state:
            next_episode_texts = text_batch[i_e + 1].squeeze()

            if (args.cuda):
                next_state = Variable(torch.FloatTensor(next_state_start), volatile=True).cuda()
            else:
                next_state = Variable(torch.FloatTensor(next_state_start), volatile=True)

            # Get target value for next state (SHOULD NOT COMPUTE GRADIENT!):
            if (args.cuda):
                target_value = q_network(Variable(next_episode_texts, volatile=True).cuda(), hidden, class_vector=next_state, read_only=True, seq=next_episode_texts.size()[1])[0].max(1)[0]
            else:
                target_value = q_network(Variable(next_episode_texts, volatile=True), hidden, class_vector=next_state, read_only=True, seq=next_episode_texts.size()[1])[0].max(1)[0]

            # Make it un-volatile again (So we actually can backpropagate):
            target_value.volatile = False

            # Discounting the next state + reward collected in this state:
            discounted_target_value = (GAMMA*target_value) + rewards

        # Final state:
        else:
            # As there is no next state, we only have the rewards:
            discounted_target_value = rewards

        discounted_target_value = discounted_target_value.view(args.batch_size, -1)

        # Calculating Bellman error:
        mse_loss = criterion(current_q_values, discounted_target_value)

        # Stats:
        total_loss += mse_loss.data[0]

        # Accumulate timestep loss:
        loss += mse_loss
        
        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###

    for key in request_dict.keys():
        request_dict[key] = sum(request_dict[key])/len(request_dict[key]) 
        accuracy_dict[key] = sum(accuracy_dict[key])/len(accuracy_dict[key])

    # Zero gradients:
    optimizer.zero_grad()

    # Backpropagating:
    loss.backward()

    # Step in SGD:
    optimizer.step()

    ### TRAINING BATCH DONE ###

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
    print("Batch Average Loss = " + str(total_loss)[:5])
    total_requests = float((100.0 * episode_request) / (args.batch_size*args.episode_size))
    print("Batch Average Requests = " + str(total_requests)[:5] + " %")
    total_reward = float(episode_reward)
    print("Batch Average Reward = " + str(total_reward)[:5])
    print("+--------------------------------------------------+\n")

    return [total_prediction_accuracy, total_requests, total_accuracy, total_loss, total_reward], request_dict, accuracy_dict


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




