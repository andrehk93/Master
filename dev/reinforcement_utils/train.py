import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy
import math

# Discount factor for future rewards, and Greedy Epsilon Constant:
START_EXP = 0.05
END_EXP = 0.05
STEP = 200
GAMMA = 0.5
#EPS = 0.2

def train(q_network, epoch, optimizer, train_loader, args, writer, reinforcement_learner, request_dict, accuracy_dict, episode, criterion):

    # Initialize training:
    q_network.train()

    # Collect a random batch:
    image_batch, label_batch = train_loader.__iter__().__next__()

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
    for v in request_dict.values():
        v.append([])
    for v in accuracy_dict.values():
        v.append([])

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
        prev_reads, prev_controller_state, prev_heads_states = hidden
        print("Heads = ", prev_heads_states)
        print("HEADS = ", prev_heads_states[0].volatile)
        input("PREDICT")
        if (args.cuda):
            q_values, hidden = q_network(Variable(state).type(torch.FloatTensor).cuda(), hidden)
        else:
            q_values, hidden = q_network(Variable(state).type(torch.FloatTensor), hidden)

        print("Q volatile = ", q_values.volatile)


        # Choosing the largest Q-values:
        q_network_actions = q_values.data.max(1)[1].view(args.batch_size)

        # Performing Epsilon Greedy Exploration:
        agent_actions = []

        EPS = END_EXP + (START_EXP - END_EXP) * math.exp((-1.0 * epoch) / STEP)
        for i in range(args.batch_size):

            # q_network choice:
            if (random.random() > EPS):
                agent_actions.append(q_network_actions[i])

            # Epsilong Greedy:
            else:
                epsilon_action = random.randint(0, 2)

                # Request:
                if (epsilon_action == 0):
                    agent_actions.append(args.class_vector_size)

                # Incorrect Prediction:
                elif (epsilon_action == 1):
                    wrong_label = random.randint(0, args.class_vector_size - 1)
                    while (wrong_label == episode_labels[i]):
                        wrong_label = random.randint(0, args.class_vector_size - 1)
                    agent_actions.append(wrong_label)

                # Correct Prediction:
                else:
                    agent_actions.append(episode_labels[i])
        
        # Collect rewards:
        rewards = reinforcement_learner.collect_reward_batch(agent_actions, one_hot_labels, args.batch_size)

        # Collecting average reward at time t:
        episode_reward += float(sum(rewards)/args.batch_size)

        # Just some statistics logging:
        for i in range(args.batch_size):

            true_label = episode_labels[i]

            # Statistics:
            reward = rewards[i]
            if (reward == reinforcement_learner.request_reward):
                episode_request += 1
                episode_predict += 1
                if (label_dict[i][true_label] in request_dict):
                    request_dict[label_dict[i][true_label]][-1].append(1)
                if (label_dict[i][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i][true_label]][-1].append(0)
            elif (reward == reinforcement_learner.prediction_reward):
                episode_correct += 1.0
                episode_predict += 1.0
                if (label_dict[i][true_label] in request_dict):
                    request_dict[label_dict[i][true_label]][-1].append(0)
                if (label_dict[i][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i][true_label]][-1].append(1)
            else:
                episode_predict += 1.0
                if (label_dict[i][true_label] in request_dict):
                    request_dict[label_dict[i][true_label]][-1].append(0)
                if (label_dict[i][true_label] in accuracy_dict):
                    accuracy_dict[label_dict[i][true_label]][-1].append(0)

        
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
            target_value = q_network(Variable(next_state, volatile=True), hidden)[0].max(1)[0]

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
        mse_loss = criterion(current_q_values, discounted_target_value).squeeze()
        print("Volatile mse loss = ", mse_loss.volatile)
        total_loss += mse_loss.data[0]

        print("mse_loss = ", mse_loss)
        print("loss = ", loss)

        # Backprop:
        loss += mse_loss
        
        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###

    # Meanify:
    #mean_loss = torch.mean(loss)

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
        predictions = accuracy_dict[key][-1]
        requests = request_dict[key][-1]
        
        accuracy = float(sum(predictions)/len(predictions))
        request_percentage = float(sum(requests)/len(requests))
        
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

    ### LOGGING TO TENSORBOARD ###
    data = {
        'training_total_requests': total_requests,
        'training_total_accuracy': total_accuracy,
        'training_total_loss': total_loss,
        'training_average_reward': total_reward
    }

    for tag, value in data.items():
        writer.scalar_summary(tag, value, epoch)
    ### DONE LOGGING ###

    return total_prediction_accuracy, total_requests, total_accuracy, total_loss, total_reward, request_dict, accuracy_dict








