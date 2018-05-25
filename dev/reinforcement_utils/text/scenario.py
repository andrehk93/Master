import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import math


def run(model, scenario_loader, batch_size, reinforcement_learner, class_vector_size, cuda, multi_state=False, state_size=5):

    # Initialize training:
    model.eval()

    # Collect a random batch:
    text_batch, label_batch = scenario_loader.__iter__().__next__()

    # Create initial state:
    state = []
    for i in range(batch_size):
        state.append([0 for i in range(class_vector_size)])

    # Initialize model between each episode:
    hidden = model.reset_hidden(batch_size)

    # Statistics again:    
    requests = []
    accuracies = []
    request_percentage = []


    # EPISODE LOOP:
    for i_e in range(len(label_batch[0])):
        episode_labels = label_batch[: , i_e]
        episode_texts = text_batch[: , i_e]

        class_representations = get_singleclass_representations(batch_size, class_vector_size, episode_labels)

        # Tensoring the state:
        if (cuda):
            state = Variable(torch.FloatTensor(state)).cuda()
        else:
            state = Variable(torch.FloatTensor(state))
        
        # Need to add text to the state vector:
        episode_texts = episode_texts.squeeze()

        # Selecting an action to perform (Epsilon Greedy):
        if (cuda):
            q_values, hidden = model(Variable(episode_texts, volatile=True).type(torch.LongTensor).cuda(), hidden, class_vector=state, seq=text_batch.size()[0])
        else:
            q_values, hidden = model(Variable(episode_texts, volatile=True).type(torch.LongTensor), hidden, class_vector=state, seq=text_batch.size()[0])
        q_values = F.softmax(q_values, dim=1)

        requests.append(torch.mean(q_values.data[:, -1]))
        accuracies.append(torch.mean(q_values.data[: ,: class_vector_size], 0))

        # Choosing the largest Q-values:
        model_actions = q_values.data.max(1)[1].view(batch_size)

        # Logging action:
        reqs = 0
        total = 0
        for a in model_actions:
            if (a == class_vector_size):
                reqs += 1
            total += 1
            
        request_percentage.append(float(reqs/total))

        # NOT Performing Epsilon Greedy Exploration:
        agent_actions = model_actions

        # Observe next state and texts:
        next_state_start = reinforcement_learner.next_state_batch(agent_actions, class_representations, batch_size)
        
        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###

    return requests, accuracies, request_percentage


def get_singleclass_representations(batch_size, classes, episode_labels):
    one_hot_labels = []
    for b in range(batch_size):
        true_label = episode_labels.squeeze()[b]
        one_hot_labels.append([1 if j == true_label else 0 for j in range(classes)])

    return one_hot_labels