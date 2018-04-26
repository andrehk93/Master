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
    image_batch, label_batch = scenario_loader.__iter__().__next__()

    # Create initial state:
    if (multi_state):
        state = []
        for i in range(batch_size):
            state.append([0 for i in range(state_size*state_size)])
    else:
        state = []
        for i in range(batch_size):
            state.append([0 for i in range(class_vector_size)])

    # Initialize model between each episode:
    hidden = model.reset_hidden(batch_size)

    # Statistics again:    
    requests = []
    accuracies = []

    if (multi_state):
        class_representations = get_multiclass_representations(batch_size, class_vector_size)


    # EPISODE LOOP:
    for i_e in range(len(label_batch)):
        episode_labels = label_batch[i_e]
        episode_images = image_batch[i_e]

        if (not multi_state):
            class_representations = get_singleclass_representations(batch_size, class_vector_size, episode_labels)

        # Tensoring the state:
        state = torch.FloatTensor(state)
        
        # Need to add image to the state vector:
        flat_images = episode_images.squeeze().view(batch_size, -1)

        # Concatenating possible labels/zero vector with image, to create the environment state:
        if (multi_state):
            state = state.view(batch_size, -1)
            state = torch.cat((state, flat_images), 1)
        else:
            state = torch.cat((state, flat_images), 1)


        # Selecting an action to perform (Epsilon Greedy):
        if (cuda):
            q_values, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor).cuda(), hidden)
        else:
            q_values, hidden = model(Variable(state, volatile=True).type(torch.FloatTensor), hidden)
        
        q_values = F.softmax(q_values, dim=1)
        """
        if (i_e == 0):
            print("Probabilities:\n")
            for p in range(len(q_values[0])):
                if (p < class_vector_size):
                    print("Class " + str(p) + ":\t" + str(q_values.data[0][p])[0:4])
                else:
                    print("Request:\t" + str(q_values.data[0][p])[0:4])
        """
        requests.append(torch.mean(q_values.data[:, -1]))
        accuracies.append(torch.mean(q_values.data[: ,: class_vector_size], 0))


        # Choosing the largest Q-values:
        model_actions = q_values.data.max(1)[1].view(batch_size)

        # NOT Performing Epsilon Greedy Exploration:
        agent_actions = model_actions

        # Observe next state and images:
        if (multi_state):
            next_state_start = reinforcement_learner.next_multistate_batch(agent_actions, class_representations, batch_size, episode_labels)
        else:
            next_state_start = reinforcement_learner.next_state_batch(agent_actions, class_representations, batch_size)
        
        # Update current state:
        state = next_state_start

        ### END TRAIN LOOP ###


    return requests, accuracies


def get_multiclass_representations(batch_size, classes):
    label_list = ['a', 'b', 'c', 'd', 'e']
    bits = np.array([np.array([np.array(np.random.choice(len(label_list), len(label_list), replace=True)) for c in range(classes)]) for b in range(batch_size)])
    one_hot_vectors = np.array([np.array([np.zeros((len(label_list), len(label_list))) for c in range(classes)]) for b in range(batch_size)])
    for b in range(batch_size):
        for c in range(classes):
            one_hot_vectors[b][c][np.arange(len(label_list)), bits[b][c]] = 1
    return one_hot_vectors

def get_singleclass_representations(batch_size, classes, episode_labels):
    one_hot_labels = []
    for b in range(batch_size):
        true_label = episode_labels.squeeze()[b]
        one_hot_labels.append([1 if j == true_label else 0 for j in range(classes)])

    return one_hot_labels