import random
import math
import torch
from torch.autograd import Variable
from transition import Transition
from torch import FloatTensor
import numpy as np


class ReinforcementLearning():

    EPS = 0.05


    def __init__(self, classes):
        self.classes = classes
        self.request_reward = -0.1
        self.prediction_reward = 1.0
        self.prediction_penalty = -1.0

    def select_action(self, model, state, label, hidden, gpu, episode):
        sample = random.random()

        # Epsilon Greedy Exploration:

        if sample > self.EPS:
            # Argmax_a Q(s)
            # Action is the index of the most significant Q-value:
            if (gpu):
                q_value, hidden = model(Variable(state, volatile=False).type(torch.FloatTensor).cuda(), hidden)
            else:
                q_value, hidden = model(Variable(state, volatile=False).type(torch.FloatTensor), hidden)
            action = q_value.data.max(1)[1].view(1)[0]
            #print("GATHERED = ", q_value.squeeze()[action])
            return torch.LongTensor([action]).view(1, 1), hidden, q_value.squeeze()[action]

        # Doing a random action:
        else:

            if (gpu):
                q_value, hidden = model(Variable(state, volatile=False).type(torch.FloatTensor).cuda(), hidden)
            else:
                q_value, hidden = model(Variable(state, volatile=False).type(torch.FloatTensor), hidden)
            # Either the correct label, incorrect label, or request label:
            action = random.randint(0, 2)
            # Requesting label:
            if (action == 0):
                return torch.LongTensor([self.classes]).view(1, 1), hidden, q_value.squeeze()[self.classes]
            # Wrong Prediction:
            elif (action == 1):
                wrong_label = random.randint(0, self.classes - 1)
                while (wrong_label == np.argmax(label)):
                    wrong_label = random.randint(0, self.classes - 1)
                return torch.LongTensor([wrong_label]).view(1, 1), hidden, q_value.squeeze()[wrong_label]
            # Correct Prediction:
            else:
                return torch.LongTensor([int(np.argmax(label))]).view(1, 1), hidden, q_value.squeeze()[int(np.argmax(label))]



    def collect_reward(self, action, labels):
        action = action.squeeze()
        # Last bit determines if the agent predicts or request the label:
        if (int(action[0]) == self.classes):
            return self.request_reward
        else:
            if (np.argmax(labels) == action[0]):
                return self.prediction_reward
            else:
                return self.prediction_penalty

    def collect_reward_batch(self, actions, labels, batch_size):
        actions = actions
        # Last bit determines if the agent predicts or request the label:
        rewards = []
        for i in range(batch_size):
            if (int(actions[i]) == self.classes):
                rewards.append(self.request_reward)
            else:
                if (np.argmax(labels[i]) == int(actions[i])):
                    rewards.append(self.prediction_reward)
                else:
                    rewards.append(self.prediction_penalty)
        return rewards

    def next_state(self, action, one_hot_label):
        action = action.squeeze()
        # Requesting the label:
        if (int(action[0]) == self.classes):
            return one_hot_label
        # Predicting (Thus returning a 0-vector to the input):
        else:
            return [0 for i in range(self.classes)]

    def next_state_batch(self, actions, one_hot_labels, batch_size):
        actions = actions
        states = []
        for i in range(batch_size):
            # Requesting the label:
            if (int(actions[i]) == self.classes):
                states.append(one_hot_labels[i])
            # Predicting (Thus returning a 0-vector to the input):
            else:
                states.append([0 for i in range(self.classes)])
        return states


# Stored observed transitions used for optimizing the model:
class ReplayMemory(object):
    
    def __init__(self, capacity, episode_capacity):
        self.capacity = capacity
        self.episode_capacity = episode_capacity
        self.memory = []
        self.episode = []
        self.position = 0
        self.episode_position = 0
        self.trace_length = 30

    def push(self, episode):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def push_transition(self, *args):
        if len(self.episode) < self.episode_capacity:
            self.episode.append(None)
        self.episode[self.episode_position] = Transition(*args)
        self.episode_position = (self.episode_position + 1) % self.episode_capacity

    def seq_sample(self):
        return self.memory

    def last(self):
        return [self.memory[-1]]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_last(self, batch_size):
    	return self.memory[len(self.memory) - batch_size:]

    def sample_episodes(self, batch_size):
        sampled_episodes = self.sample(batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-self.trace_length)
            sampledTraces.append(episode[point:point+self.trace_length])
        return sampledTraces

    def flush(self):
        self.episode = []
        self.episode_position = 0

    def __len__(self):
        return len(self.memory)
