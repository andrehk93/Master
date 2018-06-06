import random
import math
import torch
from torch.autograd import Variable
from torch import FloatTensor
import numpy as np


class ReinforcementLearning():

    EPS = 0.05


    def __init__(self, classes):
        self.classes = classes
        self.request_reward = -0.05
        self.prediction_reward = 1.0
        self.prediction_penalty = -1.0

    def select_actions(self, epoch, model_actions, batch_size, class_vector_size, episode_labels):

        # Discount factor for future rewards, and Greedy Epsilon Constant:
        START_EXP = 0.05
        END_EXP = 0.05
        STEP = 200

        # Performing Epsilon Greedy Exploration:
        agent_actions = []

        EPS = END_EXP + (START_EXP - END_EXP) * math.exp((-1.0 * epoch) / STEP)
        for i in range(batch_size):

            # Model choice:
            if (random.random() > EPS):
                agent_actions.append(model_actions[i])

            # Epsilong Greedy:
            else:
                epsilon_action = random.randint(0, 2)

                # Request:
                if (epsilon_action == 0):
                    agent_actions.append(class_vector_size)

                # Incorrect Prediction:
                elif (epsilon_action == 1):
                    
                    wrong_label = random.randint(0, class_vector_size - 1)
                    while (wrong_label == episode_labels[i]):
                        wrong_label = random.randint(0, class_vector_size - 1)
                    agent_actions.append(wrong_label)

                # Correct Prediction:
                else:
                    agent_actions.append(episode_labels[i])

        return agent_actions

    def select_multistate_actions(self, epoch, model_actions, batch_size, class_vector_size, episode_labels, one_hot_vectors):

        # Discount factor for future rewards, and Greedy Epsilon Constant:
        START_EXP = 0.05
        END_EXP = 0.05
        STEP = 200

        # Performing Epsilon Greedy Exploration:
        agent_actions = []
        label_list = ['a', 'b', 'c', 'd', 'e']

        EPS = END_EXP + (START_EXP - END_EXP) * math.exp((-1.0 * epoch) / STEP)
        for i in range(batch_size):

            # Model choice:
            if (random.random() > EPS):
                agent_actions.append(model_actions[i])

            # Epsilong Greedy:
            else:
                epsilon_action = random.randint(0, 2)

                # Request:
                if (epsilon_action == 0):
                    agent_actions.append(int(math.pow(len(one_hot_vectors[i][0]), len(one_hot_vectors[i][0]))))

                # Incorrect Prediction:
                elif (epsilon_action == 1):
                    bits = np.array(np.random.choice(len(label_list), len(label_list), replace=True))
                    wrong_label = np.zeros((len(label_list), len(label_list)))
                    wrong_label[np.arange(len(label_list)), bits] = 1

                    equal = np.array_equal(wrong_label, one_hot_vectors[i][episode_labels[i]])
                    while equal:
                        bits = np.array(np.random.choice(len(label_list), len(label_list), replace=True))
                        wrong_label = np.zeros((len(label_list), len(label_list)))
                        wrong_label[np.arange(len(label_list)), bits] = 1
                        equal = np.array_equal(wrong_label, one_hot_vectors[i][episode_labels[i]])
                    vec = wrong_label
                    index_values = [(len(vec[j]) - 1 - np.argmax(vec[j]))*(math.pow(len(vec[j]), len(vec[j]) - 1 - j)) for j in range(len(vec))]
                    value = sum(index_values)
                    agent_actions.append(int(value))

                # Correct Prediction:
                else:
                    vec = one_hot_vectors[i][episode_labels[i]]
                    index_values = [(len(vec[j]) - 1 - np.argmax(vec[j]))*(math.pow(len(vec[j]), len(vec[j]) - 1 - j)) for j in range(len(vec))]
                    value = sum(index_values)
                    agent_actions.append(int(value))

        return agent_actions








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

    """
    def collect_reward_multistate_batch(self, actions, one_hot_vectors, batch_size, episode_labels):
        # Last bit determines if the agent predicts or request the label:
        rewards = []
        for i in range(batch_size):
            if (int(actions[i]) == self.classes):
                rewards.append(self.request_reward)
            else:
                vec = one_hot_vectors[i][episode_labels[i]]
                value = sum([(len(vec[j]) - 1 - np.argmax(vec[j]))*(math.pow(len(vec[j]), len(vec[j]) - 1 - j)) for j in range(len(vec))])
                if (value == int(actions[i])):
                    rewards.append(self.prediction_reward)
                else:
                    rewards.append(self.prediction_penalty)
        return rewards
    """
    def collect_reward_multistate_batch(self, actions, one_hot_vectors, batch_size, episode_labels):
        # since its a concatenation of 
        partitioning = []
        # Last bit determines if the agent predicts or request the label:
        rewards = []
        for i in range(batch_size):
            if (int(actions[i]) == self.classes):
                rewards.append(self.request_reward)
            else:
                vec = one_hot_vectors[i][episode_labels[i]]
                value = sum([(len(vec[j]) - 1 - np.argmax(vec[j]))*(math.pow(len(vec[j]), len(vec[j]) - 1 - j)) for j in range(len(vec))])
                if (value == int(actions[i])):
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

    def next_multi_state(self, action, one_hot_vectors):
        action = action.squeeze()
        # Requesting the label:
        if (int(action[0]) == self.classes):
            return one_hot_label
        # Predicting (Thus returning a 0-vector to the input):
        else:
            zero_vector = []
            for i in range(len(one_hot_vectors)):
                zero_vector.append([0 for j in range(len(one_hot_vectors[i]))])
            return zero_vector


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

    def next_multistate_batch(self, actions, one_hot_vectors, batch_size, episode_labels):
        actions = actions
        states = []
        for i in range(batch_size):
            # Requesting the label:
            if (int(actions[i]) == self.classes):
                states.append(one_hot_vectors[i][episode_labels[i]])
            # Predicting (Thus returning a 0-vector to the input):
            else:
                zero_vector = np.zeros(((len(one_hot_vectors[i][0]), len(one_hot_vectors[i][0]))))
                states.append(zero_vector)
        return states