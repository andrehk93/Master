import random
import numpy as np


class ReinforcementLearning:

    # For Epsilon Greedy Exploration:
    EPS = 0.05

    def __init__(self, classes):
        self.classes = classes
        self.request_reward = -0.05
        self.prediction_reward = 1.0
        self.prediction_penalty = -1.0

    def select_actions(self, epoch, model_actions, batch_size, class_vector_size, episode_labels):

        # Performing Epsilon Greedy Exploration:
        agent_actions = []

        EPS = self.EPS
        for i in range(batch_size):

            # Model choice:
            if random.random() > EPS:
                agent_actions.append(model_actions[i])

            # Epsilon Greedy:
            else:
                epsilon_action = random.randint(0, 2)

                # Request:
                if epsilon_action == 0:
                    agent_actions.append(class_vector_size)

                # Incorrect Prediction:
                elif epsilon_action == 1:
                    
                    wrong_label = random.randint(0, class_vector_size - 1)
                    while wrong_label == episode_labels[i]:
                        wrong_label = random.randint(0, class_vector_size - 1)
                    agent_actions.append(wrong_label)

                # Correct Prediction:
                else:
                    agent_actions.append(episode_labels[i])

        return agent_actions

    # Collecting the rewards received, given the action chosen:
    def collect_reward_batch(self, actions, labels, batch_size):
        # Last bit determines if the agent predicts or request the label:
        rewards = []
        for i in range(batch_size):
            if int(actions[i]) == self.classes:
                rewards.append(self.request_reward)
            else:
                if np.argmax(labels[i]) == int(actions[i]):
                    rewards.append(self.prediction_reward)
                else:
                    rewards.append(self.prediction_penalty)
        return rewards

    # Collects a batch of next states, corresponding to the appropriate actions:
    def next_state_batch(self, actions, one_hot_labels, batch_size):
        actions = actions
        states = []
        for i in range(batch_size):
            # Requesting the label:
            if int(actions[i]) == self.classes:
                states.append(one_hot_labels[i])
            # Predicting (Thus returning a 0-vector to the input):
            else:
                states.append([0 for i in range(self.classes)])
        return states
