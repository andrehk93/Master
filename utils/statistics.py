from utils.dictionary_utils import merge_dicts
import numpy as np


class Statistics:
    statistics = {}

    def __init__(self):
        self.statistics['best_accuracy'] = 0.0

        # Dictionaries
        self.statistics['test_pred_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_train_pred_dict'] = {1: [], 2: [], 5: [], 10: []}

        self.statistics['acc_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_acc_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_train_acc_dict'] = {1: [], 2: [], 5: [], 10: []}

        self.statistics['req_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_req_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_train_req_dict'] = {1: [], 2: [], 5: [], 10: []}

        # Requests
        self.statistics['requests'] = []
        self.statistics['test_requests'] = []
        self.statistics['test_train_requests'] = []
        self.statistics['training_test_requests'] = []

        # Accuracy
        self.statistics['accuracy'] = []
        self.statistics['test_accuracy'] = []
        self.statistics['test_train_accuracy'] = []
        self.statistics['training_test_accuracy'] = []

        # Prediction Accuracy
        self.statistics['prediction_accuracy'] = []
        self.statistics['test_prediction_accuracy'] = []
        self.statistics['test_train_prediction_accuracy'] = []
        self.statistics['training_test_prediction_accuracy'] = []

        # Reward
        self.statistics['reward'] = []
        self.statistics['test_reward'] = []
        self.statistics['test_train_reward'] = []
        self.statistics['training_test_reward'] = []

        self.statistics['loss'] = []

        self.statistics['all_margins'] = []
        self.statistics['low_margins'] = []
        self.statistics['all_choices'] = []

        self.statistics['best'] = -30
        self.statistics['epoch'] = 1

        self.statistics['state_dict'] = {}

    def update(self, stats_dict, dictionaries):
        # Update statistics dictionary
        for key in stats_dict.keys():
            self.statistics[key].append(stats_dict[key])
        for key in dictionaries.keys():
            merge_dicts(dictionaries[key], self.statistics[key])

    def set_variables(self, variables):
        for key in variables.keys():
            self.statistics[key] = variables[key]

    def push_variables(self, variables):
        for key in variables.keys():
            self.statistics[key].append(variables[key])

    def update_state(self, state):
        self.statistics['state_dict'] = state
