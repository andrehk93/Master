from utils.dictionary_utils import merge_dicts


class Statistics:
    statistics = {}

    def __init__(self):
        self.statistics['best_accuracy'] = 0.0
        self.statistics['req_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['acc_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_pred_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_acc_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['test_req_dict'] = {1: [], 2: [], 5: [], 10: []}
        self.statistics['total_requests'] = []
        self.statistics['total_test_requests'] = []
        self.statistics['total_accuracy'] = []
        self.statistics['total_test_accuracy'] = []
        self.statistics['total_prediction_accuracy'] = []
        self.statistics['total_test_prediction_accuracy'] = []
        self.statistics['total_loss'] = []
        self.statistics['total_reward'] = []
        self.statistics['total_test_reward'] = []
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

    def update_variables(self, variables):
        for key in variables.keys():
            self.statistics[key] = variables[key]

    def update_state(self, state):
        self.statistics['state_dict'] = state
