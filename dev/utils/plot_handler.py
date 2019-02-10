from utils.plot import loss_plot, percent_scatterplot as scatterplot
from utils.dictionary_utils import concatenate_dictionaries
import numpy as np


class PlotHandler:

    def __init__(self, statistics, args):
        self.statistics = statistics
        self.args = args

    def plot(self):
        # Plot Accuracy, Prediction Accuracy & Requests simultaneously
        loss_plot.plot([self.statistics['accuracy'], self.statistics['prediction_accuracy'],
                        self.statistics['requests']],
                       ["Training Accuracy Percentage", "Training Prediction Accuracy", "Training Requests Percentage"],
                       "training_stats", self.args.name + "/", "Percentage")

        # Plot Test Accuracy, Test Prediction Accuracy & Test Requests simultaneously
        loss_plot.plot([self.statistics['training_test_accuracy'], self.statistics['training_test_prediction_accuracy'],
                        self.statistics['training_test_requests']],
                       ["Test Accuracy Percentage", "Test Prediction Accuracy",
                        "Test Requests Percentage"],
                       "training_test_stats", self.args.name + "/", "Percentage")

        # Plot Loss
        loss_plot.plot([self.statistics['loss']], ["Training Loss"], "training_loss",
                       self.args.name + "/", "Average Loss", episode_size=self.args.episode_size)

        # Plot Reward
        loss_plot.plot([self.statistics['reward']], ["Training Average Reward"],
                       "training_reward", self.args.name + "/",
                       "Average Reward", episode_size=self.args.episode_size)

        # Plot Test Reward
        loss_plot.plot([self.statistics['training_test_reward']], ["Test Average Reward"],
                       "total_test_reward", self.args.name + "/",
                       "Average Reward", episode_size=self.args.episode_size)

    def margin_plot(self):
        loss_plot.plot([self.statistics['all_margins']], ["Avg. Highest Sample Margin"],
                       "highest_sample_margin", self.args.name + "/",
                       "Avg. Highest Sample Margin", avg=5)
        loss_plot.plot([self.statistics['low_margins']], ["Avg. Lowest Sample Margin"],
                       "lowest_sample_margin", self.args.name + "/",
                       "Avg. Lowest Sample Margin", avg=5)
        all_choices = np.array(self.statistics['all_choices'])
        loss_plot.plot([all_choices[:, c] for c in range(self.args.class_vector_size + 1)],
                       ["Class " + str(c)
                        if c < self.args.class_vector_size
                        else "Request"
                        for c in range(self.args.class_vector_size + 1)],
                       "sample_q", self.args.name + "/", "Avg. Highest Q Value", avg=20)

    def scatter_plot(self, statistics):
        # Need to concatenate training + test results:
        accuracies = concatenate_dictionaries(statistics['acc_dict'], statistics['test_acc_dict'])
        requests = concatenate_dictionaries(statistics['req_dict'], statistics['test_req_dict'])
        # Scatter without zoom
        scatterplot.plot(accuracies, self.args.name + "/", self.args.epochs,
                         title="Prediction Accuracy")
        scatterplot.plot(requests, self.args.name + "/", self.args.epochs,
                         title="Total Requests")

        # Scatter with zoom
        scatterplot.plot(accuracies, self.args.name + "/", self.args.epochs,
                         title="Prediction Accuracy", zoom=True)
        scatterplot.plot(requests, self.args.name + "/", self.args.epochs,
                         title="Total Requests", zoom=True)
