from utils.plot import loss_plot, percent_scatterplot as scatterplot
import numpy as np


class PlotHandler:

    def __init__(self, statistics, args):
        self.statistics = statistics
        self.args = args

    def plot(self):
        # Plot Accuracy, Prediction Accuracy & Requests simultaneously
        loss_plot.plot([self.statistics['total_accuracy'], self.statistics['total_prediction_accuracy'],
                        self.statistics['total_requests']],
                       ["Training Accuracy Percentage", "Training Prediction Accuracy", "Training Requests Percentage"],
                       "training_stats", self.args.name + "/", "Percentage")

        # Plot Test Accuracy, Test Prediction Accuracy & Test Requests simultaneously
        loss_plot.plot([self.statistics['total_test_accuracy'], self.statistics['total_test_prediction_accuracy'],
                        self.statistics['total_test_requests']],
                       ["Test Accuracy Percentage", "Test Prediction Accuracy",
                        "Test Requests Percentage"],
                       "total_testing_stats", self.args.name + "/", "Percentage")

        # Plot Loss
        loss_plot.plot([self.statistics['total_loss']], ["Training Loss"], "training_loss",
                       self.args.name + "/", "Average Loss", episode_size=self.args.episode_size)

        # Plot Reward
        loss_plot.plot([self.statistics['total_reward']], ["Training Average Reward"],
                       "training_reward", self.args.name + "/",
                       "Average Reward", episode_size=self.args.episode_size)

        # Plot Test Reward
        loss_plot.plot([self.statistics['total_test_reward']], ["Test Average Reward"],
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
        # Scatter without zoom
        scatterplot.plot(statistics['acc_dict'], self.args.name + "/", self.args.batch_size,
                         title="Prediction Accuracy")
        scatterplot.plot(statistics['req_dict'], self.args.name + "/", self.args.batch_size,
                         title="Total Requests")

        # Scatter with zoom
        scatterplot.plot(statistics['acc_dict'], self.args.name + "/", self.args.batch_size,
                         title="Prediction Accuracy", zoom=True)
        scatterplot.plot(statistics['req_dict'], self.args.name + "/", self.args.batch_size,
                         title="Total Requests", zoom=True)
