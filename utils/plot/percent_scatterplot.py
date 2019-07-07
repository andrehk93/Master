import matplotlib.pyplot as plt
import numpy as np
import os


def plot(dictionary, folder, epochs, title="Plot", zoom=False):
    percentages = {1: [], 2: [], 5: [], 10: []}

    # Hyper parameter for plotting (How many episodes to average over per scatter):
    interval = int(epochs / 100)
    # interval = 32
    start = 0

    if zoom:
        start = int(epochs * 0.8)
        interval = 200

    for k in dictionary.keys():
        for episode_batch in range(start, len(dictionary[k]), interval):
            sub_list = dictionary[k][episode_batch: min(episode_batch + interval, len(dictionary[k]))]
            length = len(sub_list)
            percentages[k].append(float(sum(sub_list) / length))

    fig = plt.figure()
    plt.ylim((0.0, 1.0))
    plt.yticks(np.arange(0, 1, step=0.1))
    x = []
    for i in range(len(percentages[1])):
        x.append(start + (i * interval))

    # Plot line dividing training and test data-points
    plt.axvline(x=epochs, color="red")

    colors = ['indigo', 'aqua', 'red', 'lime']
    lables = ['1st Instance', '2nd Instance', '5th Instance', '10th Instance']
    plt.xlabel('Episode Batches')
    plt.grid(True)

    # Collect Labels
    if 'Prediction Accuracy' in title:
        plt.ylabel('Percent Correct')
    else:
        plt.ylabel('Percent Label Requests')

    # Plot percentages
    i = 0
    for k in percentages.keys():
        plt.plot(x, percentages[k], '-o', color=colors[i], alpha=0.6, label=lables[i], markeredgecolor="black")
        i += 1

    # Create Legend
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.01)

    # Save to folder
    if not os.path.exists("results/plots/" + folder):
        os.makedirs("results/plots/" + folder)
    if zoom:
        plt.savefig("results/plots/" + folder + title + "_zoom.png")
    else:
        plt.savefig("results/plots/" + folder + title + ".png")
    plt.show()
