import matplotlib.pyplot as plt
import numpy as np
import os


def plot(dictionary, folder, batch_size, title="Plot"):
	percentages = {1: [], 2: [], 5: [], 10: []}

	# Hyper parameter for plotting (How many episodes to average over per scatter):
	interval = 1000

	for k in dictionary.keys():
		for episode_batch in range(0, len(dictionary[k]), interval):
			sub_list = dictionary[k][episode_batch : min(episode_batch + interval, len(dictionary[k]))]
			length = len(sub_list)
			percentages[k].append(float(sum(sub_list)/length))


	fig = plt.figure()
	plt.ylim((0.0, 1.0))
	plt.yticks(np.arange(0, 1, step=0.1))
	x = []
	for i in range(len(percentages[1])):
		x.append(i*interval)

	colors = ['indigo', 'aqua', 'red', 'lime']
	lables = ['1st Instance', '2nd Instance', '5th Instance', '10th Instance']
	plt.xlabel('Episode Batches')
	plt.grid(True)
	if ('Prediction Accuracy' in title):
		plt.ylabel('Percent Correct')
	else:
		plt.ylabel('Percent Label Requests')
	i = 0
	for k in percentages.keys():
		plt.plot(x, percentages[k], '-o', color=colors[i], alpha=0.6, label=lables[i], markeredgecolor="black")
		i += 1
	plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.01)
	if (not os.path.exists("results/plots/" + folder)):
		os.makedirs("results/plots/" + folder)
	plt.savefig("results/plots/" + folder + title + ".png")
	plt.show()	
