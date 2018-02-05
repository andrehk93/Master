import matplotlib.pyplot as plt
import numpy as np
import os


def plot(dictionary, folder, batch_size, title="Plot"):
	percentages = {1: [], 2: [], 5: [], 10: []}

	# Hyper parameter for plotting (How many episodes to average over per scatter):
	interval = 10

	for k in dictionary.keys():
		for episode_batch in range(0, len(dictionary[k]), interval):
			sum_list = 0.0
			len_list = 0.0
			for i in range(interval):
				if (episode_batch + i < len(dictionary[k])):
					sum_list += float(sum(dictionary[k][episode_batch + i]))
					len_list += float(len(dictionary[k][episode_batch + i]))
			percent = float(sum_list/len_list)
			percentages[k].append(percent)


	fig = plt.figure()
	plt.ylim((0.0, 1.0))
	x = []
	for i in range(len(percentages[1])):
		x.append(i*interval*batch_size)

	colors = ['indigo', 'aqua', 'red', 'lime']
	lables = ['1st Instance', '2nd Instance', '5th Instance', '10th Instance']
	plt.xlabel('Episode')
	if ('Prediction Accuracy' in title):
		plt.ylabel('Percent Correct')
	else:
		plt.ylabel('Percent Label Requests')
	i = 0
	for k in percentages.keys():
		plt.scatter(x, percentages[k], color=colors[i], alpha=0.6, label=lables[i])
		i += 1
	plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0.01)
	if (not os.path.exists("results/plots/" + folder)):
		os.makedirs("results/plots/" + folder)
	plt.savefig("results/plots/" + folder + title + ".png")
	plt.show()	
