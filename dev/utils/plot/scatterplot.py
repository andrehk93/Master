import matplotlib.pyplot as plt
import numpy as np
import os


def plot(dictionary, folder, title="Plot"):
	percentages = {1: [], 2: [], 5: [], 10: []}

	# Hyper parameter for plotting (How many episodes to average over per scatter):
	interval = 100

	for k in dictionary.keys():
		for episode in range(0, len(dictionary[k]), interval):
			summation = 0
			length = 0
			for i in range(interval):
				if (i + episode < len(dictionary[k])):
					summation += sum(dictionary[k][episode + i])
					length += len(dictionary[k][episode + i])
			percentages[k].append(float(summation/length))


	fig = plt.figure()
	plt.ylim((0.0, 1.0))
	x = []
	for i in range(len(percentages[1])):
		x.append(i*interval)

	colors = ['indigo', 'aqua', 'red', 'lime']
	lables = ['1st Instance', '2nd Instance', '5th Instance', '10th Instance']
	plt.xlabel('Episode')
	if (title == 'Prediction Accuracy'):
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
