import matplotlib.pyplot as plt
import numpy as np
import os

def plot(lists, labels, filename, folder, ylabel):
	avg_lists = []
	average = 5
	for l in range(len(lists)):
		list_l = lists[l]
		avg_lists.append([])
		for i in range(0, len(list_l), average):
			sub_list = list_l[i : min(i + average, len(list_l))]
			length = len(sub_list)
			avg_lists[l].append(float(sum(sub_list)/length))


	x = np.arange(1, len(avg_lists[0]) + 1)
	for l in range(len(avg_lists)):
		avg_list = avg_lists[l]
		plt.plot(x, avg_list, label=labels[l])

	plt.legend(loc=0)
	plt.title("RL Training Statistics")
	plt.xlabel("Episode Batches")
	plt.ylabel(ylabel)
	plt.grid(True)
	if ("stats" in filename):
		plt.ylim((0, 100))
	directory = "results/plots/"
	if not os.path.exists(directory + folder):
		os.makedirs(directory + folder)
	plt.savefig(directory + folder + filename + ".png")
	plt.show()
