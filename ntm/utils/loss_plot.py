import matplotlib.pyplot as plt
import numpy as np
import os

def plot(lists, labels, filename, folder, ylabel):
	x = np.arange(1, 50*len(lists[0]) + 1, 50)
	for l in range(len(lists)):
		plt.plot(x, lists[l], label=labels[l])
	plt.legend(loc=0)
	plt.title("RL Training Statistics")
	plt.xlabel("Episodes")
	plt.ylabel(ylabel)
	plt.grid(True)
	if ("stats" in filename):
		plt.ylim((0, 100))
	directory = "results/plots/"
	if not os.path.exists(directory + folder):
		os.makedirs(directory + folder)
	plt.savefig(directory + folder + filename + ".png")
	plt.show()
