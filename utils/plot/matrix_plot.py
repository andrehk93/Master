import matplotlib.pylab as plt
import numpy as np
import os



def plot_matrix(predictions, labels, folder, title="matrix_plot"):
	episode_length = len(predictions)

	# Creating a numpy matrix of the predictions:
	matrix = np.matrix(predictions).transpose()

	# max_label:
	max_label = int(max(labels)) + 1

	one_hot_labels = []
	for label in labels:
		one_hot_labels.append([1 if i == label else 0 for i in range(max_label)])

	label_matrix = np.matrix(one_hot_labels).transpose()


	fig = plt.figure()
	fig.suptitle(title, fontsize=12)

	# Prediction Matrix:
	ax = fig.add_subplot(2,1,1)
	ax.set_aspect('equal')
	plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap("Oranges"))

	# Label Matrix:
	ax = fig.add_subplot(2,1,2)
	ax.set_aspect('equal')
	plt.imshow(label_matrix, interpolation='nearest', cmap=plt.get_cmap("Oranges"))

	plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
	cax = plt.axes([0.85, 0.1, 0.075, 0.8])
	plt.colorbar(cax=cax)

	if (not os.path.exists("results/plots/" + folder)):
		os.makedirs("results/plots/" + folder)
	plt.savefig("results/plots/" + folder + title + ".png")
	plt.show()





	