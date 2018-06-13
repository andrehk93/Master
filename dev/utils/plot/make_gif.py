import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import operator

model_name = "TRAIN_lrua/"
weight_vector = "w_r/"
path = "../../results/memories/" + model_name + weight_vector
print(path)

filenames = []

for root, dirs, files in os.walk(path):
	for file in files:
		if (".DS_Store" not in file):
			filenames.append(root + file)

file_indexes = {}

for f in filenames:
	file_indexes[int(f[f.find("t_") + 2:f[10:].find(".") + 10])] = f

sorted_filenames = sorted(file_indexes.items(), key=operator.itemgetter(0))


images = []
print("Writing GIF in order:")
for (i, filename) in sorted_filenames:
    #images.append(imageio.imread(filename))
    print(filename)
    images.append(plt.imread(filename))

if (not os.path.exists('../../results/memories/gifs/' + model_name + weight_vector)):
	os.makedirs('../../results/memories/gifs/' + model_name + weight_vector)
imageio.mimsave('../../results/memories/gifs/' + model_name + weight_vector + 'timeseries.gif', images, duration=0.5)