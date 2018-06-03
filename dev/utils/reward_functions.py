import matplotlib.pyplot as plt
import numpy as np
import math


def linear_step(epochs, start, end):
	step = float((end - start) / epochs)

	return step


def exp_reward(epochs, start, end, y, start_epoch, linear_step, y_t, step):
	if (y < start_epoch):
		y_e = y_t + linear_step
	else:
		y_e = start + (end - start) * (math.exp(-1.0*(epochs - y)/(epochs/step)))

	return y_e





def plot(epochs, start, end, step, start_epoch):
	# Linear:
	x = np.arange(epochs)
	linear_step_size = linear_step(epochs, start, end)
	y_0 = start
	f = []
	for y in range(epochs):
		y_0 += linear_step_size
		f.append(y_0)
	plt.plot(x, f)
	plt.show()

	# Exp:
	x = np.arange(epochs)
	y_t = start
	f = []
	linear_step_size = linear_step(start_epoch, start, start + (end - start)*math.exp(-1.0*(epochs - start_epoch)/(epochs/step)))
	for y in range(0, epochs):
		y_t = exp_reward(epochs, start, end, y, start_epoch, linear_step_size, y_t, step)
		f.append(y_t)
	print(f[0:100])
	plt.plot(x, f)
	plt.show()

	f = []
	f2 = []
	for y in range(epochs):
		f_x = 1 - math.exp(-1.0*y/10000)
		f.append(f_x)
		f2.append(math.exp(-1.0*y/10000))

	plt.plot(x, f)
	plt.plot(x, f2)
	plt.show()



# PARAMS:
epochs = 100000
step = 5
start_epoch = 10000
start = -0.05
end = -1.0

plot(epochs, start, end, step, start_epoch)
