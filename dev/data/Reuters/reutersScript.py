from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import random
import os
import os.path
import errno
import numpy as np

save_dir = os.path.join("Reuters", "raw")
path_train = os.path.join("Retuers", "training")
path_test = os.path.join("Reuters", "test")
paths = [path_train, path_test]
for path in paths:
	for (root, dirs, files) in os.walk(path):
		for working_dir in dirs:
			if not os.path.exists(os.path.join(save_dir, working_dir)):
				os.makedirs(os.path.join(save_dir, working_dir))
			for (root2, dirs2, files2) in os.walk(os.path.join(path, working_dir)):
				for file in files2:
					if (file == ".DS_Store"):
						continue
					to_save = open(os.path.join(save_dir, working_dir, file), "wb")
					file_to_read = open(os.path.join(root2, file), "rb")
					print(os.path.join(root2, file))
					to_save.write(file_to_read.read())
					to_save.close()