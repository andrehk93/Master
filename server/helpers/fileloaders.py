import numpy as np
import os

static_folder = "results/plots/"


def get_result_from_filename(filename):
    directory = os.path.join(static_folder, filename + "/k_shot_table_test.txt")
    print("Directory: ", directory)

    with open(directory) as file:
        content = file.readlines()

    return content
