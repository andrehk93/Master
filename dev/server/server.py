from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import os
import numpy as np

static_folder = "../results/tables/"
static_name = "LSTM_from_pretrained"
n = 20

def write_to_memory(current_length):
    file = open("./store/memory.txt", "w")
    file.write(str(current_length))
    file.close()


def read_from_memory():
    file = open("./store/memory.txt", "r")
    lines = file.read()
    file.close()
    if (len(lines) > 0):
        return int(lines.strip())
    else:
        return 0

def process(arr):
    processed_array = np.asarray(arr, dtype=np.float32)

    # Averaging every Nth element:
    shortening_index = len(processed_array) % n

    shortened_array = processed_array[: len(processed_array) - shortening_index]
    averaged_array = np.mean(np.array(shortened_array).reshape(-1, n), axis=1)

    return averaged_array.tolist()


def get_best(arr):
    processed_array = np.asarray(arr, dtype=np.float32)
    print("processed: ", processed_array)
    print("BEST: ", np.argmax(processed_array))
    return int(np.argmax(processed_array))


def get_results_from_filename(filename):
    directory = os.path.join(static_folder, filename)

    result_array = []

    for dirName, subdir, fileList in os.walk(directory):
        for i, file in enumerate(fileList):
            with open(os.path.join(dirName, file)) as f:
                lines = f.readlines()
                if i == 0:
                    write_to_memory(len(lines))
                result_array.append(process(list(map(str.strip, lines))))

    return result_array


def get_last_results_from_filename(filename):
    directory = os.path.join(static_folder, filename)

    result_array = []
    current_line = 0

    for dirName, subdir, fileList in os.walk(directory):
        for i, file in enumerate(fileList):
            with open(os.path.join(dirName, file)) as f:
                lines = f.readlines()
                if i == 0:
                    current_line = read_from_memory()
                    if current_line + n < len(lines):
                        write_to_memory(len(lines))
                    else:
                        return
                result_array.append(process(list(map(str.strip, lines[current_line:]))))

    return result_array


def get_best_results_from_filename(filename):
    directory = os.path.join(static_folder, filename)

    result_array = []

    with open(os.path.join(directory, "reward.txt")) as f:
        best_result_index = get_best(f.readlines())

    for dirName, subdir, fileList in os.walk(directory):
        for i, file in enumerate(fileList):
            with open(os.path.join(dirName, file)) as f:
                result_array.append(f.readlines()[best_result_index].strip())
    return result_array


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def index():
    return "Index!"


@app.route("/hello")
@cross_origin()
def hello():
    return jsonify("Hello CORS World!")


@app.route("/result")
@cross_origin()
def get_result():
    return jsonify(get_results_from_filename(static_name))


@app.route("/result/update")
@cross_origin()
def update_result():
    return jsonify(get_last_results_from_filename(static_name))


@app.route("/result/best")
@cross_origin()
def get_best_result():
    return jsonify(get_best_results_from_filename(static_name))


@app.route("/members/<string:name>/")
@cross_origin()
def getMember(name):
    return name


if __name__ == "__main__":
    app.run()
