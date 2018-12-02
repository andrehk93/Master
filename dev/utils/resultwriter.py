import os
static_folder = "./results/tables/"

def write_to_result_file(name, result_array):
    if (not os.path.exists(static_folder)):
        os.mkdir(static_folder)
    directory = os.path.join(static_folder, name)
    if (not os.path.exists(directory)):
        os.mkdir(directory)

    pred_acc = "prediction_accuracy.txt"
    req = "requests.txt"
    acc = "accuracy.txt"
    loss = "loss.txt"
    rew = "reward.txt"

    files = [pred_acc, acc, req, loss, rew]


    for (res, file) in zip(result_array, files):
        result_file = open(os.path.join(directory, file), "a")
        result_file.write(str(res) + "\n")
        result_file.close()
