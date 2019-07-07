import os


# Writes a table to file:
def write_stats(requests, accuracy, penalty, folder, test=False):
    if test:
        filename = "./results/plots/" + str(folder) + "test_table_file.txt"
    else:
        filename = "./results/plots/" + str(folder) + "table_file.txt"
    dimensions = [55, 31, 31]
    headers = ["Method", "Accuracy (%)", "Requests (%)"]
    method = "RL Prediction"
    if test:
        method += "(Rinc = " + str(penalty) + ") - Test"
    else:
        method += "(Rinc = " + str(penalty) + ") - Training"
    specs = [accuracy, requests]

    stat_list = []
    j = 0
    for s in specs:
        if j == 0:
            stat_list.append(method)
        average = float(sum(s)/len(s))
        stat_list.append(average)
        j += 1

    # Creating Line:
    table = ""
    line = "\n+"
    for d in dimensions:
        line += "-"*d + "+"
    line += "\n"

    file = open(filename, "w")

    # HEADER
    table += line
    header = "|"
    for i, d in enumerate(dimensions):
        if i == 0:
            header += "\t\t\t" + headers[i] + "\t\t\t\t" + "|"
        else:
            header += "\t" + headers[i] + "\t\t" + "|"
    table += header
    table += line

    # BODY
    for stat in range(0, len(stat_list), 3):
        table += "|"
        for i in range(3):
            if i == 0:
                table +=  "\t" + str(stat_list[stat + i]) + "\t\t" + "|"
            else:
                table +=  "\t\t" + str(stat_list[stat + i])[0:5] + "\t\t" + "|"
        # END
        table += line

    
    print(table)
    file.write(table)
    print("Table successfully written!")
    file.close()


# Writes a table to file:
def write_baseline_stats(accuracy, folder):
    filename = "./results/plots/" + str(folder) + "table_file.txt"
    stat_filename = "./results/plots/" + str(folder) + "stat_file.txt"
    dimensions = [50, 20, 20]
    headers = ["Method", "Accuracy (%)"]
    method = "Supervised"
    specs = [accuracy]
    if os.path.isfile(stat_filename):
        stat_file = open(stat_filename, "a")
    else:
        stat_file = open(stat_filename, "w")

    stat_file.write(method + "\n")

    for s in specs:
        average = float(sum(s)/len(s))
        stat_file.write(str(average)[0:5] + "\n")
    stat_file.close()

    # Reading from stat_file:
    stats = {}
    length = 3
    with open(stat_filename, "r") as statistics:
        i = 0
        current_key = ""
        for line in statistics:
            if i == 0:
                if line.rstrip() not in stats:
                    stats[line.rstrip()] = [[], []]
                current_key = line.rstrip()
            elif i < length:
                stats[current_key][i-1].append(float(line.rstrip()))
            i += 1
            if i == length:
                i = 0

    stat_list = []
    for k in stats.keys():
        stat_list.append([])
        stat_list[-1].append(k)
        for v in stats[k]:
            stat_list[-1].append(sum(v)/len(v))

    print(stats)
    print(stat_list)

    # Creating Line:
    table = ""
    line = "\n+"
    for d in dimensions:
        line += "-"*d + "+"
    line += "\n"

    if os.path.isfile(filename):
        file = open(filename, "a")

    else:
        file = open(filename, "w")

        # HEADER
        table += line
        header = "|"
        for i, d in enumerate(dimensions):
            header += int((d/2) - int(len(headers[i])/2))*" " + headers[i] + int((d/2) - int(len(headers[i])/2))*" " + "|"
        table += header
        table += line

    # BODY
    for stat in stat_list:
        table += "|"
        for i, d in enumerate(dimensions):
            table +=  int((d/2) - int(len(str(stat[i]))/2))*" " + str(stat[i]) + int((d/2) - int(len(str(stat[i]))/2))*" " + "|"
        # END
        table += line

    print(table)
    file.write(table)
    print("Table successfully written!")
    file.close()


def print_k_shot_tables(prediction_accuracies, accuracies, requests, data_set, folder):
    filename = "results/plots/" + str(folder) + "k_shot_table_" + data_set + ".txt"
    for key in accuracies.keys():
        prediction_accuracies[key] = 100.0 * float(sum(prediction_accuracies[key])/max(len(prediction_accuracies[key]), 1))
        accuracies[key] = 100.0 * float(sum(accuracies[key])/max(len(accuracies[key]), 1))
        requests[key] = 100.0 * float(sum(requests[key])/max(len(requests[key]), 1))
    with open(filename, "w") as table:
        table.write("\n\n--- K-shot predictions for " + data_set + "-set ---\n")
        table.write("Instance:\tPred. Acc:\tAccuracy:\tRequests:\n")
        for key in accuracies.keys():
            table.write(str(key) + ":\t\t" + str(prediction_accuracies[key])[0:4] + " %\t\t" + str(accuracies[key])[0:4] + " %\t\t" + str(requests[key])[0:4] + " %\n")


def print_k_shot_baseline_tables(accuracies, data_set, folder):
    filename = "results/plots/" + str(folder) + "k_shot_table_" + data_set + ".txt"
    for key in accuracies.keys():
        accuracies[key] = 100.0 * float(sum(accuracies[key])/max(len(accuracies[key]), 1))
    with open(filename, "w") as table:
        table.write("\n\n--- K-shot predictions for " + data_set + "-set ---\n")
        table.write("Instance:\tAccuracy:\t")
        for key in accuracies.keys():
            table.write(str(key) + ":\t\t" + str(accuracies[key])[0:4] + " %\n")
