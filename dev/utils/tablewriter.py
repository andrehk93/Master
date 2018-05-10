import os
import shutil


# Writes a table to file:
def write_stats(requests, accuracy, penalty, folder, test=False):
    filename = "results/plots/" + str(folder) + "table_file.txt"
    stat_filename = "results/plots/" + str(folder) + "stat_file.txt"
    dimensions = [50, 20, 20]
    headers = ["Method", "Accuracy (%)", "Requests (%)"]
    method = "RL Prediction"
    if penalty == 0:
        method = "Supervised"
    else:
        if test:
            method += "(Rinc = " + str(penalty) + ") - Test"
        else:
            method += "(Rinc = " + str(penalty) + ") - Training"
    specs = [accuracy, requests]
    if (os.path.isfile(stat_filename)):
        stat_file = open(stat_filename, "a")
    else:
        stat_file = open(stat_filename, "w")
    stat_file.write(method + "\n")

    # Averaging over 20 episodes:
    for s in specs:
        length = min(20, len(s))
        average = float(sum(s[len(s) - length:])/length)
        stat_file.write(str(average)[0:4] + "\n")
    stat_file.close()

    # Reading from stat_file:
    stats = {}
    length = 3
    with open(stat_filename, "r") as statistics:
        i = 0
        current_key = ""
        for line in statistics:
            if (i == 0):
                if (line.rstrip() not in stats):
                    stats[line.rstrip()] = [[], []]
                current_key = line.rstrip()
            elif (i < length):
                stats[current_key][i-1].append(float(line.rstrip()))
            i += 1
            if (i == length):
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

    if (os.path.isfile(filename)):
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

    # Writes a table to file:
def write_baseline_stats(accuracy, folder, test=False):
    filename = "results/plots/" + str(folder) + "table_file.txt"
    stat_filename = "results/plots/" + str(folder) + "stat_file.txt"
    dimensions = [50, 20, 20]
    headers = ["Method", "Accuracy (%)"]
    method = "Supervised"
    specs = [accuracy]
    if (os.path.isfile(stat_filename)):
        stat_file = open(stat_filename, "a")
    else:
        stat_file = open(stat_filename, "w")

    stat_file.write(method + "\n")

    # Averaging over 20 episodes:
    for s in specs:
        length = min(20, len(s))
        average = float(sum(s[len(s) - length:])/length)
        stat_file.write(str(average)[0:4] + "\n")
    stat_file.close()

    # Reading from stat_file:
    stats = {}
    length = 3
    with open(stat_filename, "r") as statistics:
        i = 0
        current_key = ""
        for line in statistics:
            if (i == 0):
                if (line.rstrip() not in stats):
                    stats[line.rstrip()] = [[], []]
                current_key = line.rstrip()
            elif (i < length):
                stats[current_key][i-1].append(float(line.rstrip()))
            i += 1
            if (i == length):
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

    if (os.path.isfile(filename)):
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



def print_k_shot_tables(accuracies, requests, dataset, folder):
    filename = "results/plots/" + str(folder) + "k_shot_table_" + dataset + ".txt"
    for key in accuracies.keys():
        accuracies[key] = 100.0 * float(sum(accuracies[key])/max(len(accuracies[key]), 1))
        requests[key] = 100.0 * float(sum(requests[key])/max(len(requests[key]), 1))
    with open(filename, "w") as table:
        table.write("\n\n--- K-shot predictions for " + dataset + "-set ---\n")
        table.write("Instance:\tAccuracy:\tRequests:\n")
        for key in accuracies.keys():
            table.write(str(key) + ":\t\t" + str(accuracies[key])[0:4] + " %\t\t" + str(requests[key])[0:4] + " %\n")

def print_k_shot_baseline_tables(accuracies, dataset, folder):
    filename = "results/plots/" + str(folder) + "k_shot_table_" + dataset + ".txt"
    for key in accuracies.keys():
        accuracies[key] = 100.0 * float(sum(accuracies[key])/max(len(accuracies[key]), 1))
    with open(filename, "w") as table:
        table.write("\n\n--- K-shot predictions for " + dataset + "-set ---\n")
        table.write("Instance:\tAccuracy:\t")
        for key in accuracies.keys():
            table.write(str(key) + ":\t\t" + str(accuracies[key])[0:4] + " %\n")
