import time
import numpy as np
import os


# Prints time left
def print_time(avg_time, eta):
    print("\n --- TIME ---")
    print("\nT/epoch = " + str(avg_time)[0:4] + " s")

    hour = eta // 3600
    eta = eta - (3600 * hour)

    minute = eta // 60
    eta = eta - (60 * minute)

    seconds = eta

    # Stringify w/padding:
    if minute < 10:
        minute = "0" + str(minute)[0]
    else:
        minute = str(minute)[0:2]
    if hour < 10:
        hour = "0" + str(hour)[0]
    else:
        hour = str(hour)
    if seconds < 10:
        seconds = "0" + str(seconds)[0]
    else:
        seconds = str(seconds)[0:4]

    print("Estimated Time Left:\t" + hour + " h " + minute + " min " + seconds + " s")
    print("\n---------------------------------------")


# Prints the current best stats
def print_best_stats(stats):
    if len(stats['reward']) == 0:
        return
    
    best_index = np.argmax(stats['reward'])

    # Static strings:
    stat_string = "\n\t\tBest Training Stats"
    table_string = "|\tReward\t|\tPred. Acc.\t|\t[Acc / Req]\t\t|"
    str_length = 72

    # Printing:
    print(stat_string)
    print("-"*str_length)
    print(table_string)
    print("-"*str_length)
    print("|\t"
          + str(stats['reward'][best_index])[0:4] + "\t|\t"
          + str(stats['prediction_accuracy'][best_index])[0:4]+ " %\t\t|\t"
          + str(stats['accuracy'][best_index])[0:4] + " % / "
          + str(stats['requests'][best_index])[0:4] + " %\t\t|\t")
    print("-"*str_length + "\n\n")


def update_dicts(from_dict_1, from_dict_2, to_dict_1, to_dict_2):
    for key in to_dict_1.keys():
        to_dict_1[key].append(from_dict_1[key])
        to_dict_2[key].append(from_dict_2[key])


class StatusHandler:
    # Init train stuff:
    epoch = 1
    done = False
    start_time = time.time()

    avg_time = 0
    eta = 0
    hour = 0
    minute = 0
    seconds = 0

    time_interval = []
    interval = 50

    best = -1000000000000

    # Constants:
    SAVE = 10
    BACKUP = 50

    stats_directory = 'results/stats/'

    def __init__(self, args):
        self.episode = (args.start_epoch - 1) * args.batch_size
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.elapsed_time = 0
        self.name = args.name
        self.setup_folders()

    def setup_folders(self):
        # Set up result directory
        if not os.path.exists(self.stats_directory):
            os.makedirs(self.stats_directory)

    def update_status(self, epoch, statistics):
        self.episode += self.batch_size
        self.epoch += 1

        # Collect time estimates:
        if self.epoch % self.interval == 0:
            if len(self.time_interval) < 2:
                self.time_interval.append(time.time())
            else:
                self.time_interval[-2] = self.time_interval[-1]
                self.time_interval[-1] = time.time()

        # Print Estimated time left:
        if len(self.time_interval) > 1:
            avg_time = (self.time_interval[-1] - self.time_interval[-2]) / self.interval
            eta = (self.epochs + 1 - epoch) * avg_time
            print_time(avg_time, eta)

        statistics.set_variables({'epoch': epoch})

    def update_best(self, reward_array):
        best = np.argmax(reward_array)
        if reward_array[best] >= self.best:
            self.best = reward_array[best]
            return True
        return False

    def finish(self, epochs):
        self.elapsed_time = time.time() - self.start_time
        print("Elapsed time: " + str(self.elapsed_time) + " s")

        with open(os.path.join(self.stats_directory, self.name + ".txt"), 'w') as file:
            stats = ""
            stats += "Epochs:\t\t\t" + str(epochs) + "\n"
            stats += "Elapsed Time:\t\t\t" + str(self.elapsed_time) + " s\n"
            stats += "Avg Time:\t\t\t" + str(self.elapsed_time/epochs) + " s\n"
            stats += "Best Reward:\t\t\t" + str(self.best) + "\n"

            file.write(stats)
            file.close()

        print("Stats written successfully to file!")


def generate_name_from_args(args, text):
    model = "LRUA"
    if args.LSTM:
        model = "LSTM"
    elif args.NTM:
        model = "NTM"

    # Data set
    data_set = "_OMNIGLOT"
    if args.INH:
        data_set = "_INH"
    elif args.QA:
        data_set = "_QA"
    elif args.REUTERS:
        data_set = "_REUTERS"

    # CMS
    cms = ""
    if args.margin_sampling:
        cms = "_CMS_(size_" + str(args.margin_size) + "_time_" + str(args.margin_time) + ")"

    # Batch_size
    batch_size = "_bsize_" + str(args.batch_size)

    # class_vector_size
    c_size = "_c_" + str(args.class_vector_size)

    # Text
    text_string = ""
    if text:
        # Pre trained model
        vectors = "_GLOVE"
        if args.FAST:
            vectors = "_FAST"

        # Embedding Size
        embedding = "_" + str(args.embedding_size)

        # Sentence Length
        s_length = "_" + str(args.sentence_length)

        text_string = vectors + embedding + s_length

    postfix = args.name_postfix

    return model + data_set + cms + batch_size + c_size + text_string + postfix
