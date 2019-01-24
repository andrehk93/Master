import os


def parse_folders_into_dataset():
    source = "source"
    dest = "raw"
    train_folder = "train.txt"
    test_folder = "test.txt"
    dictionary = {}

    dictionary = parse_dataset(os.path.join(source, train_folder), dictionary)
    _ = parse_dataset(os.path.join(source, test_folder), dictionary)
    prune_dataset(dest)


def parse_dataset(directory, dictionary):
    print("Parsing texts from ", directory)
    dest = "raw"
    with open(directory, "r") as file:
        for line in file.readlines():
            separated = line.strip().split(" ")
            label = separated[0]
            new_filename = os.path.join(dest, label.replace(":", "_"))
            if label not in dictionary:
                dictionary[label] = 1
                if not os.path.exists(new_filename):
                    os.mkdir(new_filename)
            else:
                dictionary[label] += 1
            saved_text = open(os.path.join(new_filename, str(dictionary[label])), "w")
            saved_text.write(" ".join(separated[1:]))
            saved_text.close()

    return dictionary


def prune_dataset(directory):
    print("Remving folders with to few samples...")
    folders_to_delete = []
    delete = 0
    keep = 0
    for (root, dirs, files) in os.walk(directory):
        if len(files) < 15 and root != "raw":
            folders_to_delete.append(root)
            delete += 1
        else:
            keep += 1

    print(folders_to_delete)
    print("Keep-percentage: ", (100.0*keep)/(keep + delete))

    for folder in folders_to_delete:
        try:
            for (root, dirs, files) in os.walk(folder):
                for file in files:
                    print("Removing: ", os.path.join(root, file))
                    os.unlink(os.path.join(root, file))
            os.rmdir(folder)
        except FileNotFoundError:
            continue

    print("Successfully pruned dataset!")


parse_folders_into_dataset()
