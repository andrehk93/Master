def merge_dicts(from_dict, to_dict):
    for key in to_dict.keys():
        to_dict[key].append(from_dict[key])
