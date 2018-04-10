# Development for Masters Project 2018

## Important
Go to https://github.com/brendenlake/omniglot/tree/master/python and download the OMNIGLOT dataset (images_background.zip & images_evaluation.zip). Unzip both files and put ALL folders (50 folders total) from the unzipped folder (these folders have names of different alphabets i.e "Angelic", "Atlantean" etc.) in a folder you call "raw" which you put in data/omniglot/, and the scripts will do the rest.

## Models
Here we have three different models:

1. LSTM Baseline model
Implemented from "Active One-Shot Learning" (https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf). This is used as a baseline for my experiments with different memory structures.

2. NTM Model
Implemented partially from https://github.com/loudinthecloud/pytorch-ntm, with added functionality similar to "Active One-Shot Learning" and "Meta-Learning with Memory-Augmented Neural Networks" (http://proceedings.mlr.press/v48/santoro16.pdf). 

3. LRUA Model
Simply an augmented version of the NTM model, similar to the LRUA in http://proceedings.mlr.press/v48/santoro16.pdf. The only difference is that the number of read heads is identical to the number of write heads, and that every memory location is either written to the least used location, or simply the first location, in memory.
