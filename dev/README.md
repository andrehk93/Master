# Development for Masters Project 2018

## Important
# For the OMNIGLOT dataset:
Go to https://github.com/brendenlake/omniglot/tree/master/python and download the OMNIGLOT dataset (images_background.zip & images_evaluation.zip). Unzip both files and put BOTH folders in a folder you call "raw", which you put in data/omniglot/, and the scripts will do the rest.

To use the model/methods that utilize the OMNIGLOT dataset, please use "main_no_target.py" to train your model.

# For the Reuters dataset:
Go to http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-115Cat.tar.gz, and it will download the dataset immediatly. Unzip the file, and from the unzipped folder, put bot the "test" and "training"-folders inside data/reuters/. Run the script inside said folder, and then you're good to go.

To use the model/methods that utilize the REUTERS dataset, please use "main_text.py" to train your model. (NOTE: Currently only tested on Windows 10)

## Models
Both datasets can be trained on three different models:

1. LSTM Baseline model
Implemented from "Active One-Shot Learning" (https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf). This is used as a baseline for my experiments with different memory structures.

2. NTM Model
Implemented partially from https://github.com/loudinthecloud/pytorch-ntm, with added functionality similar to "Active One-Shot Learning" and "Meta-Learning with Memory-Augmented Neural Networks" (http://proceedings.mlr.press/v48/santoro16.pdf). 

3. LRUA Model
Simply an augmented version of the NTM model, similar to the LRUA in http://proceedings.mlr.press/v48/santoro16.pdf. The only difference is that the number of read heads is identical to the number of write heads, and that every memory location is either written to the least used location, or simply the first location, in memory.
