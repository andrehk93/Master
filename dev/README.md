# Development for Masters Project 2018

## Important
### For the OMNIGLOT dataset:
Go to https://github.com/brendenlake/omniglot/tree/master/python and download the OMNIGLOT dataset (images_background.zip & images_evaluation.zip). Unzip both files and put BOTH folders in a folder you call "raw", which you put in data/omniglot/, and the scripts will do the rest.

To use the model/methods that utilize the OMNIGLOT dataset, please use "main_image.py" to train your model. The models can also be trained as in "http://proceedings.mlr.press/v48/santoro16.pdf", using "non_rl_main_image.py" and the OMNIGLOT dataset.

### For the INH dataset:
Go to https://www.kaggle.com/therohk/india-headlines-news-dataset, and download the dataset (You will need to sign in/sign up to Kaggle.com in order to do this). Unzip the file, and place the resulting .csv file called "india-news-headlines.csv" into "data/text/headlines/". Then run the script in the same folder called "csv_to_folders.py". This script will create folders for each headline category, and put each headline into it's corresponding category. This will take some time, as the dataset is over 2.7 million rows long. NOTE: The script may fail at the end, but this is due to some compromises made, and is not of concern, all files and folders have been made at this point!

To use the model/methods that utilize the INH dataset, please use "main_text.py" to train your model. 

NOTE: The first time you run main_text.py, the scripts will create word index vectors and store these for training. Thus, to not having to do this multiple times, be sure that the wanted sentence lentgh, number of sentences and dictionary size are as you want them to be.

### For the Reuters dataset:
Go to http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-115Cat.tar.gz, and it will download the dataset immediatly. Unzip the file, and from the unzipped folder, put bot the "test" and "training"-folders inside data/reuters/. Run the script inside said folder, and then you're good to go.

To use the model/methods that utilize the REUTERS dataset, please use "main_text.py" to train your model, and switch the "dataset" variable to the location of the REUTERS dataset. (NOTE: Currently only tested on Windows 10)

## Models
Both datasets can be trained on three different models:

1. LSTM Baseline model
Implemented from "Active One-Shot Learning" (https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf). This is used as a baseline for my experiments with different memory structures.

2. NTM Model
Implemented partially from https://github.com/loudinthecloud/pytorch-ntm, with added functionality similar to "Active One-Shot Learning" and "Meta-Learning with Memory-Augmented Neural Networks" (http://proceedings.mlr.press/v48/santoro16.pdf). 

3. LRUA Model
Simply an augmented version of the NTM model, similar to the LRUA in http://proceedings.mlr.press/v48/santoro16.pdf. The only difference is that the number of read heads is identical to the number of write heads, and that every memory location is either written to the least used location, or simply the first location, in memory.

# Training a model:
When running either "main_image.py", "main_text.py" or "non_rl_main_image.py", be sure to also supply which model you want to train. Each argument can be done like this:

python main_image.py --LSTM --margin-sampling --margin-size 3 --name "LSTM_margin_3"

This will result in a LSTM network, with margin sampling of CMS=3, with the name "LSTM_margin_3" being trained. All commands are those below:

## main_image.py -h

usage: main_image.py [-h] [--batch-size N] [--mini-batch-size N]
                     [--test-batch-size N] [--episode-size N] [--epochs N]
                     [--start-epoch N] [--class-vector-size N] [--no-cuda]
                     [--load-checkpoint LOAD_CHECKPOINT]
                     [--load-best-checkpoint LOAD_BEST_CHECKPOINT]
                     [--load-test-checkpoint LOAD_TEST_CHECKPOINT] [--name S]
                     [--seed N] [--margin-sampling] [--margin-size N]
                     [--margin-time N] [--LSTM] [--NTM] [--LRUA]

### PyTorch Reinforcement Learning For Images:
```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 50) (default:
                        50)
  --mini-batch-size N   How many episodes to train on at a time (default: 1)
                        (default: 50)
  --test-batch-size N   How many episodes to test on at a time (default: 1)
                        (default: 32)
  --episode-size N      input episode size for training (default: 30)
                        (default: 30)
  --epochs N            number of epochs to train (default: 2000) (default:
                        200)
  --start-epoch N       starting epoch (default: 1) (default: 1)
  --class-vector-size N
                        Number of classes per episode (default: 3) (default:
                        3)
  --no-cuda             enables CUDA training (default: True)
  --load-checkpoint LOAD_CHECKPOINT
                        path to latest checkpoint (default: none) (default:
                        pretrained/IMAGE_ntm/checkpoint.pth.tar)
  --load-best-checkpoint LOAD_BEST_CHECKPOINT
                        path to best checkpoint (default: none) (default:
                        pretrained/reinforced_ntm/best.pth.tar)
  --load-test-checkpoint LOAD_TEST_CHECKPOINT
                        path to best checkpoint (default: none) (default:
                        pretrained/reinforced_ntm/testpoint.pth.tar)
  --name S              name of file (default: reinforced_ntm_LAST)
  --seed N              random seed (default: 1) (default: 1)
  --margin-sampling     Enables margin sampling for selecting clases to train
                        on (default: False)
  --margin-size N       Multiplier for number of classes in pool of classes
                        during margin sampling (default: 2)
  --margin-time N       Number of samples per class during margin sampling
                        (default: 4)
  --LSTM                Enables LSTM as chosen Q-network (default: False)
  --NTM                 Enables NTM as chosen Q-network (default: False)
  --LRUA                Enables LRUA as chosen Q-network (default: False)
```
## main_text.py -h

usage: main_text.py [-h] [--batch-size N] [--test-batch-size N]
                    [--episode-size N] [--epochs N] [--start-epoch N]
                    [--class-vector-size N] [--no-cuda]
                    [--load-checkpoint LOAD_CHECKPOINT]
                    [--load-best-checkpoint LOAD_BEST_CHECKPOINT]
                    [--load-test-checkpoint LOAD_TEST_CHECKPOINT]
                    [--name NAME] [--seed S] [--margin-sampling]
                    [--margin-size S] [--margin-time S] [--LSTM] [--NTM]
                    [--LRUA] [--INH] [--REUTERS] [--embedding-size S]
                    [--sentence-length S] [--nof-sentences S]
                    [--dictionary-size S]

### PyTorch Reinforcement Learning For Text:
```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        Input batch size for training (default: 32)
  --test-batch-size N   Input batch size for testing (default: 32)
  --episode-size N      Input episode size for training (default: 30)
  --epochs N            Number of epochs to train (default: 60000)
  --start-epoch N       Starting epoch (default: 1)
  --class-vector-size N
                        Number of classes per episode (default: 3)
  --no-cuda             Enables CUDA training (default: True)
  --load-checkpoint LOAD_CHECKPOINT
                        Path to latest checkpoint (default:
                        pretrained/headlines_lstm/checkpoint.pth.tar)
  --load-best-checkpoint LOAD_BEST_CHECKPOINT
                        Path to best checkpoint (default:
                        pretrained/headlines_lstm/best.pth.tar)
  --load-test-checkpoint LOAD_TEST_CHECKPOINT
                        Path to post test-checkpoint (default:
                        pretrained/headlines_lstm/testpoint.pth.tar)
  --name NAME           Name of file (default: headlines_lstm)
  --seed S              random seed for predictable RNG behaviour (default: 1)
  --margin-sampling     Enables margin sampling for selecting clases to train
                        on (default: False)
  --margin-size S       Multiplier for number of classes in pool of classes
                        during margin sampling (default: 2)
  --margin-time S       Number of samples per class during margin sampling
                        (default: 4)
  --LSTM                Enables LSTM as chosen Q-network (default: False)
  --NTM                 Enables NTM as chosen Q-network (default: False)
  --LRUA                Enables LRUA as chosen Q-network (default: False)
  --INH                 Enables INH as chosen dataset (default: False)
  --REUTERS             Enables REUTERS as chosen dataset (default: False)
  --embedding-size S    Size of word-embedding layer (default: 128)
  --sentence-length S   Number of words in a sentence. NOTE: Changing this
                        will force you to recompile the datasets (default: 12)
  --nof-sentences S     Number of sentences collected from a sample. NOTE:
                        Changing this will force you to recompile the datasets
                        (default: 1)
  --dictionary-size S   Dictionary max size, with 0 and N + 1 as reserved
                        tokens. NOTE: Changing this will force you to
                        recompile the datasets (default: 10000)
  ```
  ## non_rl_main_image.py -h
  
  usage: non_rl_main_image.py [-h] [--batch-size N] [--test-batch-size N]
                            [--episode-size N] [--epochs N] [--start-epoch N]
                            [--class-vector-size N] [--no-cuda]
                            [--load-checkpoint LOAD_CHECKPOINT]
                            [--load-best-checkpoint LOAD_BEST_CHECKPOINT]
                            [--load-test-checkpoint LOAD_TEST_CHECKPOINT]
                            [--name NAME] [--seed S] [--margin-sampling]
                            [--margin-size N] [--margin-time N] [--LSTM]
                            [--NTM] [--LRUA]

### PyTorch Deterministic Meta-Learning:
```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 50) (default:
                        32)
  --test-batch-size N   How many episodes to test on at a time (default: 1)
                        (default: 32)
  --episode-size N      input episode size for training (default: 30)
                        (default: 50)
  --epochs N            number of epochs to train (default: 2000) (default:
                        100000)
  --start-epoch N       starting epoch (default: 1) (default: 1)
  --class-vector-size N
                        Number of classes per episode (default: 3) (default:
                        5)
  --no-cuda             enables CUDA training (default: True)
  --load-checkpoint LOAD_CHECKPOINT
                        path to latest checkpoint (default: none) (default:
                        pretrained/deterministic_lstm_5/checkpoint.pth.tar)
  --load-best-checkpoint LOAD_BEST_CHECKPOINT
                        path to best checkpoint (default: none) (default:
                        pretrained/deterministic_lstm_5/best.pth.tar)
  --load-test-checkpoint LOAD_TEST_CHECKPOINT
                        path to best checkpoint (default: none) (default:
                        pretrained/deterministic_lstm_5/testpoint.pth.tar)
  --name NAME           name of file (default: deterministic_lstm_5)
  --seed S              random seed (default: 1) (default: 1)
  --margin-sampling     Enables margin sampling for selecting clases to train
                        on (default: False)
  --margin-size N       Multiplier for number of classes in pool of classes
                        during margin sampling (default: 2)
  --margin-time N       Number of samples per class during margin sampling
                        (default: 4)
  --LSTM                Enables LSTM as chosen Q-network (default: False)
  --NTM                 Enables NTM as chosen Q-network (default: False)
  --LRUA                Enables LRUA as chosen Q-network (default: False)
```
