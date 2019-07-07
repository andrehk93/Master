# Active One-Shot Learning with Memory Augmented Neural Networks
This is the repository for my Masters-project "Active One-Shot Learning with Memory Augmented Neural Networks". 
It features three different RNN models, an LSTM, a NTM and an LRUA. The goal of active one-shot learning is to be able
to classify samples after only seeing them once, with a high accuracy. This particular project involves the use of reinforcement learning (RL)
more specifically Q-learning to learn a strategy for stream-based sample classification (images & text).

# Prerequisites

```
Versions:
Python      3.*
PyTorch     0.4.1
Numpy       1.14.3
Matplotlib  2.2.2
```

## Datasets
### Training on the OMNIGLOT Dataset:
Download the [OMNIGLOT](https://github.com/brendenlake/omniglot/tree/master/python) dataset (images_background.zip & images_evaluation.zip). Unzip both files and put BOTH folders in a folder you call "raw", which you put in data/omniglot/, and the scripts will do the rest.

### Training on the INH Dataset:
Download the [INH](https://www.kaggle.com/therohk/india-headlines-news-dataset) dataset (You will need to sign in/sign up to Kaggle.com in order to do this). Unzip the file, and place the resulting .csv file called "india-news-headlines.csv" into "data/text/headlines/". Then run the script in the same folder called "csv_to_folders.py". This script will create folders for each headline category, and put each headline into it's corresponding category. This will take some time, as the dataset is over 2.7 million rows long. NOTE: The script may fail at the end, but this is due to some compromises made, and is not of concern, all files and folders have been made at this point!
 
NOTE: The first time you run main.py, the scripts will create word index vectors and store these for training. Thus, to not having to do this multiple times, be sure that the wanted sentence length, number of sentences and dictionary size are as you want them to be.

### Training on the Reuters Dataset:
Download the [Reuters](http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-115Cat.tar.gz) dataset. Unzip the file, and from the unzipped folder, put bot the "test" and "training"-folders inside a folder called /raw inside data/reuters/. Run main_text.py and you should be good to go.

## Word Vectors
Each model can use three different types of word vectors. 

1. Non pre-trained (Not recommended)
2. GloVe (current default)
3. fastText 

In order to be able to use the pre-trained word vectors, they have to be downloaded and put in the correct directory.

### GloVe
Download the [GLOVE](http://nlp.stanford.edu/data/glove.6B.zip) word-vectors. Put in the folder data/text/glove, and make sure the filenames in utils/text/glove.py are identical to what you call the downloaded .txt file. 

#### Embedding size
GloVe comes with a few different sizes of word embeddings, which also can be used as long as the proper naming convention is used. For example, if you want to use word embeddings of size 50 with an LSTM on the INH dataset, just add the corresponding GloVe file to the folder data/text/glove/ under the name provided underneath together with the command (Windows):

```
glove.6B.50d.txt

python main.py --LSTM --INH --embedding-size 50
```


### FastText
Download the [FastText](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) word-vectors and unzip to data/text/fast_text. Only supports the 300 dimension word embeddings.

## Models
Both datasets can be trained on three different models:

1. LSTM Baseline model:
Implemented from [Active One-Shot Learning](https://cs.stanford.edu/~woodward/papers/active_one_shot_learning_2016.pdf). This is used as a baseline for my experiments with different memory structures.

2. NTM Model:
Implemented partially from [pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm), with added functionality similar to "Active One-Shot Learning" and [Meta-Learning with Memory-Augmented Neural Networks](http://proceedings.mlr.press/v48/santoro16.pdf). 

3. LRUA Model:
Simply an augmented version of the NTM model, similar to the LRUA in [Meta-Learning with Memory-Augmented Neural Networks](http://proceedings.mlr.press/v48/santoro16.pdf). The only difference is that the number of read heads is identical to the number of write heads, and that every memory location is either written to the least used location, or simply the first location, in memory.

# Training a model:
First of all, any changes to the specific model architecture (LSTM size, NTM memory sizes, etc.) can be done in "models/reinforcement_models.py". Needless to say, changing architecture and then loading an earlier checkpoint of a model will not work.

When running "main.py", be sure to also supply which model you want to train. Each argument can be done like this:
```
python main.py --LSTM --margin-sampling --margin-size 3 
```
This will result in a LSTM network, with margin sampling of MARGIN_SIZE=3 being trained. 

All commands can be seen at the end of this readme.

## Model names (File names)
The naming scheme is automated based on the parameters of the model, but it's possible to both overwrite the name, and give it a postfix.

```
--name-postfix _something

Results in:
results/automated_name_based_on_parameters_something/

--name _something

Results in:
results/_something/
```

## Running the Training Procedure
To run the main training procedure, use the main.py file. It has only two mandatory parameters
```
Model: --LSTM | -- NTM | --LRUA
Dataset: --MNIST | --OMNIGLOT | --INH | --REUTERS | --QA
``` 
All available parameters are as follows:
```
python main.py [-h] [--batch-size N] [--test-batch-size N] [--episode-size N]
               [--epochs N] [--start-epoch N] [--class-vector-size N]
               [--embedding-size N] [--sentence-length N] [--no-cuda]
               [--seed S] [--load-checkpoint LOAD_CHECKPOINT] [--name NAME]
               [--name-postfix NAME_POSTFIX] [--margin-sampling]
               [--margin-size S] [--margin-time S] [--LSTM] [--NTM] [--LRUA]
               [--MNIST] [--OMNIGLOT] [--INH] [--REUTERS] [--QA] [--GLOVE]
               [--FAST]
```
### Detailed Explanation of Parameters
```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        Input batch size for training (default: 32)
  --test-batch-size N   Input batch size for testing (default: 32)
  --episode-size N      Input episode size for training (default: 30)
  --epochs N            Number of epochs to train (default: 100000)
  --start-epoch N       Starting epoch (default: 1)
  --class-vector-size N
                        Number of classes per episode (default: 3)
  --embedding-size N    size of embedding layer (default: 100)
  --sentence-length N   Number of words in each sentence (default: 6)
  --no-cuda             Enables CUDA training (default: True)
  --seed S              random seed for predictable RNG behaviour (default: 1)
  --load-checkpoint LOAD_CHECKPOINT
                        Path to latest checkpoint (default: pretrained/name/)
  --name NAME           Name of file (Will be overwritten!) (default: name)
  --name-postfix NAME_POSTFIX
                        Custom name to append to the end of generated name (if
                        duplicate model structures) (default: )
  --margin-sampling     Enables margin sampling for selecting clases to train
                        on (default: False)
  --margin-size S       Multiplier for number of classes in pool of classes
                        during margin sampling (default: 2)
  --margin-time S       Number of samples per class during margin sampling
                        (default: 4)
  --LSTM                Enables LSTM as chosen Q-network (default: False)
  --NTM                 Enables NTM as chosen Q-network (default: False)
  --LRUA                Enables LRUA as chosen Q-network (default: False)
  --MNIST               Enables MNIST as chosen dataset (default: False)
  --OMNIGLOT            Enables OMNIGLOT as chosen dataset (default: False)
  --INH                 Enables INH as chosen dataset (default: False)
  --REUTERS             Enables REUTERS as chosen dataset (default: False)
  --QA                  Enables QA as chosen dataset (default: False)
  --GLOVE               Enables GloVe pre-trained word vectors (default:
                        False)
  --FAST                Enables GloVe pre-trained word vectors (default:
                        False)
```

For text models, GloVE is the default choice of word vectors right now, even when no pretrained word embedding is supplied. This might get changed later.
