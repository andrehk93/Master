def parse_arguments(parser):

    """
    Training and Testing parameters:
    """
    # Batch size:
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for training')

    # Test batch size:
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='Input batch size for testing')

    # Episode size:
    parser.add_argument('--episode-size', type=int, default=30, metavar='N',
                        help='Input episode size for training')

    # Epochs:
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='Number of epochs to train')

    # Starting Epoch:
    parser.add_argument('--start-epoch', type=int, default=1, metavar='N',
                        help='Starting epoch')

    # Nof Classes:
    parser.add_argument('--class-vector-size', type=int, default=3, metavar='N',
                        help='Number of classes per episode')

    """
    PyTorch specific parameters:
    """
    # CUDA:
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Enables CUDA training')

    # Seed:
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed for predictable RNG behaviour')

    """
    Checkpoints- and Result-names:
    """
    # Checkpoint Loader:
    parser.add_argument('--load-checkpoint', default='pretrained/name/', type=str,
                        help='Path to latest checkpoint')

    # Network Name:
    parser.add_argument('--name', default='name', type=str,
                        help='Name of file')

    """
    Class Margin Sampling parameters:
    """
    # Margin:
    parser.add_argument('--margin-sampling', action='store_true', default=False,
                        help='Enables margin sampling for selecting clases to train on')

    # Margin size:
    parser.add_argument('--margin-size', type=int, default=2, metavar='S',
                        help='Multiplier for number of classes in pool of classes during margin sampling')

    # Margin time:
    parser.add_argument('--margin-time', type=int, default=4, metavar='S',
                        help='Number of samples per class during margin sampling')

    """
    Network architecture:
    """
    # LSTM:
    parser.add_argument('--LSTM', action='store_true', default=False,
                        help='Enables LSTM as chosen Q-network')

    # NTM:
    parser.add_argument('--NTM', action='store_true', default=False,
                        help='Enables NTM as chosen Q-network')

    # LRUA:
    parser.add_argument('--LRUA', action='store_true', default=False,
                        help='Enables LRUA as chosen Q-network')

    """
    Dataset:
    """
    # MNIST:
    parser.add_argument('--MNIST', action='store_true', default=False,
                        help='Enables MNIST as chosen dataset')

    # OMNIGLOT:
    parser.add_argument('--OMNIGLOT', action='store_true', default=False,
                        help='Enables OMNIGLOT as chosen dataset')

    # INH:
    parser.add_argument('--INH', action='store_true', default=False,
                        help='Enables INH as chosen dataset')

    # REUTERS:
    parser.add_argument('--REUTERS', action='store_true', default=False,
                        help='Enables REUTERS as chosen dataset')

    # QA:
    parser.add_argument('--QA', action='store_true', default=False,
                        help='Enables QA as chosen dataset')

    """
    Pretrained GloVe or fastText vectors:
    """
    # GloVe:
    parser.add_argument('--GLOVE', action='store_true', default=False,
                        help='Enables GloVe pre-trained word vectors')

    # fastText:
    parser.add_argument('--FAST', action='store_true', default=False,
                        help='Enables GloVe pre-trained word vectors')

