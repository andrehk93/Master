from torch import nn
from .lstm.model import ReinforcedLSTM
from .ntm.aio import EncapsulatedNTM as NTM


# Baseline LSTM:
class ReinforcedRNN(nn.Module):
    # PARAMETERS:
    hidden_layers = 1
    hidden_nodes = 200

    def __init__(self, batch_size, cuda, classes, input_size, embedding_weight_matrix=None,
                 embedding=False, dict_size=5000):

        super(ReinforcedRNN, self).__init__()
        self.q_network = ReinforcedLSTM(input_size, self.hidden_nodes, self.hidden_layers, classes, batch_size,
                                        cuda, embedding_weight_matrix=embedding_weight_matrix, EMBEDDING=embedding,
                                        DICT_SIZE=dict_size + 2)

        self.batch_size = batch_size
        self.gpu = cuda

    def create_embedding_layer(self, non_trainable=False):
        self.q_network.create_embedding_layer(non_trainable)

    def reset_hidden(self, batch_size=0):
        if batch_size == 0:
            return self.q_network.reset_hidden(self.batch_size)
        else:
            return self.q_network.reset_hidden(batch_size)

    def forward(self, inp, hidden, class_vector=None, read_only=False, seq=1, display_embeddings=False):
        return self.q_network(inp, hidden, class_vector=class_vector, seq=seq, display_embeddings=display_embeddings)


# NTM:
class ReinforcedNTM(nn.Module):
    # PARAMETERS:
    """
    M: Memory slot size
    N: Nof. Memory slots
    Controller_size: Size of LSTM controller
    """
    M = 40
    N = 128
    num_read_heads = 4
    num_write_heads = 1
    controller_size = 200
    controller_layers = 1

    def __init__(self, batch_size, cuda, classes, input_size, embedding_weight_matrix=None,
                 embedding=False, dict_size=5000):

        super(ReinforcedNTM, self).__init__()

        self.q_network = NTM(input_size + classes, classes + 1, classes, False,
                             self.controller_size, self.controller_layers, self.num_read_heads,
                             self.num_write_heads, self.N, self.M,
                             embedding_weight_matrix=embedding_weight_matrix, embedding=embedding,
                             dict_size=dict_size + 2, embedding_size=input_size)

        self.batch_size = batch_size
        self.gpu = cuda

    def create_embedding_layer(self, non_trainable=False):
        self.q_network.create_embedding_layer(non_trainable)

    def reset_hidden(self, batch_size=0):
        if batch_size == 0:
            return self.q_network.init_sequence(self.batch_size)
        else:
            return self.q_network.init_sequence(batch_size)

    def forward(self, inp, hidden, class_vector=None, read_only=False, seq=1):
        return self.q_network(inp, hidden, class_vector=class_vector, read_only=read_only)


# LRUA:
class ReinforcedLRUA(nn.Module):
    # PARAMETERS:
    """
    M: Memory slot size
    N: Nof. Memory slots
    Controller_size: Size of LSTM controller
    """
    M = 40
    N = 128
    num_read_heads = 4
    num_write_heads = num_read_heads
    controller_size = 200
    controller_layers = 1

    def __init__(self, batch_size, cuda, classes, input_size, embedding_weight_matrix=None,
                 embedding=False, dict_size=5000):

        super(ReinforcedLRUA, self).__init__()

        self.q_network = NTM(input_size + classes, classes + 1, classes, True,
                             self.controller_size, self.controller_layers, self.num_read_heads,
                             self.num_write_heads, self.N, self.M,
                             embedding_weight_matrix=embedding_weight_matrix, embedding=embedding,
                             dict_size=dict_size + 2, embedding_size=input_size)

        self.batch_size = batch_size
        self.gpu = cuda

    def create_embedding_layer(self, non_trainable=False):
        self.q_network.create_embedding_layer(non_trainable)

    def reset_hidden(self, batch_size=0):
        if batch_size == 0:
            return self.q_network.init_sequence(self.batch_size)
        else:
            return self.q_network.init_sequence(batch_size)

    def forward(self, inp, hidden, class_vector=None, read_only=False, seq=1):
        return self.q_network(inp, hidden, class_vector=class_vector, read_only=read_only)
