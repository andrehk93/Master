"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
import torch.autograd as autograd


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers, num_classes, embedding_weight_matrix=None, embedding=False,
                 dict_size=5000, embedding_size=128):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.dict_size = dict_size
        self.embedding = embedding

        if self.embedding:
            if embedding_weight_matrix is not None:
                self.embedding_weight_matrix = torch.Tensor(embedding_weight_matrix)
                self.embedding_layer = self.create_embedding_layer(with_grad=False)

            # Non pre-trained word embeddings
            else:
                self.embedding_layer = nn.Embedding(self.dict_size, self.embedding_size)

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

    # Setup pre-trained word embedding layer
    def create_embedding_layer(self, with_grad=False):
        # Pre-trained
        return nn.Embedding.from_pretrained(self.embedding_weight_matrix, freeze=not with_grad)

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = autograd.Variable(torch.zeros(self.num_layers, batch_size, self.num_outputs))
        lstm_c = autograd.Variable(torch.zeros(self.num_layers, batch_size, self.num_outputs))
        return lstm_h, lstm_c

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state, prev_reads=None, class_vector=None, seq=1):
        if not self.embedding:
            x = x.unsqueeze(0)
            outp, state = self.lstm(x, prev_state)

        # Different procedure for text embedding:
        else:
            # Get the word-vectors (ONLY supports one sentence - hence .squeeze()):
            x = self.embedding_layer(x).squeeze()
            lstm_input = []

            # Appending each word-vector with the associated class:
            for i in range(x.size()[1]):
                embedding_with_memory = torch.cat([x[:, i]] + prev_reads, dim=1)
                lstm_input.append(torch.cat((class_vector, embedding_with_memory), dim=1))
            lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[1], x.size()[0], -1)

            # Sending to LSTM:
            outp, state = self.lstm(lstm_input, prev_state)
        
        return outp[-1], state
