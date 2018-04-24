"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers, num_classes, embedding=False, dict_size=5000, embedding_size=128):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.dict_size = dict_size
        self.embedding = embedding

        if (self.embedding):
            self.embedding_layer = nn.Embedding(self.dict_size, self.embedding_size)
        

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state, prev_reads=None, class_vector=None, seq=1):
        if (not self.embedding):
            x = x.unsqueeze(0)
            outp, state = self.lstm(x, prev_state)
        else:
            x = self.embedding_layer(x)
            lstm_input = []
            
            for i in range(x.size()[1]):
                embedding_with_memory = torch.cat([x[:, i]] + prev_reads, dim=1)
                lstm_input.append(torch.cat((class_vector, embedding_with_memory), dim=1))
            lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[1], x.size()[0], -1)
            outp, state = self.lstm(lstm_input, prev_state)
        
        return outp[-1], state