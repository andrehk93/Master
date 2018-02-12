import torch
import torch.nn as nn
import torch.autograd as autograd



class ReinforcedLSTM(nn.Module):
    def __init__(self, IMAGE_SIZE, HIDDEN_NODES, HIDDEN_LAYERS, OUTPUT_CLASSES, BATCH_SIZE, CUDA):
        super(ReinforcedLSTM, self).__init__()

        # Parameters
        self.image_size = IMAGE_SIZE
        self.hidden_nodes = HIDDEN_NODES
        self.hidden_layers = HIDDEN_LAYERS
        self.output_size = OUTPUT_CLASSES
        self.gpu = CUDA

        print("Model Input Size: ", str(self.image_size + self.output_size))
        print("Model Output Size: ", str(self.output_size + 1))

        # Architecture
        self.lstm = nn.LSTM(self.image_size + self.output_size, self.hidden_nodes)
        self.hidden2probs = nn.Linear(self.hidden_nodes, self.output_size + 1)


    def init_hidden(self, batch_size):
        if (self.gpu):
            h0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes)).cuda()
            c0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes)).cuda()
        else:
            h0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes))
            c0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes))

        return (h0, c0)

    # Not sure if necessary:
    def reset_hidden(self, batch_size):
        hidden = self.init_hidden(batch_size)
        return (hidden[0].detach(), hidden[1].detach())

        
    def forward(self, x, hidden, seq=1):
        batch_size = hidden[1].size()[1]
        x = x.view(seq, batch_size, -1)
        lstm_out, next_hidden = self.lstm(x, hidden)
        x = self.hidden2probs(lstm_out[-1])
        return x, next_hidden
