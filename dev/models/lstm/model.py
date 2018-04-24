import torch
import torch.nn as nn
import torch.autograd as autograd



class ReinforcedLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_NODES, HIDDEN_LAYERS, INPUT_CLASSES, BATCH_SIZE, CUDA, OUTPUT_CLASSES=3, EMBEDDING=False, DICT_SIZE=5000):
        super(ReinforcedLSTM, self).__init__()

        # Parameters
        self.input_size = INPUT_SIZE
        self.embedding = EMBEDDING
        self.dict_size = DICT_SIZE
        self.hidden_nodes = HIDDEN_NODES
        self.hidden_layers = HIDDEN_LAYERS
        self.output_size = OUTPUT_CLASSES
        self.input_classes = INPUT_CLASSES
        self.gpu = CUDA

        print("Model Input Size: ", str(self.input_size + self.input_classes))
        print("Model Output Size: ", str(self.output_size + 1))

        if (EMBEDDING):
            self.embedding_layer = nn.Embedding(self.dict_size, self.input_size)

        # Architecture
        self.lstm = nn.LSTM(self.input_size + self.input_classes, self.hidden_nodes)
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

        
    def forward(self, x, hidden, class_vector=None, seq=1):
        batch_size = hidden[1].size()[1]
        
        d_size = 20000
        # If we handle text, we need additional embeddings for each token/word:
        if (self.embedding):
            try:
                x = self.embedding_layer(x)
            except RuntimeError as e:
                print(e)
                print(x)
            lstm_input = []
            for i in range(x.size()[1]):
                lstm_input.append(torch.cat((class_vector, x[:, i]), 1))
            lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[1], x.size()[0], -1)
            lstm_out, next_hidden = self.lstm(lstm_input, hidden)
        else:
            x = x.view(seq, batch_size, -1)
            lstm_out, next_hidden = self.lstm(x, hidden)
            
        x = self.hidden2probs(lstm_out[-1])
        return x, next_hidden
