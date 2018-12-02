import torch
import torch.nn as nn
import torch.autograd as autograd
import os


class ReinforcedLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_NODES, HIDDEN_LAYERS, INPUT_CLASSES, BATCH_SIZE, CUDA, weights_matrix=None, EMBEDDING=False, DICT_SIZE=5000, NON_RL=False):
        super(ReinforcedLSTM, self).__init__()

        # Parameters
        self.input_size = INPUT_SIZE
        self.embedding = EMBEDDING
        self.dict_size = DICT_SIZE
        self.hidden_nodes = HIDDEN_NODES
        self.hidden_layers = HIDDEN_LAYERS
        self.input_classes = INPUT_CLASSES
        self.gpu = CUDA
        if (EMBEDDING):
            self.weights_matrix = torch.Tensor(weights_matrix)

        print("Model Input Size: ", str(self.input_size + self.input_classes))
        print("Model Output Size: ", str(self.input_classes + 1))

        if (EMBEDDING):
            self.embedding_layer, num_embeddings, embedding_dim = self.create_embedding_layer(non_trainable=True)

        # Architecture
        self.lstm = nn.LSTM(self.input_size + self.input_classes, self.hidden_nodes)
        if (NON_RL):
            self.hidden2probs = nn.Linear(self.hidden_nodes, self.input_classes)
        else:
            self.hidden2probs = nn.Linear(self.hidden_nodes, self.input_classes + 1)

    def create_embedding_layer(self, non_trainable=False):
        num_embeddings, embedding_dim = self.weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': self.weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = True
        return emb_layer, num_embeddings, embedding_dim


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
        
        # If we handle text, we need additional embeddings for each token/word:
        if (self.embedding):
            try:
                x = self.embedding_layer(x)
            except RuntimeError as e:
                print(e)
                return False

            lstm_input = []

            
            if (len(x.size()) == 3):
                x = x.unsqueeze(1)

            for i in range(x.size()[len(x.size()) - 2]):
                for j in range(x.size()[1]):
                    lstm_input.append(torch.cat((class_vector, x[:, j, i]), 1))
            if (x.size()[1] == 1):
                lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[2], x.size()[0], -1)
                lstm_out, next_hidden = self.lstm(lstm_input, hidden)
            else:
                next_hidden = hidden
                lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[1], x.size()[2], x.size()[0], -1)
                for sentence in range(seq):
                    lstm_out, next_hidden = self.lstm(lstm_input[sentence], next_hidden)
            

            x = self.hidden2probs(lstm_out[-1])

        else:
            x = x.view(seq, batch_size, -1)
            lstm_out, next_hidden = self.lstm(x, hidden)

            if (seq > 1):
                x = self.hidden2probs(lstm_out)
            else:
                x = self.hidden2probs(lstm_out[-1])
        
        return x, next_hidden
