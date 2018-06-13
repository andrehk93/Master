import torch
import torch.nn as nn
import torch.autograd as autograd
import os
import matplotlib.pyplot as plt

def plot_memory_matrix(w, t):
    w_to_plot = w[1].squeeze()

    imgs = []
    x_dim = 10
    y_dim = 20

    memory_x = 1
    memory_y = 1
    memory_slots = memory_x*memory_y

    name = "cell_lstm_r2_single/"
    memory_vector = "w_r/"
    
    path = "results/memories/" + name + memory_vector
    filename = path + "t_000" + str(t)

    if (not os.path.exists(path)):
        os.makedirs(path)

    fig=plt.figure(figsize=(8, 8))
    fig.suptitle('T = ' + str(t+1), fontsize=14, fontweight='bold')


    for z in range(1, memory_slots + 1):
        img = []
        for x in range(x_dim):
            img.append([])
            for y in range(y_dim):
                img[x].append(float(w_to_plot[z-1][(x*y_dim) + y]))
        fig.add_subplot(memory_x, memory_y, z)
        plt.imshow(img, cmap="gray")

    
    plt.savefig(filename)
    plt.close()



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
        self.t = 0

        print("Model Input Size: ", str(self.input_size + self.input_classes))
        print("Model Output Size: ", str(self.output_size))

        if (EMBEDDING):
            self.embedding_layer = nn.Embedding(self.dict_size, self.input_size)

        # Architecture
        self.lstm = nn.LSTM(self.input_size + self.input_classes, self.hidden_nodes)
        self.hidden2probs = nn.Linear(self.hidden_nodes, self.output_size)


    def init_hidden(self, batch_size):
        if (self.gpu):
            h0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes)).cuda()
            c0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes)).cuda()
        else:
            h0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes))
            c0 = autograd.Variable(torch.zeros(self.hidden_layers, batch_size, self.hidden_nodes))
        self.t = 0
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
            for i in range(x.size()[1]):
                lstm_input.append(torch.cat((class_vector, x[:, i]), 1))
            lstm_input = torch.cat([lstm_input[i] for i in range(len(lstm_input))]).view(x.size()[1], x.size()[0], -1)
            lstm_out, next_hidden = self.lstm(lstm_input, hidden)
        else:
            x = x.view(seq, batch_size, -1)
            lstm_out, next_hidden = self.lstm(x, hidden)

        #plot_memory_matrix(hidden, self.t)
        #self.t += 1
        
        x = self.hidden2probs(lstm_out[-1])
        
        return x, next_hidden
