"""An NTM's memory implementation."""
import torch
import time
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import numpy as np
import numpy
import matplotlib.pyplot as plt
import os

def plot_memory_matrix(w, t):
    w_to_plot = w

    imgs = []
    x_dim = 8
    y_dim = 16

    memory_x = 1
    memory_y = 1
    memory_slots = memory_x*memory_y

    name = "TRAIN_lrua/"
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

    #input("NEXT")


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-2:], w, w[:2]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c[1:-1]


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        #self.mem_bias = Variable(torch.Tensor(N, M))
        #self.mem_bias = Variable(torch.Tensor(N, M))
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size

        self.memory = Variable(self.mem_bias.clone().repeat(batch_size, 1, 1))
        #self.memory = Variable(torch.zeros(batch_size, self.N, self.M))

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        for b in range(self.batch_size):
            erase = torch.ger(w[b], e[b])
            add = torch.ger(w[b], a[b])
            self.memory[b] = self.prev_mem[b] * (1 - erase) + add

    def lrua_write(self, w, k, head_nr=-1, t=0):
        """ Write to memory using the Least Recently Used Addressing scheme, used in MANN"""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        lrua = torch.matmul(w.unsqueeze(-1), k.unsqueeze(1))
        self.memory = self.prev_mem + lrua
        
        #if (head_nr == 0):
        #    plot_memory_matrix(self.memory[0], t)



    def address(self, k, β, g, n, gamma, w_prev, access, head_nr=-1, t=0):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param n: Amount of reads to memory
        """

        # Get the cosine similarity probability:
        w_r = self._similarity(k)

        # Need only read weights for reading:
        if (access == 1):
            return w_r

        # Unpacking previous weights:
        w_u_prev = w_prev[:, 0]
        w_r_prev = w_prev[:, 1]
        w_lu_prev = w_prev[:, 2]

        # Calc. the write weights:
        w_w = self._interpolate(w_r_prev, w_lu_prev, g)

        # Calc. the usage weights:
        w_u = gamma*w_u_prev + w_r + w_w

        # Creating the Least Recently Used Vector, by Equation (6) from MANN:
        n_smallest_matrix = np.partition(np.array(w_u.data), n-1)[:, n-1]
        w_lu = Variable(torch.FloatTensor(((np.array(w_u.data).transpose() <= n_smallest_matrix).astype(int)).transpose()))

        # Zero out all least-used slots (from previous step):
        erase_vector = Variable(torch.ones(w_lu_prev.size()[:]).type(torch.LongTensor)) - w_lu_prev.type(torch.LongTensor)
        zeroed_memory = self.memory.data.clone()
        for b in range(len(erase_vector)):
            for m in range(len(erase_vector[b])):
                if (erase_vector.data[b][m] == 0):
                    zeroed_memory[b][m] = torch.zeros(self.M)

        self.memory = Variable(zeroed_memory)

        # Plotting memory:
        if (head_nr == 0):
            plot_memory_matrix(w_r, t)

        return w_u, w_r, w_w, w_lu

    def _similarity(self, k):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=-1)
        return w

    def _interpolate(self, w_r_prev, w_lu_prev, g):
        return g * w_r_prev + (1 - g) * w_lu_prev

    def _shift(self, wg, s):
        result = Variable(torch.zeros(wg.size()))
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w