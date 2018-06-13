"""An NTM's memory implementation."""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_memory_matrix(w, t):
    w_to_plot = w

    imgs = []
    x_dim = 5
    y_dim = 8

    memory_x = 1
    memory_y = 1
    memory_slots = memory_x*memory_y

    name = "MNIST_ntm_single/"
    memory_vector = "memory_slot/"
    
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


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


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
        self.register_buffer('mem_bias', torch.Tensor(N, M))
        #self.register_buffer('mem_bias', self.mem_bias.data)

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

    def write(self, w, e, a, head_nr=-1, t=0):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        """
        print("\nBEFORE:")
        print("Memory: ", self.prev_mem)
        print("Memory Size: ", self.prev_mem.size())
        input("OK")
        """
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

        #if (head_nr == 0):
        #    plot_memory_matrix(self.memory[0], t)
        """
        print("\nAFTER:")
        print("Memory: ", self.memory)
        print("Memory Size: ", self.memory.size())
        input("OK")
        """

    

    def address(self, k, w_prev, head_nr=-1, t=0):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """

        # Content focus
        w_r = self._similarity_mann(k)
        #if (head_nr == 0):
            #plot_memory_matrix(w_r, t)

        return w_r

        """
        print("Read Weights: ", w_r)
        print("Read Weights Size: ", w_r.size())
        print("SUM: ", torch.sum(w_r[0, :]))
        input("OK")
        """
        # Location focus
        #w_g = self._interpolate(w_prev, w_r, g)
        """
        print("Interpol Weights: ", w_g)
        print("Interpol Weights Size: ", w_g.size())
        print("SUM: ", torch.sum(w_g[0, :]))
        input("OK")
        """
        #ŵ = self._shift(w_g, s)
        """
        print("Shifted Weights: ", ŵ)
        print("Shifted Weights Size: ", ŵ.size())
        print("SUM: ", torch.sum(ŵ[0, :]))
        input("OK")
        """
        #w_t = self._sharpen(ŵ, γ)
        """
        print("Sharpened Weights: ", w_t)
        print("Sharpened Weights Size: ", w_t.size())
        print("SUM: ", torch.sum(w_t[0, :]))
        input("OK")
        """

        return w_r


    def _similarity_mann(self, k):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _similarityMann(self, k):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = Variable(torch.zeros(wg.size()))
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        #print("Blurred: ", ŵ[0][0:2])
        #print("Sharp: ", w[0][0:2])
        #print("Size ŵ: ", ŵ.size())
        #print("Size w: ", w.size())
        #input("OK")
        return w