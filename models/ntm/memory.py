"""An NTM's memory implementation."""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np


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
        :param lrua: Enables LRUA addressing scheme
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = Variable(self.mem_bias.clone().repeat(batch_size, 1, 1))

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    # Standard NTM write procedure
    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    # LRUA write procedure
    def lrua_write(self, w, k):
        """ Write to memory using the Least Recently Used Addressing scheme, used in MANN"""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        lrua = torch.matmul(w.unsqueeze(-1), k.unsqueeze(1))
        self.memory = self.prev_mem + lrua

    # Standard NTM addressing
    def address(self, k, β, g, s, γ, w_prev):
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
        w_r = self._similarity(k, β)

        # Writing to memory:
        # Location focus
        w_g = self._interpolate(w_prev, w_r, g)
        ŵ = self._shift(w_g, s)
        w_t = self._sharpen(ŵ, γ)

        return w_t

    # LRUA addressing
    def lrua_address(self, k, g, n, gamma, w_prev, access):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param n: Amount of reads to memory
        """

        # Get the cosine similarity probability:
        w_r = self._similarity_mann(k)

        # Need only read weights for reading:
        if access == 1:
            return w_r

        # Unpacking previous weights:
        w_u_prev = w_prev[:, 0]
        w_r_prev = w_prev[:, 1]
        w_lu_prev = w_prev[:, 2]

        # Calc. the write weights:
        w_w = self._interpolate(w_lu_prev, w_r_prev, g)

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
                if erase_vector.data[b][m] == 0:
                    zeroed_memory[b][m] = torch.zeros(self.M)

        self.memory = Variable(zeroed_memory)

        return w_u, w_r, w_w, w_lu

    # Utility functions
    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _similarity_mann(self, k):
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
        return w