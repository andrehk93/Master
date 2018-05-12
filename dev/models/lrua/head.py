"""NTM Read/Write Head."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.
        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, n, w_prev, access):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        gamma = 0.95

        # READ:
        if (access == 1):
            w_r = self.memory.address(k, β, g, n, gamma, w_prev, access)
            return w_r

        # WRITE:
        else:
            w_u, w_r, w_w, w_lu = self.memory.address(k, β, g, n, gamma, w_prev, access)
            return w_u, w_r, w_w, w_lu


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g sizes from the paper
        self.read_lengths = [self.M, 1, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step weightings (1):
        return Variable(torch.zeros(batch_size, 1, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_read.weight, gain=1.4)
        nn.init.normal(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev, n):
        """NTMReadHead forward function.
        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        k, β, g = _split_cols(o, self.read_lengths)

        # Read from memory
        #w_u, w_r, w_w, w_lu = self._address_memory(k, β, g, n, w_prev, 1)
        w_r = self._address_memory(k, β, g, n, w_prev, 1)
        r = self.memory.read(w_r)

        #w = torch.cat((w_u, w_r, w_lu), dim=0).view(w_u.size()[0], 3, w_u.size()[1])
        
        return r, w_r


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, w_r_prev, w_w_prev sizes from the paper
        self.write_lengths = [self.M, 1, 1]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return Variable(torch.zeros(batch_size, 3, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
        nn.init.normal(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev, n):
        """NTMWriteHead forward function.
        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        k, β, g = _split_cols(o, self.write_lengths)

        # Address memory
        w_u, w_r, w_w, w_lu = self._address_memory(k, β, g, n, w_prev, 0)

        # With LRUA we use the cosine similarity vector wc for writing to memory
        self.memory.lrua_write(w_w, k)

        w = torch.cat((w_u, w_r, w_lu), dim=1).view(w_u.size()[0], 3, w_u.size()[1])

        return w