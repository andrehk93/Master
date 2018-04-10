"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from torch.autograd import Variable

from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_read_heads, num_write_heads, N, M):
        """Initialize an EncapsulatedNTM.
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_read_heads + num_write_heads
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = NTMMemory(N, M)
        controller = LSTMController(num_inputs + M*num_read_heads, controller_size, controller_layers)
        heads = nn.ModuleList([])
        for i in range(num_read_heads):
            heads += [
                NTMReadHead(memory, controller_size),
            ]
        for i in range(num_write_heads):
            heads += [
                NTMWriteHead(memory, controller_size)
            ]
        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)
        return self.previous_state

    def forward(self, x=None, previous_state=None, read_only=False):
        # For testing copy-task:
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        # For RL:
        if (previous_state == None):
            o, self.previous_state = self.ntm(x, self.previous_state, read_only)
            return o, self.previous_state
        else:
            return self.ntm(x, previous_state, read_only)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
