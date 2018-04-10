import torch
from torch import nn
from torch.autograd import Variable
from .lstm.model import ReinforcedLSTM
from .ntm.aio import EncapsulatedNTM as NTM
from .lrua.aio import EncapsulatedNTM as LRUA

# Baseline LSTM:
class ReinforcedRNN(nn.Module):

	# PARAMETERS:
	hidden_layers = 1
	hidden_nodes = 200

	def __init__(self, batch_size, cuda, classes, image_size):

		super(ReinforcedRNN, self).__init__()
		
		self.q_network = ReinforcedLSTM(image_size, self.hidden_nodes, self.hidden_layers, classes, batch_size, cuda)
		self.batch_size = batch_size
		self.gpu = cuda
	
	def reset_hidden(self):
		return self.q_network.reset_hidden(self.batch_size)

	def forward(self, inp, hidden, read_only=False, seq=1):
		return self.q_network(inp, hidden, seq=seq)


# NTM:
class ReinforcedNTM(nn.Module):

	# PARAMETERS:
	M = 40
	N = 128
	num_read_heads = 4
	num_write_heads = 1
	controller_size = 200
	controller_layers = 1
	image_size = 784


	def __init__(self, batch_size, cuda, classes, image_size):

		super(ReinforcedNTM, self).__init__()

		self.q_network = NTM(image_size + classes, classes + 1,
                self.controller_size, self.controller_layers, self.num_read_heads, self.num_write_heads, self.N, self.M)

		self.batch_size = batch_size
		self.gpu = cuda

	def reset_hidden(self):
		return self.q_network.init_sequence(self.batch_size)

	def forward(self, inp, hidden, read_only=False, seq=1):
		return self.q_network(inp, hidden, read_only)



# LRUA:
class ReinforcedLRUA(nn.Module):

	# PARAMETERS:
	M = 40
	N = 128
	num_read_heads = 4
	num_write_heads = num_read_heads
	controller_size = 200
	controller_layers = 1


	def __init__(self, batch_size, cuda, classes, image_size):

		super(ReinforcedLRUA, self).__init__()

		self.q_network = LRUA(image_size + classes, classes + 1,
                self.controller_size, self.controller_layers, self.num_read_heads, self.num_write_heads, self.N, self.M)

		self.batch_size = batch_size
		self.gpu = cuda

	def reset_hidden(self):
		return self.q_network.init_sequence(self.batch_size)

	def forward(self, inp, hidden, read_only=False, seq=1):
		return self.q_network(inp, hidden, read_only)

	






