import torch
from torch.autograd import Variable
import torch.optim as optim

def size_to_str(size):
	return '('+(', ').join(['%d' % v for v in size])+')'

seen = set()

params = None

def add_nodes(var):
	if var not in seen:
		if torch.is_tensor(var):
			print("Node: ", str(id(var)), size_to_str(var.size()))
		elif hasattr(var, 'variable'):
			u = var.variable
			name = param_map[id(u)] if params is not None else ''
			node_name = '%s\n %s' % (name, size_to_str(u.size()))
			#print("Node: ", str(id(var)), node_name)
		else:
			print("Node: ", str(id(var)), str(type(var).__name__))
		seen.add(var)

		if hasattr(var, 'next_functions'):
			for u in var.next_functions:
				if u[0] is not None:
					#print("Edge: ", str(id(u[0])), str(id(var)))
					add_nodes(u[0])
		if hasattr(var, 'saved_tensors'):
			for t in var.saved_tensors:
				#print("Edge: ", str(id(t)), str(id(var)))
				add_nodes(t)

		