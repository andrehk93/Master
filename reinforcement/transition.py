from collections import namedtuple

# FROM PyTorch Reinforcement Learning DQN Tutorial:
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))
