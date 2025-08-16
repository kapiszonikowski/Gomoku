import torch, numpy as np
from Gomoku_NN import GomokuNet
net = GomokuNet(board_size=8).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
state = np.zeros((8,8), dtype=np.int8)
out = net(state)
print(out['policy_logits'].shape, out['value'].shape)