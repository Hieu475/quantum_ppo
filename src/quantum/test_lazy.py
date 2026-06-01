import torch
import torch.nn as nn
net = nn.Sequential(nn.LazyLinear(10))
for m in net.modules():
    if isinstance(m, nn.Linear):
        print(m.weight)
        nn.init.orthogonal_(m.weight)
