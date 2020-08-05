# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

mish = Mish()
x = torch.linspace(-10, 10, 1000)
y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()
