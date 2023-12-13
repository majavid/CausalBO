from causalbo.sample_data.toy_graph import ToyGraph
from causalbo.modules import CausalMean
import torch
import pandas as pd
import matplotlib.pyplot as plt

t = ToyGraph()
m = CausalMean(['X', 'Y'], t.graph)

tens = torch.Tensor([[0],[0],[0],[0]])

print(m.forward(tens))