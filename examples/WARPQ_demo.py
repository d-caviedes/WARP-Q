from warpq import warpq
from torch import randn
import torch

g = torch.manual_seed(1)
preds = randn(8000)
target = randn(8000)
print(warpq(preds, target, fs=16000))
