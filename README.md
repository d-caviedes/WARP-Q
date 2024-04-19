

# Quality Prediction For Generative Neural Speech Codecs (WARP-Q)
This code is to run the WARP-Q speech quality metric in a installable mode
https://github.com/WissamJassim/WARP-Q.git

WARP-Q is an objective, full-reference metric for perceived speech quality. It uses a subsequence dynamic time warping (SDTW) algorithm as a similarity between a reference (original) and a test (degraded) speech signal to produce a raw quality score. It is designed to predict quality scores for speech signals processed by low bit rate speech coders. 

# Install
```bash
make requirements
```

# Usage example
```python
from warpq import warpq
from torch import randn
import torch

g = torch.manual_seed(1)
preds = randn(8000)
target = randn(8000)
print(warpq(preds, target, fs=16000))
(tensor(1.4610), tensor(4.2980))
```