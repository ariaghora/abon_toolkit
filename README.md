## Available kernels:
- Radial basis function (RBF) kernel
- Polynomial kernel

## Installation:

`pip install --upgrade git+https://github.com/ariaghora/torch_kernel.git`

## Usage:

```python
import torch
from torch_kernel import rbf_kernel

x = torch.randn(4, 3)
y = torch.randn(5, 3)
k = rbf_kernel(x, y)

print(k)
```
