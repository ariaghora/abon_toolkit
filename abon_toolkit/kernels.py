import torch

def euclidean_distances(x, y, squared=True):
    n_1, n_2 = x.shape[0], y.shape[0]
    
    norms_1 = torch.sum(x**2, dim=1, keepdim=True)
    norms_2 = torch.sum(y**2, dim=1, keepdim=True)
    
    norms = (norms_1.expand(n_1, n_2) +
             norms_2.transpose(0, 1).expand(n_1, n_2))
    
    distances_squared = norms - 2 * x.mm(y.t())
    return distances_squared if squared else torch.sqrt(torch.abs(distances_squared))

def mmd_rbf(x, y, deg=2):
    m = x.shape[0]
    n = y.shape[0]
    return (
        (torch.sum(rbf_kernel(x, x)) / (m**2)) - 
        (torch.sum(rbf_kernel(x, y)) * (2 / (m*n))) + 
        (torch.sum(rbf_kernel(y, y)) / (n**2)))

def rbf_kernel(x, y, gamma=None):
    assert(x.shape[1] == y.shape[1])
    
    if gamma is None: 
        gamma = 1.0 / x.shape[1]
        
    K  = euclidean_distances(x, y)
    K *= -gamma
    return torch.exp(K)

def polynomial_kernel(x, y, degree=2):
    assert(x.shape[1] == y.shape[1])
    return ((x @ y.T) * (1/x.shape[1]) + 1) ** degree