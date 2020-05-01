import numpy as np

def minibatchify(X, y=None, batch_size=32, shuffle=True):
    if y is not None:
        assert len(X) == len(y), 'len(X) and len(y) must be equal.'

    sz         = len(X)
    batch_size = batch_size if sz > batch_size else sz
    
    idx_shuf = list(range(sz))
    if shuffle:
        np.random.shuffle(idx_shuf)
    
    X = X[idx_shuf]
    if y is not None:
        y = y[idx_shuf]
        
    ret_X = []
    ret_y = []
    for i in range(0, sz, batch_size):
        ret_X.append(X[i:i+batch_size])
        
        if y is not None:
            ret_y.append(y[i:i+batch_size])
            
    return (ret_X, ret_y) if y is not None else ret_X