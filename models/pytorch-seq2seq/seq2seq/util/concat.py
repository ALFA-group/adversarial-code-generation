import torch

def torch_concat(x, *y, dim=0):
    '''
    x, y: Tensors
    Returns x = [x, y]
    '''
    if x is None:
        if len(y) != 1:
            raise Exception('Invalid argument')
        else:
            x = y[0]
    else:
        for i in y:
            x = torch.cat((x, i), dim)
    return x