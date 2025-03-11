from common.np import *
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        

class Affine:
    pass

class Softmax:
    pass


class SoftmaxWithLoss:
    pass

class Sigmoid:
    pass
