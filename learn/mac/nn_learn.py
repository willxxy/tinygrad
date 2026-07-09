# from tinygrad import Tensor, dtypes, nn
from tinygrad import nn

class TinyNet:
    def __init__(self):
        self.l1 = nn.Linear(784, 128, bias = False)
        self.l2 = nn.Linear(128, 10, bias = False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leaky_relu()
        x = self.l2(x)
        return x

if __name__ == "__main__":
    pass