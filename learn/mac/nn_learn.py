from tinygrad import Tensor, nn
from tinygrad.nn.optim import SGD
from tinygrad.nn.datasets import mnist
from tinygrad import Context
from tinygrad.helpers import getenv
import numpy as np

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
    tiny_net = TinyNet()
    optimizer = SGD(nn.state.get_parameters(tiny_net), lr=3e-4)
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

    with Context(TRAINING=1):
        for step in range(1000):
            # random sample a batch
            samp = np.random.randint(0, X_train.shape[0], size=(64))
            batch = Tensor(X_train[samp])
            # get the corresponding labels
            labels = Tensor(Y_train[samp])

            # forward pass
            out = tiny_net(batch)

            # compute loss
            loss = out.sparse_categorical_crossentropy(labels)

            # zero gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update parameters
            optimizer.step()

            # calculate accuracy
            pred = out.argmax(axis=-1)
            acc = (pred == labels).mean()

            if step % 100 == 0:
                print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")