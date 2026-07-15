from typing import Callable
from tinygrad.nn.optim import SGD
from tinygrad.nn.datasets import mnist
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, function, Context
from tinygrad.helpers import getenv, colored, trange

class TinyNet:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
        nn.Conv2d(1, 32, 5), Tensor.relu,
        nn.Conv2d(32, 32, 5), Tensor.relu,
        nn.BatchNorm(32), Tensor.max_pool2d,
        nn.Conv2d(32, 64, 3), Tensor.relu,
        nn.Conv2d(64, 64, 3), Tensor.relu,
        nn.BatchNorm(64), Tensor.max_pool2d,
        lambda x: x.flatten(1), nn.Linear(576, 10)]

    @function
    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

    @TinyJit
    @Context(TRAINING=1)
    def train_step(self, X_train: Tensor, Y_train: Tensor) -> Tensor:
        optimizer.zero_grad()
        samples = Tensor.randint(getenv("BS", 64), high = X_train.shape[0])
        loss = self(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
        return loss.realize(*optimizer.schedule_step())

    def get_test_acc(self, X_test: Tensor, Y_test: Tensor) -> Tensor:
        return (self(X_test).argmax(axis=1) == Y_test).mean() * 100

if __name__ == "__main__":
    tiny_net = TinyNet()
    optimizer = SGD(nn.state.get_parameters(tiny_net), lr=3e-4)
    X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

    test_acc = float("nan")
    for i in (t:=trange(getenv("STEPS", 70))):
        GlobalCounters.reset()
        loss = tiny_net.train_step(X_train, Y_train)
        if i % 10 == 9:
            test_acc = tiny_net.get_test_acc(X_test, Y_test).item()
        t.set_description(f"loss: {loss.item():.4f}, test_acc: {test_acc:.2f}%")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))