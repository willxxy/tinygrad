import numpy as np
# from tinygrad.helpers import Timing
from tinygrad import Tensor

def print_tensor_properties(tg_tensor: Tensor):
    print("Tensor:", tg_tensor)
    print("Realized:", tg_tensor.realize())
    print("Data:", tg_tensor.data())
    print("Numpy:", tg_tensor.numpy())
    print("Shape:", tg_tensor.shape)
    print("Dtype:", tg_tensor.dtype)

if __name__ == "__main__":
    tg_tensor_1 = Tensor([1, 2, 3])
    print(tg_tensor_1)
    np_arr_1 = np.array([1, 2, 3])
    print(np_arr_1, np_arr_1.dtype)
    tg_tensor_2 = Tensor(np_arr_1)
    print(tg_tensor_2)

    full = Tensor.full(shape=(2,3), fill_value=5)
    print_tensor_properties(full)

    zeros = Tensor.zeros(shape=(2,3))
    print_tensor_properties(zeros)

    ones = Tensor.ones(shape=(2,3))
    print_tensor_properties(ones)