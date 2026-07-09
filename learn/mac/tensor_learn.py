import numpy as np
# from tinygrad.helpers import Timing
from tinygrad import Tensor
from tinygrad import dtypes


def print_tensor_properties(tg_tensor: Tensor):
    print("Tensor:", tg_tensor)
    print("Realized:", tg_tensor.realize())
    print("Data:", tg_tensor.data())
    print("Numpy:", tg_tensor.numpy())
    print("Shape:", tg_tensor.shape)
    print("Dtype:", tg_tensor.dtype)

def print_all_close(tg_tensor_1: Tensor, tg_tensor_2: Tensor):
    print("TG All close:", Tensor.allclose(tg_tensor_1, tg_tensor_2).numpy())
    print("NP All close:", np.allclose(tg_tensor_1.numpy(), tg_tensor_2.numpy()))

if __name__ == "__main__":
    tg_tensor_1 = Tensor([1, 2, 3])
    print(tg_tensor_1)
    np_arr_1 = np.array([1, 2, 3])
    print(np_arr_1, np_arr_1.dtype)
    tg_tensor_2 = Tensor(np_arr_1)
    print(tg_tensor_2)
    print("---" * 10)

    full = Tensor.full((2,3), 5, dtype=dtypes.float32)
    print_tensor_properties(full)
    print("---" * 10)

    zeros = Tensor.zeros(2,3)
    print_tensor_properties(zeros)
    print("---" * 10)

    ones = Tensor.ones(2,3)
    print_tensor_properties(ones)
    print_all_close(ones, ones)
    print("---" * 10)

    full_like = Tensor.full_like(full, fill_value=2)
    print_tensor_properties(full_like)
    print("---" * 10)

    full_like_ones = Tensor.full_like(ones, fill_value=1)
    print_tensor_properties(full_like_ones)
    print_all_close(ones, full_like_ones)
    print("---" * 10)

    full_like_zeros = Tensor.full_like(zeros, fill_value=0)
    print_tensor_properties(full_like_zeros)
    print_tensor_properties(zeros)
    print_all_close(zeros, full_like_zeros)
    print("---" * 10)

    zeros_like = Tensor.zeros_like(full)
    print_tensor_properties(zeros_like)
    print_all_close(zeros, zeros_like)
    print("---" * 10)

    ones_like = Tensor.ones_like(full)
    print_tensor_properties(ones_like)
    print_all_close(ones, ones_like)
    print("---" * 10)

    rand = Tensor.rand(2, 3)
    print_tensor_properties(rand)
    print("----" * 10)

    randn = Tensor.randn(2, 3)
    print_tensor_properties(randn)
    print("----" * 10)

    uniform = Tensor.uniform(2, 3, low=0, high=10)
    print_tensor_properties(uniform)
    print("----" * 10)

