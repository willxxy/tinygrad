import numpy as np
import random
import torch
# from tinygrad.helpers import Timing
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv


def print_tg_tensor_properties(tg_tensor: Tensor):
    print("Tensor:", tg_tensor)
    print("Realized:", tg_tensor.realize())
    print("Data:", tg_tensor.data())
    print("Numpy:", tg_tensor.numpy())
    print("Shape:", tg_tensor.shape)
    print("Dtype:", tg_tensor.dtype)

def print_torch_tensor_properties(torch_tensor: torch.Tensor):
    print("Tensor:", torch_tensor)
    print("Numpy:", torch_tensor.numpy())
    print("Shape:", torch_tensor.shape)
    print("Dtype:", torch_tensor.dtype)

def print_all_close(tg_tensor_1: Tensor, tg_tensor_2: Tensor):
    print("TG All close:", Tensor.allclose(tg_tensor_1, tg_tensor_2).numpy())
    print("NP All close:", np.allclose(tg_tensor_1.numpy(), tg_tensor_2.numpy()))

def set_seed(seed):
    Tensor.manual_seed(seed)
    random.seed(seed)

hyp = {
    "seed": 201
}

if __name__ == "__main__":
    tg_tensor_1 = Tensor([1, 2, 3])
    print(tg_tensor_1)
    np_arr_1 = np.array([1, 2, 3])
    print(np_arr_1, np_arr_1.dtype)
    tg_tensor_2 = Tensor(np_arr_1)
    print(tg_tensor_2)
    print("---" * 10)

    full = Tensor.full((2,3), 5, dtype=dtypes.float32)
    print_tg_tensor_properties(full)
    print("---" * 10)

    zeros = Tensor.zeros(2,3)
    print_tg_tensor_properties(zeros)
    print("---" * 10)

    ones = Tensor.ones(2,3)
    print_tg_tensor_properties(ones)
    print_all_close(ones, ones)
    print("---" * 10)

    full_like = Tensor.full_like(full, fill_value=2)
    print_tg_tensor_properties(full_like)
    print("---" * 10)

    full_like_ones = Tensor.full_like(ones, fill_value=1)
    print_tg_tensor_properties(full_like_ones)
    print_all_close(ones, full_like_ones)
    print("---" * 10)

    full_like_zeros = Tensor.full_like(zeros, fill_value=0)
    print_tg_tensor_properties(full_like_zeros)
    print_tg_tensor_properties(zeros)
    print_all_close(zeros, full_like_zeros)
    print("---" * 10)

    zeros_like = Tensor.zeros_like(full)
    print_tg_tensor_properties(zeros_like)
    print_all_close(zeros, zeros_like)
    print("---" * 10)

    ones_like = Tensor.ones_like(full)
    print_tg_tensor_properties(ones_like)
    print_all_close(ones, ones_like)
    print("---" * 10)

    seed = getenv("SEED", hyp["seed"])

    for _ in range(5):
        set_seed(seed)
        x = Tensor.uniform(2, 3, low=0, high=1)
        print(x.numpy())

    seed = getenv("SEED", hyp["seed"])
    set_seed(seed)

    set_seed(seed)
    rands_1 = [Tensor.uniform(2, 3, low=0, high=1).realize() for _ in range(5)]

    set_seed(seed)
    rands_2 = [Tensor.uniform(2, 3, low=0, high=1).realize() for _ in range(5)]

    for i, (rand_1, rand_2) in enumerate(zip(rands_1, rands_2)):
        print(f"Tensor {i}")
        print_all_close(rand_1, rand_2)
        print("----" * 10)

    tg_randn = Tensor.randn(2, 3, dtype=dtypes.float32)
    tg_randn_2 = Tensor.randn(2, 3, dtype=dtypes.bfloat16)
    tg_multiplied_randn = tg_randn_2 * tg_randn
    print_tg_tensor_properties(tg_multiplied_randn)
    print("----" * 10)

    torch_randn = torch.randn(2, 3, dtype=torch.float32)
    torch_randn_2 = torch.randn(2, 3, dtype=torch.bfloat16)
    torch_multiplied_randn = torch_randn_2 * torch_randn
    print_torch_tensor_properties(torch_multiplied_randn)
    print("----" * 10)

    tg_randn_3 = Tensor.randn(2, 3, dtype=dtypes.float32)
    tg_randn_4 = Tensor.randn(2, 3, dtype=dtypes.float16)
    print_tg_tensor_properties(tg_randn_3)
    print("----" * 10)
    print_tg_tensor_properties(tg_randn_4)
    print("----" * 10)

    torch_randn_3 = torch.randn(2, 3, dtype=torch.float32)
    torch_randn_4 = torch.randn(2, 3, dtype=torch.float16)
    print_torch_tensor_properties(torch_randn_3)
    print("----" * 10)
    print_torch_tensor_properties(torch_randn_4)

    uniform = Tensor.uniform(2, 3, low=0, high=10)
    print_tg_tensor_properties(uniform)
    print("----" * 10)

    t4 = Tensor([1, 2, 3])
    print_tg_tensor_properties(t4)
    print("----" * 10)
    t5 = (t4 + 1) * 2
    print_tg_tensor_properties(t5)
    print("----" * 10)
    t6 = (t5 * t4).relu()
    print_tg_tensor_properties(t6)
    print("----" * 10)
    t7 = t6.log_softmax()
    print_tg_tensor_properties(t7)

    print("----" * 10)
    print("----" * 10)
    t8 = (t5 * t4).relu().log_softmax()
    print_tg_tensor_properties(t8)
    print("----" * 10)

    print_all_close(t7, t8)

    t1 = Tensor([1, 2, 3])
    m1 = t1.data()
    m2 = t1.data()

    print(m1 is m2)        # probably False
    print(m1.tolist())
    print(m2.tolist())


