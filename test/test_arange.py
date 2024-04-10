import unittest
from tinygrad import Tensor, GlobalCounters

class TestArange(unittest.TestCase):
  def _get_flops(self, N):
    GlobalCounters.reset()
    Tensor.arange(N).realize()
    return GlobalCounters.global_ops

  def test_complexity(self):
    f1 = self._get_flops(256)
    f2 = self._get_flops(2560)
    print(f"{f1=}, {f2=}")
    assert f2 / f1 < 15, f"bad complexity, flops {f2/f1:.1f}X while inputs 10X"