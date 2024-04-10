import argparse
from collections import defaultdict
from extra.optimization.helpers import kern_str_to_lin
from test.external.fuzz_linearizer import compare_linearizer
from tinygrad.helpers import colored
from tinygrad.features.graph import print_tree
from tinygrad.features.search import time_linearizer

# Use this with the LOGKERN options to verify that all executed kernels are valid and evaluate to the same ground truth results

# Example for GPT2:
# 1) Run the model to log all kernels: `PYTHONPATH=. LOGKERN=/tmp/gpt2_kerns.txt JIT=1 HALF=1 BEAM=2 CACHELEVEL=0 CAST_BEFORE_VIEW=0 python3 examples/gpt2.py --count 10 --temperature 0 --timing`   # noqa: E501
# 2) Validate the kernel correctness: `PYTHONPATH=. python3 ./test/external/verify_kernel.py --file /tmp/gpt2_kerns.txt`

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verify the correctness of one or more kernel", formatter_class=argparse.ArgumentDefaultsHelpFormatter)    # noqa: E501
  parser.add_argument("--kernel", type=str, default=None, help="a string of a tuple of (ast, applied_opts,)")
  parser.add_argument("--file", type=str, default=None, help="a file containing a tuple of ast and applied_opts, one per line")
  parser.add_argument("--rtol", type=float, default=1e-2, help="relative tolerance for numerical comparison")
  parser.add_argument("--atol", type=float, default=1e-2, help="absolute tolerance for numerical comparison")
  parser.add_argument("--timing", action='store_true', help="show final timing for the kernel")
  parser.add_argument("--expected-failures", type=int, default=0, help="the number of expected failed kernels")
  args = parser.parse_args()

  if args.kernel is not None:
    print("loading kernel from args")
    kern_strs = [args.kernel]
  elif args.file is not None:
    print(f"loading kernel from file '{args.file}'")
    with open(args.file, 'r') as file:
      kern_strs = file.readlines()
  else:
    raise RuntimeError("no kernel specified; use --kernel or --file options")

  print(f"verifying {len(kern_strs)} kernels")

  failed_ids = []
  failures = defaultdict(list)
  for i, kern_str in enumerate(kern_strs):
    print(f"testing kernel {i}")
    test_lin = kern_str_to_lin(kern_str)
    for op in test_lin.ast: print_tree(op)
    print(test_lin.colored_shape())
    (msg,rb,vv,gt) = compare_linearizer(test_lin, None, None, None, rtol=args.rtol, atol=args.atol)
    if msg != "PASS":
      failed_ids.append(i)
      failures[msg].append((test_lin.ast, test_lin.applied_opts))
    if args.timing:
      tm = time_linearizer(test_lin, rb, allow_test_size=False, cnt=10)
      print(f"final time {tm*1e6:9.0f} us")

  for msg, errors in failures.items():
    for i, (ast, opts) in enumerate(errors):
      print(f"{msg} {i} AST: {ast}")
      print(f"{msg} {i} OPTS: {opts}\n")

  print(f"tested {len(kern_strs)} kernels")
  if failures:
    print(f"{failed_ids=}")
    for msg, errors in failures.items():
      print(f"{msg}: {len(errors)}")
    if len(failed_ids) == args.expected_failures:
      print(colored(f"{len(failed_ids)} failed as expected", "yellow"))
  if len(failed_ids) != args.expected_failures:
    raise RuntimeError(f"failed on {len(failed_ids)} kernels, expected {args.expected_failures}")
  else:
    print(colored("all passed", "green"))