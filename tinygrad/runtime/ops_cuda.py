from __future__ import annotations
import subprocess, hashlib, tempfile, ctypes, ctypes.util, functools, re
from pathlib import Path
from typing import Tuple, Optional, List
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import DEBUG, getenv, from_mv, to_char_p_p, init_c_var, init_c_struct_t, colored, cpu_time_execution
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator, Compiler, CompilerOptions
from tinygrad.buffer import BufferOptions
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.renderer.assembly import PTXRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl  # noqa: F401

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers  # noqa: E501
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers  # noqa: E501
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

CUDACPU = getenv("CUDACPU") == 1
if CUDACPU:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
  cuda.cuLaunchKernel = lambda src, gx, gy, gz, lx, ly, lz, shared, stream, unused_extra, args: gpuocelot_lib.ptx_run(src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), lx, ly, lz, gx, gy, gz, shared)  # type: ignore  # noqa: E501

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")  # noqa: E501

def encode_args(args, vals) -> Tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(tuple([(f'f{i}', cuda.CUdeviceptr_v2) for i in range(len(args))] +
                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs

def cu_time_execution(cb, enable=False) -> Optional[float]:
  if CUDACPU: return cpu_time_execution(cb, enable=enable)
  if not enable: return cb()
  evs = [init_c_var(cuda.CUevent(), lambda x: cuda.cuEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
  cuda.cuEventRecord(evs[0], None)
  cb()
  cuda.cuEventRecord(evs[1], None)
  check(cuda.cuEventSynchronize(evs[1]))
  cuda.cuEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: cuda.cuEventDestroy_v2(ev)
  return ret.value * 1e-3

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

class PTXCompiler(Compiler):
  compiler_opts = CompilerOptions("CUDA", suffix="PTX", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    self.version = "7.8" if arch >= "sm_89" else "7.5"
    PTXCompiler.compiler_opts = PTXCompiler.compiler_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    super().__init__(f"compile_ptx_{self.arch}")
  def render(self, name:str, uops) -> str: return PTXRenderer(name, uops).replace("TARGET", self.arch).replace("VERSION", self.version)
  def compile(self, src:str) -> bytes: return src.encode()

class CUDACompiler(Compiler):
  compiler_opts = CompilerOptions("CUDA", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024], shared_max=49152)
  def __init__(self, arch:str):
    self.arch = arch
    CUDACompiler.compiler_opts = CUDACompiler.compiler_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    check(cuda.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.compile_options = [f'--gpu-architecture={arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    if (nvrtcMajor.value, nvrtcMinor.value) >= (12, 4): self.compile_options.append("--minimal")
    super().__init__(f"compile_cuda_{self.arch}")
  def render(self, name:str, uops) -> str: return CUDARenderer(name, uops)
  def compile(self, src:str) -> bytes:
    check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(self.compile_options), to_char_p_p([o.encode() for o in self.compile_options]))

    if status != 0: raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetPTX, cuda.nvrtcGetPTXSize, check)

def cuda_disassemble(lib, arch):
  try:
    fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
    with open(fn + ".ptx", "wb") as f: f.write(lib)
    subprocess.run(["ptxas", f"-arch={arch}", "-o", fn, fn+".ptx"], check=True)
    print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
  except Exception as e: print("failed to generate SASS", str(e))

class CUDAProgram:
  def __init__(self, device:CUDADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))
    if DEBUG >= 6: cuda_disassemble(lib, device.arch)

    if CUDACPU: self.prg = lib
    else:
      check(cuda.cuCtxSetCurrent(self.device.context))
      self.module = cuda.CUmodule()
      status = cuda.cuModuleLoadData(ctypes.byref(self.module), lib)
      if status != 0:
        del self.module
        cuda_disassemble(lib, device.arch)
        raise RuntimeError("module load failed")
      check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
      self.prg = prg #type: ignore

  def __del__(self):
    if hasattr(self, 'module'): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if CUDACPU: self.vargs = args+tuple(vals)
    else:
      check(cuda.cuCtxSetCurrent(self.device.context))
      if not hasattr(self, "vargs"):
        self.c_args, self.vargs = encode_args(args, vals) #type: ignore
      else:
        for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
        for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs)), enable=wait)

class CUDAAllocator(LRUAllocator):
  def __init__(self, device:CUDADevice):
    self.device = device
    super().__init__()
  def _alloc(self, size, options:BufferOptions):
    check(cuda.cuCtxSetCurrent(self.device.context))
    if options.host: return init_c_var(ctypes.c_void_p(), lambda x: check(cuda.cuMemHostAlloc(ctypes.byref(x), size, 0x01)))
    else: return init_c_var(cuda.CUdeviceptr(), lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque, options:BufferOptions):
    if options.host: return check(cuda.cuMemFreeHost(opaque))
    else: check(cuda.cuMemFree_v2(opaque))
  def copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.device.context))
    host_mem = self.alloc(len(src), BufferOptions(host=True))
    self.device.pending_copyin.append((host_mem, len(src), BufferOptions(host=True)))
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(cuda.cuMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))
  def copyout(self, dest:memoryview, src):
    CUDADevice.synchronize_system()
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyDtoH_v2(from_mv(dest), src, len(dest)))
  def transfer(self, dest, src, sz:int, src_dev, dest_dev):
    check(cuda.cuCtxSetCurrent(src_dev.context))
    check(cuda.cuEventCreate(ctypes.byref(sync_event := cuda.CUevent()), 0))
    check(cuda.cuMemcpyDtoDAsync_v2(dest, src, sz, None))
    check(cuda.cuEventRecord(sync_event, None))
    check(cuda.cuCtxSetCurrent(dest_dev.context))
    check(cuda.cuStreamWaitEvent(None, sync_event, 0)) # sync the default stream on the dest dev

class CUDADevice(Compiled):
  devices: List[CUDADevice] = []

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    if not CUDACPU:
      check(cuda.cuInit(0))
      check(cuda.cuDeviceGet(ctypes.byref(cu_device := cuda.CUdevice()), device_id))
      self.context = init_c_var(cuda.CUcontext(), lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, cu_device)))
      check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))

    self.arch = f"sm_{major.value}{minor.value}" if not CUDACPU else "sm_35"
    self.pending_copyin: List[Tuple[int, int, Optional[BufferOptions]]] = []
    CUDADevice.devices.append(self)

    from tinygrad.runtime.graph.cuda import CUDAGraph
    super().__init__(device, CUDAAllocator(self) if not CUDACPU else MallocAllocator,
                     PTXCompiler(self.arch) if getenv("PTX") else CUDACompiler(self.arch),
                     functools.partial(CUDAProgram, self), graph=CUDAGraph if not CUDACPU else None)

  def synchronize(self):
    if CUDACPU: return
    check(cuda.cuCtxSetCurrent(self.context))
    check(cuda.cuCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in CUDADevice.devices: d.synchronize()
