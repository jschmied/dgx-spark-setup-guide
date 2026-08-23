# Environment — vLLM DFlash fused context-KV bug report

Collected with `vllm collect-env` from the venv the bug was reproduced in.

## Deviations from a stock install (please read before judging the report)

The venv is **not** a clean `pip install vllm`. It is a clone of a working
`v0.27.1` venv with the nightly wheel installed over it via `--no-deps`, so that
`torch 2.13.0+cu130` was not disturbed:

```
vllm-0.26.1rc1.dev1049+gf8e060271-cp38-abi3-manylinux_2_28_aarch64.whl
```

That wheel contains the merge of #52816 (`ahead=1, behind=0` against merge commit
`b389ac2946`).

Because of `--no-deps`, `pip check` reports these below the wheel's declared pins.
None of them are on the code path in question, but they are disclosed for
completeness:

| package | installed | wheel requires |
|---|---|---|
| flashinfer-python | 0.6.16.post3 | 0.6.17 |
| fastsafetensors | 0.3.2 | >= 0.3.3 |
| nvidia-cutlass-dsl | 4.6.0 | 4.6.2 |
| quack-kernels | 0.6.1 | 0.6.4 |
| huggingface_hub | 1.21.0 | >= 1.28.0 |
| instanttensor | not installed | required (only for `--load-format instanttensor`) |

`FLASHINFER_DISABLE_VERSION_CHECK=1` is set because of the first row.

**Two local patches were applied on top of the wheel:**

1. **Required to reach this bug at all** — the draft model's quantization config is
   not wired into `packed_modules_mapping`, so a quantized drafter fails to load
   first. That is a separate defect, reported as
   [#53107](https://github.com/vllm-project/vllm/issues/53107); the local patch calls
   `configure_quant_config()` for the draft architecture in
   `get_draft_quant_config()`. Without it you never get far enough to see the crash
   described in the issue.
2. Unrelated to this bug: the DFlash2 candidate top-k is taken in FP32 rather than in
   the head's own precision.

## `vllm collect-env`

```
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.4 LTS (aarch64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.13.0+cu130
Is debug build               : False
CUDA used to build PyTorch   : 13.0
ROCM used to build PyTorch   : N/A
XPU used to build PyTorch    : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.17.0-1029-nvidia-aarch64-with-glibc2.39
    
==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 13.0.88
CUDA_MODULE_LOADING set to   : 
GPU models and configuration : GPU 0: NVIDIA GB10
Nvidia driver version        : 580.173.02
cuDNN version                : Could not collect
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : False

==============================
          CPU Info
==============================
Architektur:                             aarch64
CPU Operationsmodus:                     64-bit
Byte-Reihenfolge:                        Little Endian
CPU(s):                                  20
Liste der Online-CPU(s):                 0-19
Anbieterkennung:                         ARM
Modellname:                              Cortex-X925
Modell:                                  1
Thread(s) pro Kern:                      1
Kern(e) pro Sockel:                      10
Sockel:                                  1
Stepping:                                r0p1
Übertaktung:                             deaktiviert
Skalierung der CPU(s):                   100%
Maximale Taktfrequenz der CPU:           3900,0000
Minimale Taktfrequenz der CPU:           1378,0000
BogoMIPS:                                2000,00
Markierungen:                            fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
Modellname:                              Cortex-A725
Modell:                                  1
Thread(s) pro Kern:                      1
Kern(e) pro Sockel:                      10
Sockel:                                  1
Stepping:                                r0p1
Skalierung der CPU(s):                   100%
Maximale Taktfrequenz der CPU:           2808,0000
Minimale Taktfrequenz der CPU:           338,0000
BogoMIPS:                                2000,00
Markierungen:                            fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
L1d Cache:                               1,3 MiB (20 Instanzen)
L1i Cache:                               1,3 MiB (20 Instanzen)
L2 Cache:                                25 MiB (20 Instanzen)
L3 Cache:                                24 MiB (2 Instanzen)
NUMA-Knoten:                             1
NUMA-Knoten0 CPU(s):                     0-19
Schwachstelle Gather data sampling:      Not affected
Schwachstelle Ghostwrite:                Not affected
Schwachstelle Indirect target selection: Not affected
Schwachstelle Itlb multihit:             Not affected
Schwachstelle L1tf:                      Not affected
Schwachstelle Mds:                       Not affected
Schwachstelle Meltdown:                  Not affected
Schwachstelle Mmio stale data:           Not affected
Schwachstelle Old microcode:             Not affected
Schwachstelle Reg file data sampling:    Not affected
Schwachstelle Retbleed:                  Not affected
Schwachstelle Spec rstack overflow:      Not affected
Schwachstelle Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Schwachstelle Spectre v1:                Mitigation; __user pointer sanitization
Schwachstelle Spectre v2:                Mitigation; CSV2, BHB
Schwachstelle Srbds:                     Not affected
Schwachstelle Tsa:                       Not affected
Schwachstelle Tsx async abort:           Not affected
Schwachstelle Vmscape:                   Not affected

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.16.post3
[pip3] nccl4py==0.4.1
[pip3] numpy==2.3.5
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cccl==13.3.3.4.1
[pip3] nvidia-cuda-crt==13.3.73
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvcc==13.2.78
[pip3] nvidia-cuda-nvdisasm==13.3.73
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cuda-tileiras==13.2.78
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cudnn-frontend==1.25.0
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cufile==1.15.1.6
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-cutlass-dsl==4.6.0
[pip3] nvidia-cutlass-dsl-libs-base==4.6.0
[pip3] nvidia-cutlass-dsl-libs-core==4.6.0
[pip3] nvidia-cutlass-dsl-libs-cu12==4.6.0
[pip3] nvidia-cutlass-dsl-libs-cu13==4.6.0
[pip3] nvidia-ml-py==13.610.43
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.0.88
[pip3] nvidia-nvshmem-cu13==3.4.5
[pip3] nvidia-nvtx==13.0.85
[pip3] nvidia-nvvm==13.2.78
[pip3] pyzmq==27.1.0
[pip3] tokenspeed-triton==3.7.10.post20260531
[pip3] torch==2.13.0+cu130
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torchaudio==2.11.0+cu130
[pip3] torchcodec==0.14.0
[pip3] torchvision==0.28.0+cu130
[pip3] transformers==5.12.1
[pip3] triton==3.7.1
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.26.1rc1.dev1049+gf8e060271 (git sha: f8e060271)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled
GPU Topology:
  	[4mGPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	0-19	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

==============================
     Environment Variables
==============================
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_jschmied
LD_LIBRARY_PATH=/home/jschmied/vllm-venv-main-dflash2/lib/python3.12/site-packages/cv2/../../lib64:
VLLM_WORKER_MULTIPROC_METHOD=spawn

```
