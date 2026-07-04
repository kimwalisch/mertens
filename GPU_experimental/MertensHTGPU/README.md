# MertensHTGPU (experimental)

A GPU-offload experiment for MertensHT: the FacToSumMu evaluation inside SArr
runs as Metal compute kernels. This is the code behind the Helfgott-Thompson
GPU results in Appendix D of the paper.

Experimental means experimental — it has not been scrutinized to the level of
the CPU implementations. It exists for exploration, and maybe a real
implementation down the road.

This directory keeps its own copy of the MertensHT sources (`src/`) so the
CPU tree stays GPU-free.

## Platform

Metal only, so macOS only: Apple Silicon, and the dual AMD GPUs of a 2019
Mac Pro (multi-GPU is supported). No CUDA.

## Building and running

```
brew install libomp boost
make            # CPU baseline (no Metal) — the validated reference
make GPU=1      # with the Metal engine

./build/mertens_ht_gpu 1e16 --profile
./build/mertens_ht_gpu --gpu-selftest    # GPU=1 builds: validate the Metal path
```

The CLI is otherwise the same as the CPU `mertens_ht` (see
[MertensHT/README.md](../../MertensHT/README.md)).
