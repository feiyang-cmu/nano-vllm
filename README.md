<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

### Option 1: Install from GitHub

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

### Option 2: Local Development Setup (Linux + CUDA)

If you want a reproducible local environment (recommended):

```bash
cd /YOUR/PATH/nano-vllm

# 1) Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 2) Core dependencies
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
python -m pip install triton==3.0.0 transformers xxhash

# 3) flash-attn (source build; avoids binary compatibility issues on older Linux)
python -m pip install -U ninja
export FLASH_ATTN_FORCE_BUILD=TRUE
python -m pip install --no-build-isolation --no-cache-dir --no-binary flash-attn --no-deps flash-attn==2.5.9.post1

# 4) Install nano-vllm itself
python -m pip install -e .
```

## Model Download

Use `hf` (from `huggingface_hub`) to download model weights:

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B \
  --max-workers 4
```

If needed, login first:

```bash
hf auth login
```

## Environment Checklist

```bash
python -c "import torch, flash_attn, nanovllm; print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
hf --help
```

## Common Pitfalls

- Python version too old: this project requires Python >=3.10 (3.11 recommended).
- Broken venv selection in IDE: ensure VS Code is using the project `.venv` interpreter.
- Wrong package name: `huggingface-cli` is not a PyPI package; install `huggingface_hub` and use `hf`.
- CUDA env var mistakes can break build tools:
  - Avoid `CUDA_HOME=$CUDA_HOME:/path` style concatenation.
  - Avoid Windows-style `%PATH` on Linux shells.
- `flash-attn` prebuilt wheels may fail on older Linux `glibc`; source build is often more reliable.

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)