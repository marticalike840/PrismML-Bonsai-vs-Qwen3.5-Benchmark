# Bonsai-8B (1-bit) — Jetson Orin Deployment

## Hardware

- **Device**: NVIDIA Jetson Orin (compute capability 8.7)
- **Unified Memory**: 30,696 MiB (shared CPU/GPU)
- **CPU Threads**: 12

## Model

| Property | Value |
|----------|-------|
| File | `Bonsai-8B.gguf` |
| Architecture | Qwen3-8B dense transformer (GQA, SwiGLU, RoPE, RMSNorm) |
| Parameters | 8.19 B (~6.95 B non-embedding) |
| Quantization | Q1_0_g128 (1.125 BPW — 1-bit with group scale) |
| File size | 1.1 GiB |
| Training context | 65,536 tokens |
| Layers | 36 transformer decoder blocks |
| GQA | 32 query heads / 8 KV heads |
| License | Apache 2.0 |
| Source | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) |

Bonsai-8B is an end-to-end 1-bit quantized version of Qwen3-8B. All major weight
tensors (embeddings, attention projections, MLP projections, LM head) are stored in
Q1_0_g128 format: 1 sign bit per weight with a 16-bit scale shared across groups
of 128. Dequantization happens inline on the GPU — no FP16 materialization.

### llama.cpp Compatibility

The Q1_0_g128 format is **not supported by upstream llama.cpp**. This deployment uses
the [PrismML fork](https://github.com/PrismML-Eng/llama.cpp) which includes the
custom dequantization kernels. The fork is checked out at `/opt/llama.cpp-prismml/`.

## Server Command

Managed via systemd unit `llama-server-bonsai-8b.service`:

```bash
/opt/llama.cpp-prismml/build/bin/llama-server \
    -m /models/Bonsai-8B.gguf \
    --alias Bonsai-8B \
    --host 0.0.0.0 \
    --port 8001 \
    -np 1 \
    -fa on \
    --no-mmap \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 65536 \
    --temp 0.5 \
    --top-k 20 \
    --top-p 0.9 \
    --min-p 0.0 \
    --cache-reuse 256
```

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-np 1` | 1 parallel slot | Single user; all memory goes to context depth |
| `-fa on` | Flash attention | More memory-efficient attention computation |
| `--no-mmap` | Disable memory mapping | Required on Jetson — mmap causes double memory occupation in unified memory |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV memory (4,896 MiB vs 9,216 MiB at f16) |
| `--ctx-size 65536` | 64K context | Model's full training context; fits easily given the tiny weight footprint |
| `--temp 0.5` | Temperature | Model metadata recommendation (range 0.5–0.7) |
| `--top-k 20` | Top-K sampling | Model metadata recommendation (range 20–40) |
| `--top-p 0.9` | Top-P sampling | Model metadata recommendation (range 0.85–0.95) |
| `--min-p 0.0` | Min-P disabled | Filtering handled by top_k + top_p |
| `--cache-reuse 256` | Prompt cache reuse | Reuses cached KV from prior requests sharing a common prefix |

## Memory Breakdown

| Component | GPU (MiB) | CPU (MiB) |
|-----------|----------:|----------:|
| Model weights | 1,016.0 | 83.3 |
| KV cache (q8_0, 64K ctx) | 4,896.0 | — |
| Compute buffers | 304.2 | 144.0 |
| Output buffer | — | 0.6 |
| **Total used** | **~6,216** | **~228** |
| **Free (of 30,696)** | **~24,480** | — |

The 1-bit weights leave the vast majority of memory for KV cache. This model could
theoretically run alongside a second model on the same Jetson.

## Benchmark Results

**Date**: 2026-03-31
**llama.cpp**: PrismML fork (built from source, CUDA sm_87)

### Generation Performance

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 19 | 8 | 0.27 | 41.42 | 285.13 |
| Code generation | 32 | 408 | 10.50 | 39.17 | 444.42 |
| Code review (759 tok input) | 759 | 2,048 | 58.09 | 35.71 | 1,060.61 |
| Algorithm design | 47 | 2,048 | 55.32 | 37.12 | 601.14 |

### Server-Side Timing Detail

```
Test 1 — Short Q&A:
  prompt eval:     66.64 ms /    19 tokens (  3.51 ms/tok,  285.13 tok/s)
  generation:     193.13 ms /     8 tokens ( 24.14 ms/tok,   41.42 tok/s)

Test 2 — Code generation:
  prompt eval:     65.25 ms /    32 tokens (  2.04 ms/tok,  444.42 tok/s)
  generation:   10415.37 ms /   408 tokens ( 25.53 ms/tok,   39.17 tok/s)

Test 3 — Code review:
  prompt eval:    712.79 ms /   759 tokens (  0.94 ms/tok, 1060.61 tok/s)
  generation:  57348.25 ms /  2048 tokens ( 27.99 ms/tok,   35.71 tok/s)

Test 4 — Algorithm design:
  prompt eval:     73.19 ms /    47 tokens (  1.56 ms/tok,  601.14 tok/s)
  generation:  55172.66 ms /  2048 tokens ( 26.94 ms/tok,   37.12 tok/s)
```

### Comparison with Qwen3.5-27B Distilled (Q4_K_M)

| Metric | Bonsai-8B (1-bit) | Qwen3.5-27B (Q4_K_M) | Ratio |
|--------|:-:|:-:|:-:|
| Gen tok/s (avg) | ~38 | ~7.2 | **5.3x faster** |
| Prompt tok/s (759 tok) | 1,061 | 208 | **5.1x faster** |
| Model weight memory | 1,016 MiB | 15,082 MiB | **14.8x smaller** |
| Total GPU memory | ~6.2 GiB | ~20.1 GiB | **3.2x less** |
| ms per output token | ~26 ms | ~139 ms | **5.3x lower latency** |
| Parameters | 8.19 B | 26.90 B | 0.3x |
| Effective bits/weight | 1.125 | 4.92 | 0.23x |

**Speed advantage**: The 1-bit weights are ~5x cheaper to move through Orin's memory
bandwidth (~205 GB/s), which is the bottleneck for autoregressive decoding. The smaller
model size also means less compute per token.

**Quality tradeoff**: Bonsai-8B is a 1-bit 8B model vs a 4-bit 27B reasoning model.
The Qwen distill will produce substantially higher quality output, especially for
complex reasoning, code generation, and multi-step tasks. Bonsai-8B is better suited
for latency-sensitive applications where speed matters more than peak quality.

### Key Findings

- **Generation speed is stable at ~37 tok/s**, roughly 5x the Qwen 27B distill.
  The per-token latency (~27 ms) is dominated by memory bandwidth for the KV cache
  reads, not the weight loads (which are trivially small at 1-bit).
- **Prompt ingestion exceeds 1,000 tok/s** at batch sizes around 750 tokens.
  A 10K-token codebase would be ingested in under 10 seconds.
- **~24.5 GiB free memory** after loading — enough to run a second model concurrently
  or massively extend context with larger KV cache quantization.
- **Short Q&A produced only 8 tokens** (just "The capital of France is Paris."),
  suggesting the model is concise but may lack the elaboration depth of larger models.
- **Tests 3 and 4 hit the 2,048 token limit**, indicating the model is sufficiently
  generative for longer outputs.

## Troubleshooting Notes

### `ggml type 41` / `invalid ggml type`

Upstream llama.cpp does not support Q1_0 (type ID 41). You must use the PrismML fork.
If you see this error, the service is pointing at the wrong llama-server binary.

### Rebuilding the PrismML Fork

```bash
cd ~/ai/llama.cpp-prismml
git pull
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc) --target llama-server
```

`nvcc` is not on the default PATH on this Jetson; the export line is required.
