# Qwen3.5-27B — Jetson Orin Deployment

## Hardware

- **Device**: NVIDIA Jetson Orin (compute capability 8.7)
- **Unified Memory**: 30,696 MiB (shared CPU/GPU)
- **CPU Threads**: 12

## Model

| Property | Value |
|----------|-------|
| File | `Qwen3.5-27B-Q4_K_M.gguf` |
| Architecture | Qwen3.5 (hybrid SSM + SWA + Full Attention), 64 layers |
| Parameters | 26.90 B |
| Quantization | Q4_K_M (4.98 BPW) |
| File size | 15.58 GiB |
| Training context | 262,144 tokens |
| GQA | 24 heads / 4 KV heads, head dim 256 |
| SSM | Mamba-style (d_inner=6144, d_state=128, d_conv=4) |
| Layers | 64 total (16 full attention with KV cache, 48 SSM/SWA) |

Same architecture as the distilled variant but this is the base Qwen3.5-27B model
(not the Claude Opus reasoning distillation). The hybrid design means only 16 of 64
layers use full attention KV cache; the remaining 48 use SSM recurrent state or
sliding window attention.

## Server Command

Managed via systemd unit `llama-server-qwen3.5-27b.service`:

```bash
/home/arman/ai/llama.cpp/llama-server \
    -m /media/arman/BlueSSD/AI/ggufs/Qwen3.5-27B-Q4_K_M.gguf \
    --alias Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 8001 \
    -np 1 \
    -fa on \
    --no-mmap \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 32768 \
    --temp 0.6 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --cache-reuse 256 \
    --reasoning off
```

The service also sets the environment variable via `EnvironmentFile`:

```
LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
```

### Disabling Thinking

Qwen3.5-27B has thinking enabled by default (unlike the smaller 9B variant). Two
mechanisms are used to disable it:

1. **`--reasoning off`** — tells llama.cpp to disable reasoning extraction
2. **`LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}`** — modifies the chat
   template to omit the `<think>` block from the prompt format

See [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5) for details.

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-np 1` | 1 parallel slot | Single user; all memory goes to context depth |
| `-fa on` | Flash attention | More memory-efficient attention computation |
| `--no-mmap` | Disable memory mapping | Required on Jetson — mmap causes double memory occupation in unified memory |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV memory (1,088 MiB vs ~2,176 MiB at f16) |
| `--ctx-size 32768` | 32K context | Conservative; could go higher — hybrid arch keeps KV cheap (only 16 layers) |
| `--temp 0.6` | Temperature | Focused output for code tasks |
| `--top-k 20` | Top-K sampling | Standard filtering |
| `--top-p 0.95` | Top-P sampling | Standard nucleus sampling |
| `--min-p 0.0` | Min-P disabled | Filtering handled by top_k + top_p |
| `--cache-reuse 256` | Prompt cache reuse | Not effective — hybrid arch forces full prompt re-processing |
| `--reasoning off` | Disable thinking | Prevents `<think>` block generation and extraction |

### Notes on Hybrid Architecture

- **cache-reuse is disabled at runtime** — the hybrid SSM/attention context does not support
  partial KV reuse. Each request forces full prompt re-processing.
- **Only 16 of 64 layers have KV cache** — the rest use recurrent state (149.6 MiB). This
  keeps KV cache small (1,088 MiB for 32K context) relative to the model's size.

## Memory Breakdown

| Component | GPU (MiB) | CPU (MiB) |
|-----------|----------:|----------:|
| Model weights | 15,272.8 | 682.0 |
| KV cache (q8_0, 32K ctx, 16 layers) | 1,088.0 | — |
| Recurrent state (SSM, 64 layers) | 149.6 | — |
| Compute buffers | 495.0 | 84.0 |
| Output buffer | — | 1.0 |
| **Total used** | **~17,005** | **~767** |
| **Free (of 30,696)** | **~13,691** | — |

Heavier than the 9B variant but still leaves ~13.7 GiB free. The hybrid architecture
saves significant KV memory — a pure transformer 27B at 32K would need ~4x the KV cache.

## Benchmark Results (thinking = off)

**Date**: 2026-03-31
**llama.cpp build**: 8589 (08f21453a)

### Generation Performance

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 19 | 75 | 10.63 | 7.39 | 43.05 |
| Code generation | 32 | 303 | 42.03 | 7.31 | 66.56 |
| Code review (821 tok input) | 821 | 2,048 | 288.10 | 7.21 | 215.70 |
| Algorithm design | 47 | 1,978 | 273.63 | 7.25 | 83.37 |

### Server-Side Timing Detail

```
Test 1 — Short Q&A:
  prompt eval:    441.36 ms /    19 tokens ( 23.23 ms/tok,   43.05 tok/s)
  generation:   10147.42 ms /    75 tokens (135.30 ms/tok,    7.39 tok/s)

Test 2 — Code generation:
  prompt eval:    480.75 ms /    32 tokens ( 15.02 ms/tok,   66.56 tok/s)
  generation:   41439.58 ms /   303 tokens (136.76 ms/tok,    7.31 tok/s)

Test 3 — Code review:
  prompt eval:   3806.30 ms /   821 tokens (  4.64 ms/tok,  215.70 tok/s)
  generation:  284173.00 ms /  2048 tokens (138.76 ms/tok,    7.21 tok/s)

Test 4 — Algorithm design:
  prompt eval:    563.78 ms /    47 tokens ( 11.99 ms/tok,   83.37 tok/s)
  generation:  272764.15 ms /  1978 tokens (137.90 ms/tok,    7.25 tok/s)
```

### Comparison with Other Models on This Device

| Metric | Qwen3.5-27B (Q4_K_M) | Qwen3.5-9B (Q4_K_M) | Bonsai-8B (1-bit) | Qwen3.5-27B Distilled (Q4_K_M) |
|--------|:-:|:-:|:-:|:-:|
| Gen tok/s (avg) | **~7.3** | ~21 | ~38 | ~7.2 |
| Prompt tok/s (large) | 216 | 695 | 1,061 | 208 |
| Model weight memory | 15,955 MiB | 5,407 MiB | 1,099 MiB | 15,764 MiB |
| Total GPU memory | ~17.0 GiB | ~6.0 GiB | ~6.2 GiB | ~20.1 GiB |
| ms per output token | ~137 ms | ~47 ms | ~27 ms | ~139 ms |
| Parameters | 26.90 B | 8.95 B | 8.19 B | 26.90 B |
| Effective bits/weight | 4.98 | 5.07 | 1.125 | 4.92 |

### Key Findings

- **Generation speed is ~7.3 tok/s**, essentially identical to the distilled variant
  (~7.2 tok/s). Both are 27B Q4_K_M models — the memory bandwidth bottleneck is the same.
- **~137 ms per output token**: Consistent with the Orin's ~205 GB/s memory bandwidth
  moving ~27 GiB of effective weight data per token.
- **Prompt ingestion reaches ~216 tok/s** at 821 tokens, comparable to the distilled
  variant (208 tok/s at 547 tokens).
- **Less memory than the distilled variant** (~17 GiB vs ~20 GiB) because this deployment
  uses 32K context instead of 128K. The KV cache is 1,088 MiB vs 4,352 MiB.
- **No prompt caching**: Same as the 9B — hybrid architecture forces full re-processing.
- **Thinking off reduces output waste**: Without `<think>` blocks, the model's full
  output budget goes to the actual answer. Tests 1 and 2 completed naturally under
  the token limit, while tests 3 and 4 used most of their budget on substantive content.
- **Base vs distilled**: Same throughput, but the distilled variant was fine-tuned with
  Claude Opus reasoning traces. For pure speed, they're interchangeable; for quality on
  reasoning tasks, the distilled variant should be preferred.
