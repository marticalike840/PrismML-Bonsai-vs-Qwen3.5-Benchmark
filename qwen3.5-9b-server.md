# Qwen3.5-9B — Jetson Orin Deployment

## Hardware

- **Device**: NVIDIA Jetson Orin (compute capability 8.7)
- **Unified Memory**: 30,696 MiB (shared CPU/GPU)
- **CPU Threads**: 12

## Model

| Property | Value |
|----------|-------|
| File | `Qwen3.5-9B-Q4_K_M.gguf` |
| Architecture | Qwen3.5 (hybrid SSM + attention), 32 layers |
| Parameters | 8.95 B |
| Quantization | Q4_K_M (5.07 BPW) |
| File size | 5.28 GiB |
| Training context | 262,144 tokens |
| GQA | 16 heads / 4 KV heads, head dim 256 |
| SSM | Mamba-style (d_inner=4096, d_state=128, d_conv=4) |

Qwen3.5-9B is a hybrid SSM + attention model. Only 8 of 32 layers use full attention
(requiring KV cache); the remaining 24 layers use SSM with O(1) recurrent state. This
makes the KV cache extremely small relative to model size.

## Server Command

Managed via systemd unit `llama-server-qwen3.5-9b.service`:

```bash
/opt/llama.cpp/llama-server \
    -m /models/Qwen3.5-9B-Q4_K_M.gguf \
    --alias Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8001 \
    -np 1 \
    -fa on \
    --no-mmap \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 32768 \
    --temp 1.0 \
    --top-k 20 \
    --top-p 0.95 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --cache-reuse 256 \
    --reasoning off
```

The service also sets the environment variable via `EnvironmentFile`:

```
LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}
```

### Disabling Thinking

Qwen3.5 models support a thinking/reasoning mode controlled at two levels:

1. **`--reasoning off`** — tells llama.cpp to disable reasoning extraction
2. **`LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}`** — modifies the chat
   template to omit the `<think>` block from the prompt format

Both are needed. The `--reasoning off` flag alone doesn't prevent the model from
generating `<think>` blocks if the template still enables them. See
[Unsloth docs](https://unsloth.ai/docs/models/qwen3.5) for details.

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-np 1` | 1 parallel slot | Single user; all memory goes to context depth |
| `-fa on` | Flash attention | More memory-efficient attention computation |
| `--no-mmap` | Disable memory mapping | Required on Jetson — mmap causes double memory occupation in unified memory |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV memory (544 MiB vs ~1,088 MiB at f16) |
| `--ctx-size 32768` | 32K context | Conservative choice; model trains to 262K but hybrid arch makes KV cheap — could go higher |
| `--temp 1.0` | Temperature | Higher temp for more varied output |
| `--top-k 20` | Top-K sampling | Standard filtering |
| `--top-p 0.95` | Top-P sampling | Standard nucleus sampling |
| `--min-p 0.0` | Min-P disabled | Filtering handled by top_k + top_p |
| `--presence-penalty 1.5` | Presence penalty | Reduces repetition |
| `--cache-reuse 256` | Prompt cache reuse | Note: logs show this is not supported by the hybrid/recurrent context |
| `--reasoning off` | Disable thinking | Prevents `<think>` block generation and extraction |

### Notes on Hybrid Architecture

- **cache-reuse is disabled at runtime** — the hybrid SSM/attention context does not support
  partial KV reuse. Each request forces full prompt re-processing.
- **Only 8 of 32 layers have KV cache** — the rest use recurrent state (50.25 MiB). This is
  why the KV cache is tiny (544 MiB for 32K context) compared to a pure transformer.

## Memory Breakdown

| Component | GPU (MiB) | CPU (MiB) |
|-----------|----------:|----------:|
| Model weights | 4,861.3 | 545.6 |
| KV cache (q8_0, 32K ctx, 8 layers) | 544.0 | — |
| Recurrent state (SSM, 32 layers) | 50.3 | — |
| Compute buffers | 501.0 | 80.0 |
| Output buffer | — | 1.0 |
| **Total used** | **~5,957** | **~627** |
| **Free (of 30,696)** | **~24,739** | — |

Extremely lightweight footprint. The hybrid architecture keeps KV cache minimal and the
recurrent state adds negligible overhead (50 MiB).

## Benchmark Results

**Date**: 2026-03-31
**llama.cpp build**: 8589 (08f21453a)

### Generation Performance

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 17 | 131 | 6.38 | 21.18 | 104.78 |
| Code generation | 30 | 536 | 25.68 | 21.04 | 193.79 |
| Code review (819 tok input) | 819 | 2,048 | 99.12 | 20.92 | 695.06 |
| Algorithm design | 45 | 1,570 | 74.82 | 21.05 | 263.70 |

### Server-Side Timing Detail

```
Test 1 — Short Q&A:
  prompt eval:    162.24 ms /    17 tokens (  9.54 ms/tok,  104.78 tok/s)
  generation:    6185.25 ms /   131 tokens ( 47.22 ms/tok,   21.18 tok/s)

Test 2 — Code generation:
  prompt eval:    154.81 ms /    30 tokens (  5.16 ms/tok,  193.79 tok/s)
  generation:   25470.42 ms /   536 tokens ( 47.52 ms/tok,   21.04 tok/s)

Test 3 — Code review:
  prompt eval:   1178.31 ms /   819 tokens (  1.44 ms/tok,  695.06 tok/s)
  generation:   97882.90 ms /  2048 tokens ( 47.79 ms/tok,   20.92 tok/s)

Test 4 — Algorithm design:
  prompt eval:    170.65 ms /    45 tokens (  3.79 ms/tok,  263.70 tok/s)
  generation:   74568.63 ms /  1570 tokens ( 47.50 ms/tok,   21.05 tok/s)
```

### Generation Performance (thinking = off)

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 19 | 76 | 3.77 | 21.20 | 125.68 |
| Code generation | 32 | 349 | 16.73 | 21.12 | 205.92 |
| Code review (821 tok input) | 821 | 2,048 | 99.04 | 20.94 | 697.24 |
| Algorithm design | 47 | 2,048 | 97.56 | 21.05 | 275.01 |

### Server-Side Timing Detail (thinking = off)

```
Test 1 — Short Q&A:
  prompt eval:    151.18 ms /    19 tokens (  7.96 ms/tok,  125.68 tok/s)
  generation:    3585.34 ms /    76 tokens ( 47.18 ms/tok,   21.20 tok/s)

Test 2 — Code generation:
  prompt eval:    155.40 ms /    32 tokens (  4.86 ms/tok,  205.92 tok/s)
  generation:   16524.53 ms /   349 tokens ( 47.35 ms/tok,   21.12 tok/s)

Test 3 — Code review:
  prompt eval:   1177.51 ms /   821 tokens (  1.43 ms/tok,  697.24 tok/s)
  generation:   97807.26 ms /  2048 tokens ( 47.76 ms/tok,   20.94 tok/s)

Test 4 — Algorithm design:
  prompt eval:    170.90 ms /    47 tokens (  3.64 ms/tok,  275.01 tok/s)
  generation:   97298.08 ms /  2048 tokens ( 47.51 ms/tok,   21.05 tok/s)
```

### Thinking On vs Off

| Test | Thinking On (tokens) | Thinking Off (tokens) | Wall Time On | Wall Time Off |
|------|:---:|:---:|:---:|:---:|
| Short Q&A | 131 | 76 | 6.38s | 3.77s |
| Code generation | 536 | 349 | 25.68s | 16.73s |
| Code review | 2,048 | 2,048 | 99.12s | 99.04s |
| Algorithm design | 1,570 | 2,048 | 74.82s | 97.56s |

Generation speed (~21 tok/s) is identical in both modes — the difference in wall time
comes entirely from fewer tokens generated when thinking is off (no `<think>` block
overhead). For tasks that hit the max token limit, there is no wall time difference, but
the model dedicates all output tokens to the actual answer instead of splitting them
between thinking and response.

### Comparison with Other Models on This Device

| Metric | Qwen3.5-9B (Q4_K_M) | Bonsai-8B (1-bit) | Qwen3.5-27B (Q4_K_M) |
|--------|:-:|:-:|:-:|
| Gen tok/s (avg) | **~21** | ~38 | ~7.2 |
| Prompt tok/s (large) | 695 | 1,061 | 208 |
| Model weight memory | 5,407 MiB | 1,099 MiB | 15,764 MiB |
| Total GPU memory | ~6.0 GiB | ~6.2 GiB | ~20.1 GiB |
| ms per output token | ~47 ms | ~27 ms | ~139 ms |
| Parameters | 8.95 B | 8.19 B | 26.90 B |
| Effective bits/weight | 5.07 | 1.125 | 4.92 |

### Key Findings

- **Generation speed is stable at ~21 tok/s** regardless of prompt size or task type.
  This is 3x the Qwen 27B distill and about half the Bonsai 1-bit model's speed.
- **~47 ms per output token**: Faster than the 27B model (~139 ms) due to fewer parameters
  to move through memory bandwidth, but slower than Bonsai-8B (~27 ms) despite similar
  parameter count — the 5 BPW Q4_K_M weights are ~4.5x larger than 1-bit weights.
- **Prompt ingestion reaches ~695 tok/s** at 819 tokens. A 10K-token codebase would be
  ingested in ~14 seconds.
- **Hybrid architecture pays off in memory**: Only 544 MiB KV cache for 32K context
  (vs 4,896 MiB for Bonsai-8B at 64K, or 4,352 MiB for Qwen 27B at 128K). The model
  could easily run at 128K+ context since only 8 layers need KV cache.
- **No prompt caching**: The hybrid SSM/attention architecture forces full prompt
  re-evaluation on every request. Multi-turn conversations pay the full prompt cost
  each time.
- **~24.7 GiB free memory** — similar to Bonsai-8B, enough to run a second model
  concurrently or extend context dramatically.
