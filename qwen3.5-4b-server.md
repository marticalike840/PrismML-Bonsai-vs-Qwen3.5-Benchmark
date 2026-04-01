# Qwen3.5-4B — Jetson Orin Deployment

## Hardware

- **Device**: NVIDIA Jetson Orin (compute capability 8.7)
- **Unified Memory**: 30,696 MiB (shared CPU/GPU)
- **CPU Threads**: 12

## Model

| Property | Value |
|----------|-------|
| File | `Qwen3.5-4B-Q4_K_M.gguf` |
| Architecture | Qwen3.5 (hybrid Gated DeltaNet + attention), 32 layers |
| Parameters | 4.21 B |
| Quantization | Q4_K_M (5.19 BPW) |
| File size | 2.54 GiB |
| Training context | 262,144 tokens |
| GQA | 16 heads / 4 KV heads, head dim 256 |
| SSM | Gated DeltaNet (d_inner=4096, d_state=128, d_conv=4) |

Qwen3.5-4B uses a hybrid architecture with Gated Delta Networks (linear attention)
and standard grouped-query attention. The layer layout repeats 8 blocks of
`3×(Gated DeltaNet → FFN) → 1×(Gated Attention → FFN)`, so only 8 of 32 layers
use full attention KV cache. The rest use recurrent state with O(1) memory.

## Server Command

Managed via systemd unit `llama-server-qwen3.5-4b.service`:

```bash
/home/arman/ai/llama.cpp/llama-server \
    -m /media/arman/BlueSSD/AI/ggufs/Qwen3.5-4B-Q4_K_M.gguf \
    --alias Qwen3.5-4B \
    --host 0.0.0.0 \
    --port 8001 \
    -np 1 \
    -fa on \
    --no-mmap \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 32768 \
    --temp 0.7 \
    --top-k 20 \
    --top-p 0.8 \
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

Qwen3.5-4B has thinking disabled by default (smaller models ≤9B), but we set both
mechanisms explicitly for consistency:

1. **`--reasoning off`** — tells llama.cpp to disable reasoning extraction
2. **`LLAMA_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}`** — modifies the chat
   template to omit the `<think>` block

See [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5) for details.

### Parameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `-np 1` | 1 parallel slot | Single user; all memory goes to context depth |
| `-fa on` | Flash attention | More memory-efficient attention computation |
| `--no-mmap` | Disable memory mapping | Required on Jetson — mmap causes double memory occupation in unified memory |
| `-ctk q8_0 -ctv q8_0` | Quantized KV cache | Halves KV memory (544 MiB vs ~1,088 MiB at f16) |
| `--ctx-size 32768` | 32K context | Conservative; hybrid arch makes KV cheap — could go much higher |
| `--temp 0.7` | Temperature | Non-thinking mode recommendation from Unsloth |
| `--top-k 20` | Top-K sampling | Model metadata recommendation |
| `--top-p 0.8` | Top-P sampling | Non-thinking mode recommendation (tighter than thinking mode's 0.95) |
| `--min-p 0.0` | Min-P disabled | Filtering handled by top_k + top_p |
| `--presence-penalty 1.5` | Presence penalty | Reduces repetition |
| `--cache-reuse 256` | Prompt cache reuse | Not effective — hybrid arch forces full prompt re-processing |
| `--reasoning off` | Disable thinking | Prevents `<think>` block generation |

### Notes on Hybrid Architecture

- **cache-reuse is disabled at runtime** — the hybrid DeltaNet/attention context does
  not support partial KV reuse. Each request forces full prompt re-processing.
- **Only 8 of 32 layers have KV cache** — the rest use recurrent state (50.25 MiB).
  Same ratio as the 9B variant.

## Memory Breakdown

| Component | GPU (MiB) | CPU (MiB) |
|-----------|----------:|----------:|
| Model weights | 2,603.5 | 497.3 |
| KV cache (q8_0, 32K ctx, 8 layers) | 544.0 | — |
| Recurrent state (SSM, 32 layers) | 50.3 | — |
| Compute buffers | 490.0 | 74.0 |
| Output buffer | — | 1.0 |
| **Total used** | **~3,688** | **~572** |
| **Free (of 30,696)** | **~27,008** | — |

The lightest Qwen3.5 model. Uses only ~3.7 GiB of GPU memory, leaving ~27 GiB free.
Could easily run at 128K+ context or alongside other models.

## Benchmark Results (thinking = off)

**Date**: 2026-03-31
**llama.cpp build**: 8589 (08f21453a)

### Generation Performance

| Test | Prompt Tokens | Completion Tokens | Wall Time (s) | Gen (tok/s) | Prompt (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| Short Q&A | 19 | 67 | 2.43 | 29.35 | 168.45 |
| Code generation | 32 | 314 | 10.88 | 29.28 | 306.20 |
| Code review (821 tok input) | 821 | 2,048 | 71.65 | 28.92 | 1,048.33 |
| Algorithm design | 47 | 2,048 | 70.41 | 29.17 | 407.22 |

### Server-Side Timing Detail

```
Test 1 — Short Q&A:
  prompt eval:    112.80 ms /    19 tokens (  5.94 ms/tok,  168.45 tok/s)
  generation:    2283.10 ms /    67 tokens ( 34.08 ms/tok,   29.35 tok/s)

Test 2 — Code generation:
  prompt eval:    104.51 ms /    32 tokens (  3.27 ms/tok,  306.20 tok/s)
  generation:   10722.66 ms /   314 tokens ( 34.15 ms/tok,   29.28 tok/s)

Test 3 — Code review:
  prompt eval:    783.15 ms /   821 tokens (  0.95 ms/tok, 1048.33 tok/s)
  generation:   70812.00 ms /  2048 tokens ( 34.58 ms/tok,   28.92 tok/s)

Test 4 — Algorithm design:
  prompt eval:    115.42 ms /    47 tokens (  2.46 ms/tok,  407.22 tok/s)
  generation:   70208.76 ms /  2048 tokens ( 34.28 ms/tok,   29.17 tok/s)
```

### Comparison with All Models on This Device

| Metric | Qwen3.5-4B | Qwen3.5-9B | Qwen3.5-27B | Qwen3.5-27B Distilled | Bonsai-8B (1-bit) |
|--------|:-:|:-:|:-:|:-:|:-:|
| Gen tok/s (avg) | **~29** | ~21 | ~7.3 | ~7.2 | ~38 |
| Prompt tok/s (large) | 1,048 | 695 | 216 | 208 | 1,061 |
| Model weight memory | 3,101 MiB | 5,407 MiB | 15,955 MiB | 15,764 MiB | 1,099 MiB |
| Total GPU memory | ~3.7 GiB | ~6.0 GiB | ~17.0 GiB | ~20.1 GiB | ~6.2 GiB |
| ms per output token | ~34 ms | ~47 ms | ~137 ms | ~139 ms | ~27 ms |
| Parameters | 4.21 B | 8.95 B | 26.90 B | 26.90 B | 8.19 B |
| Effective bits/weight | 5.19 | 5.07 | 4.98 | 4.92 | 1.125 |

### Key Findings

- **Generation speed is ~29 tok/s**, the second fastest after Bonsai-8B (~38 tok/s).
  The 4B model has roughly half the weights of the 9B, directly translating to ~1.4x
  faster generation (29 vs 21 tok/s), consistent with memory-bandwidth-bound decoding.
- **~34 ms per output token**: Between Bonsai-8B (27 ms) and Qwen3.5-9B (47 ms).
  The model moves ~2.6 GiB of weights per token through Orin's ~205 GB/s bandwidth.
- **Prompt ingestion exceeds 1,000 tok/s** at 821 tokens, matching Bonsai-8B and
  far exceeding the 27B variants. A 10K-token codebase would be ingested in ~10 seconds.
- **Smallest memory footprint of the Qwen3.5 family**: Only 3.7 GiB total, leaving
  27 GiB free. Could run at 128K context (adding ~2 GiB KV) or alongside another model.
- **Same hybrid architecture tradeoffs as the 9B**: No prompt caching, identical KV
  cache size (544 MiB for 32K, 8 attention layers), same recurrent state overhead (50 MiB).
- **Best speed-to-memory ratio** among the Q4_K_M models. For latency-sensitive tasks
  where 4B quality is acceptable, this is the most efficient option on Orin.
