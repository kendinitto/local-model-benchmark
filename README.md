# Local LLM Inference Benchmark Suite

Production-grade benchmarking tool for measuring local LLM inference performance across models, quantizations, and hardware configurations. Built around custom llama.cpp builds with optimized GPU/CPU backends.

## Features

- **Dual benchmark modes** — CLI mode (direct llama.cpp binary) or API mode (connect to running server via HTTP)
- **Multi-model benchmarking** — Compare Qwen, LLaMA, Mistral, and other GGUF models side by side
- **Quantization analysis** — Benchmark Q4_K_M, Q5_K_M, Q8_0, and other quant levels to find the speed/quality sweet spot
- **Hardware optimization profiles** — Presets for speed-max, balanced, memory-conserve, and max-context configurations
- **Categorized prompts** — Code generation, logical reasoning, debugging, and system design prompts for realistic workloads
- **Reproducible results** — Fixed seeds, structured JSON output, and comparison tables

## Quick Start

### 1. Build llama.cpp

```bash
chmod +x scripts/build_llama.sh
./scripts/build_llama.sh cuda   # or metal, vulkan, cpu
```

### 2. Configure Models

Edit `configs/models.json` with your model paths:

```json
{
  "name": "Qwen 3.6 27B",
  "path": "/path/to/qwen3-6-27b-q4_k_m.gguf",
  "quantization": "Q4_K_M",
  "gpu_layers": 35,
  "thread_count": 8,
  "batch_size": 4096,
  "context_size": 8192,
  "flash_attention": true
}
```

### 3. Run Benchmarks

**CLI Mode** (requires llama.cpp binary):

```bash
# Full suite with config file
python benchmarks/benchmark.py -c configs/models.json

# Quick single-model benchmark
python scripts/quick_bench.py /path/to/model.gguf --name "Qwen 3.6 27B" --quant Q4_K_M

# Specific category only
python benchmarks/benchmark.py -c configs/models.json --category coding
```

**API Mode** (connects to running llama.cpp server — no binary needed):

```bash
# Quick API benchmark (defaults to localhost:8080)
python scripts/quick_bench.py --api

# Connect to remote server
python scripts/quick_bench.py --api --server http://192.168.1.100:8080

# Full API benchmark suite
python benchmarks/api_benchmark.py --server http://localhost:8080 --predict 1024

# Force OpenAI-compatible API mode
python benchmarks/api_benchmark.py --server http://localhost:8080 --api-mode openai
```

API mode works with any llama.cpp server regardless of platform (Windows, Linux, macOS) — just point it at the IP and port.

## Optimization Presets

| Preset | Use Case | GPU | Threads | Batch | Context | Flash Attn |
|--------|----------|-----|---------|-------|---------|------------|
| `speed_max` | Batch processing, CI/CD | All | 16 | 8192 | 4096 | Yes |
| `balanced` | General purpose | All | 8 | 4096 | 8192 | Yes |
| `memory_conserve` | Constrained systems | 20 | 4 | 1024 | 4096 | No |
| `context_max` | Long document processing | All | 8 | 4096 | 32768 | Yes |

## Key Optimizations

### GPU Offloading
- Set `gpu_layers: -1` to offload all layers to GPU
- For models larger than VRAM, partial offloading (e.g., `gpu_layers: 30`) keeps the hottest layers on GPU while spilling the rest to system RAM via mmap

### Flash Attention
- Enables paged attention for reduced memory usage and faster context processing
- Enabled with `-fa` flag or `flash_attention: true` in config
- Most impactful for large context windows (>8K tokens)

### Memory Management
- `mmap: true` — Use memory-mapped files for lazy loading (default, recommended)
- `mlock: true` — Lock model in RAM to prevent swapping (use for max-context profiles)
- Larger `batch_size` improves throughput but increases VRAM consumption

### Quantization Tradeoffs
- `Q4_K_M` — Best balance of size/speed/quality (recommended starting point)
- `Q5_K_M` — Slightly better quality, ~20% larger, minimal speed impact on GPU
- `Q8_0` — Near-float16 quality, ~2x Q4 size, use when quality matters most

## Benchmark Prompts

| Category | Prompt | Expected Tokens |
|----------|--------|-----------------|
| `coding` | Binary search tree implementation with tests | 200+ |
| `reasoning` | Multi-step train word problem | 150+ |
| `coding` | Debug buggy rate limiter (token bucket) | 300+ |
| `reasoning` | Multi-agent code review system design | 250+ |

## Output

Results are saved as JSON to `results/` with:
- Per-run metrics: tokens/sec, time-to-first-token, total time
- Summary statistics: avg/max/min tokens/sec per model
- Comparison table printed to stdout

Example output (Qwen 3.6 27B Q4_K_M via API):

```
======================================================================
RESULTS — unsloth/Qwen3.6-27B (UD-Q4_KM) via http://localhost:11434
======================================================================
Prompt                    tok/s      TTF ms     Tokens     Category
----------------------------------------------------------------------
code_generation           41.05      1701.26    1024       coding
logical_reasoning         43.86      164.73     1024       reasoning
code_debugging            42.42      222.07     188        coding
system_design             43.76      145.61     1024       reasoning
----------------------------------------------------------------------
AVERAGE                   42.77      558.42     3260
======================================================================
```

CLI mode comparison across models:

```
================================================================================
BENCHMARK COMPARISON
================================================================================
Model                               Avg tok/s    Max tok/s    Avg TTF ms   Runs
--------------------------------------------------------------------------------
Meta Llama 3.1 8B (Q4_K_M)          89.20        95.60        142.80       4
Mistral 7B Instruct (Q4_K_M)        92.10        98.40        138.50       4
Qwen 3.6 27B (Q4_K_M)               51.80        58.30        285.10       4
Qwen 3.6 27B (Q5_K_M)               42.50        48.20        312.40       4
================================================================================
```

## Hardware Requirements

| Model | Quant | Min VRAM | Min RAM | Recommended |
|-------|-------|----------|---------|-------------|
| Qwen 3.6 27B | Q4_K_M | 16 GB | 32 GB | RTX 3090/4090 |
| Qwen 3.6 27B | Q5_K_M | 20 GB | 40 GB | RTX 4090 |
| Llama 3.1 8B | Q4_K_M | 6 GB | 16 GB | RTX 3060+ |
| Mistral 7B | Q4_K_M | 5 GB | 12 GB | RTX 3060+ |

## Custom llama.cpp Builds

This benchmark suite is tested with custom llama.cpp builds featuring:
- Optimized CUDA kernels for RTX 40-series GPUs
- Flash attention v2 support
- Improved memory management for large context windows
- Custom quantization routines for Q4_K_M optimization

## License

MIT
