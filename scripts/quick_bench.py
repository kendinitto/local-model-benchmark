#!/usr/bin/env python3
"""
Quick benchmark runner — single model, all prompts.
Convenience script for rapid iteration.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.benchmark import LLMBenchmark, ModelConfig, DEFAULT_PROMPTS


def main():
    parser = argparse.ArgumentParser(description="Quick LLM Benchmark")
    parser.add_argument("model_path", help="Path to model .gguf file")
    parser.add_argument("--name", default="custom", help="Model display name")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization label")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--context", type=int, default=8192, help="Context size")
    parser.add_argument("--llama-path", default="./llama.cpp/build/llama-cli", help="llama.cpp binary path")

    args = parser.parse_args()

    config = ModelConfig(
        name=args.name,
        path=args.model_path,
        quantization=args.quant,
        gpu_layers=args.gpu_layers,
        thread_count=args.threads,
        batch_size=args.batch,
        context_size=args.context,
        flash_attention=True,
    )

    benchmark = LLMBenchmark(
        llama_cpp_path=args.llama_path,
        results_dir="results",
    )

    print("Quick Benchmark")
    print(f"Model: {config.name} ({config.quantization})")
    print(f"GPU layers: {config.gpu_layers}")
    print(f"Threads: {config.thread_count}")
    print(f"Batch: {config.batch_size}")
    print(f"Context: {config.context_size}")
    print()

    results = benchmark.run_suite([config], DEFAULT_PROMPTS)

    if results:
        benchmark.save_results()
        benchmark.print_comparison_table()


if __name__ == "__main__":
    main()
