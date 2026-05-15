#!/usr/bin/env python3
"""
Quick benchmark runner — CLI or API mode.
Convenience script for rapid iteration.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.benchmark import LLMBenchmark, ModelConfig, DEFAULT_PROMPTS
from benchmarks.api_benchmark import ApiBenchmark


def main():
    parser = argparse.ArgumentParser(
        description="Quick LLM Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CLI Mode (requires llama.cpp binary):
  python quick_bench.py /path/to/model.gguf --name "Qwen 3.6 27B" --quant Q4_K_M

API Mode (connects to running server):
  python quick_bench.py --api --server http://192.168.1.100:8080
  python quick_bench.py --api  # defaults to http://localhost:8080
        """,
    )

    parser.add_argument("model_path", nargs="?", help="Path to model .gguf file (CLI mode)")
    parser.add_argument("--api", action="store_true", help="Use API mode (connect to running server)")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL for API mode")
    parser.add_argument("--name", default="custom", help="Model display name")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization label")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--batch", type=int, default=4096, help="Batch size")
    parser.add_argument("--context", type=int, default=8192, help="Context size")
    parser.add_argument("--llama-path", default="./llama.cpp/build/llama-cli", help="llama.cpp binary path")
    parser.add_argument("--api-mode", choices=["auto", "native", "openai"], default="auto", help="API mode")
    parser.add_argument("--predict", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--category", help="Run only prompts from specific category")

    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.category:
        prompts = [p for p in DEFAULT_PROMPTS if p.category == args.category]

    if args.api:
        benchmark = ApiBenchmark(
            server_url=args.server,
            results_dir="results",
        )
        print("API Benchmark Mode")
        print(f"Server: {args.server}")
        print()

        results = benchmark.run_suite(prompts, args.predict, args.api_mode)

        if results:
            benchmark.save_results()
            benchmark.print_summary()
    elif args.model_path:
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

        print("CLI Benchmark Mode")
        print(f"Model: {config.name} ({config.quantization})")
        print(f"GPU layers: {config.gpu_layers}")
        print(f"Threads: {config.thread_count}")
        print(f"Batch: {config.batch_size}")
        print(f"Context: {config.context_size}")
        print()

        results = benchmark.run_suite([config], prompts)

        if results:
            benchmark.save_results()
            benchmark.print_comparison_table()
    else:
        parser.error("Either MODEL_PATH or --api is required")


if __name__ == "__main__":
    main()
