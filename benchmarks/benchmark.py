#!/usr/bin/env python3
"""
Local LLM Inference Benchmark Suite
=====================================
Measures tokens/sec, latency, and memory usage for local LLM inference
across multiple models and optimization configurations.

Built around custom llama.cpp builds with optimized backends (GPU/CPU),
quantization levels, and runtime parameters.
"""

import subprocess
import time
import json
import argparse
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a model benchmark run."""
    name: str
    path: str
    quantization: str
    gpu_layers: int  # -1 for full CPU
    thread_count: int
    batch_size: int
    context_size: int
    flash_attention: bool = False
    mmap: bool = True
    mlock: bool = False


@dataclass
class PromptTemplate:
    """Benchmark prompt with expected output characteristics."""
    name: str
    text: str
    expected_min_tokens: int
    category: str  # "coding", "reasoning", "creative", "summarization"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model: str
    quantization: str
    prompt_name: str
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    time_to_first_token_ms: float
    total_time_ms: float
    gpu_layers: int
    thread_count: int
    batch_size: int
    context_size: int
    flash_attention: bool
    peak_memory_mb: Optional[float] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


DEFAULT_PROMPTS = [
    PromptTemplate(
        name="code_generation",
        text=(
            "Write a Python function that implements a binary search tree with insert, "
            "delete, and search operations. Include type hints and docstrings. "
            "Then write a unit test class for it using pytest."
        ),
        expected_min_tokens=200,
        category="coding"
    ),
    PromptTemplate(
        name="logical_reasoning",
        text=(
            "A train leaves station A at 60 mph. Another train leaves station B at 80 mph. "
            "The stations are 420 miles apart. If they leave at the same time heading toward "
            "each other, how long until they meet? Show your work step by step, then explain "
            "what would happen if train A left 30 minutes earlier."
        ),
        expected_min_tokens=150,
        category="reasoning"
    ),
    PromptTemplate(
        name="code_debugging",
        text=(
            "Debug this Python code that's supposed to implement a rate limiter using "
            "the token bucket algorithm but has several bugs:\n\n"
            "class RateLimiter:\n"
            "    def __init__(self, rate, capacity):\n"
            "        self.rate = rate\n"
            "        self.capacity = capacity\n"
            "        self.tokens = capacity\n"
            "        self.last_time = time.time()\n\n"
            "    def allow(self):\n"
            "        now = time.time()\n"
            "        elapsed = now - self.last_time\n"
            "        self.tokens = self.capacity + elapsed * self.rate\n"
            "        self.last_time = now\n"
            "        if self.tokens >= 1:\n"
            "            self.tokens -= 1\n"
            "            return True\n"
            "        return False\n\n"
            "Identify all bugs and provide the corrected version."
        ),
        expected_min_tokens=300,
        category="coding"
    ),
    PromptTemplate(
        name="system_design",
        text=(
            "Design a multi-agent orchestration system for automated code review. "
            "Describe the architecture, agent roles, communication patterns, "
            "and how you would handle conflicting reviews between agents. "
            "Include considerations for scalability and error handling."
        ),
        expected_min_tokens=250,
        category="reasoning"
    ),
]


class LLMBenchmark:
    """Benchmark local LLM inference using llama.cpp."""

    def __init__(
        self,
        llama_cpp_path: str = "llama-cli",
        results_dir: str = "results",
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ):
        self.llama_cpp_path = llama_cpp_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.results: list[BenchmarkResult] = []

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count (rough estimate for benchmark purposes)."""
        return len(text.split()) * 1.3

    def build_command(
        self,
        model_config: ModelConfig,
        prompt: str,
        n_predict: int = 1024,
    ) -> list[str]:
        """Build llama.cpp command for benchmarking."""
        cmd = [
            self.llama_cpp_path,
            "-m", model_config.path,
            "-p", prompt,
            "-n", str(n_predict),
            "-t", str(model_config.thread_count),
            "-b", str(model_config.batch_size),
            "-c", str(model_config.context_size),
            "-g", str(model_config.gpu_layers),
            "--temp", str(self.temperature),
            "--top-p", str(self.top_p),
            "--repeat-penalty", str(self.repeat_penalty),
            "--no-display-prompt",
            "-s", "0",  # Fixed seed for reproducibility
            "--log-disable",
            "-i",  # Interactive mode disabled for benchmark
        ]

        if model_config.flash_attention:
            cmd.append("-fa")
        if not model_config.mmap:
            cmd.append("--no-mmap")
        if model_config.mlock:
            cmd.append("--mlock")

        return cmd

    def parse_benchmark_output(self, output: str, start_time: float, input_len: int) -> Optional[BenchmarkResult]:
        """Parse llama.cpp benchmark output from stderr."""
        ttf_pattern = r"prompt eval\. time (\d+\.\d+) ms"
        tps_predict_pattern = r"eval\.\t+time (\d+\.\d+) ms\/(\d+) tokens \((\d+\.\d+) tokens\/sec\)"
        tps_prompt_pattern = r"prompt eval\. time.*?\((\d+\.\d+) tokens\/sec\)"
        total_time_pattern = r"total time (\d+\.\d+) ms"

        ttf_match = re.search(ttf_pattern, output)
        tps_match = re.search(tps_predict_pattern, output)
        total_match = re.search(total_time_pattern, output)

        if not tps_match:
            return None

        time_to_first_token = float(ttf_match.group(1)) if ttf_match else 0.0
        output_time_ms = float(tps_match.group(1))
        output_tokens = int(tps_match.group(2))
        tokens_per_sec = float(tps_match.group(3))
        total_time = float(total_match.group(1)) if total_match else (time_to_first_token + output_time_ms)

        return BenchmarkResult(
            model=model_config.name,
            quantization=model_config.quantization,
            prompt_name=prompt_template.name,
            input_tokens=int(self.count_tokens_approx(prompt_template.text)),
            output_tokens=output_tokens,
            tokens_per_second=round(tokens_per_sec, 2),
            time_to_first_token_ms=round(time_to_first_token, 2),
            total_time_ms=round(total_time, 2),
            gpu_layers=model_config.gpu_layers,
            thread_count=model_config.thread_count,
            batch_size=model_config.batch_size,
            context_size=model_config.context_size,
            flash_attention=model_config.flash_attention,
        )

    def run_benchmark(
        self,
        model_config: ModelConfig,
        prompt_template: PromptTemplate,
        n_predict: int = 1024,
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark and return results."""
        cmd = self.build_command(model_config, prompt_template.text, n_predict)

        print(f"\n{'='*60}")
        print(f"Model: {model_config.name} ({model_config.quantization})")
        print(f"Prompt: {prompt_template.name} [{prompt_template.category}]")
        print(f"Config: GPU layers={model_config.gpu_layers}, "
              f"threads={model_config.thread_count}, "
              f"batch={model_config.batch_size}, "
              f"context={model_config.context_size}")
        print(f"{'='*60}")

        try:
            start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            elapsed = time.time() - start_time
            benchmark_result = self.parse_benchmark_output(
                result.stderr, start_time, len(prompt_template.text)
            )

            if benchmark_result:
                self.results.append(benchmark_result)
                print(f"\n✓ Result: {benchmark_result.tokens_per_second} tokens/sec")
                print(f"  TTF: {benchmark_result.time_to_first_token_ms}ms")
                print(f"  Output: {benchmark_result.output_tokens} tokens")
                print(f"  Total: {benchmark_result.total_time_ms}ms")
            else:
                print("✗ Failed to parse benchmark output")
                print(f"stderr: {result.stderr[-500:]}")

            return benchmark_result

        except subprocess.TimeoutExpired:
            print("✗ Benchmark timed out after 300 seconds")
            return None
        except FileNotFoundError:
            print(f"✗ llama.cpp not found at: {self.llama_cpp_path}")
            print("  Build llama.cpp first or update the path.")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def run_suite(
        self,
        model_configs: list[ModelConfig],
        prompts: Optional[list[PromptTemplate]] = None,
    ) -> list[BenchmarkResult]:
        """Run full benchmark suite across models and prompts."""
        prompts = prompts or DEFAULT_PROMPTS
        total = len(model_configs) * len(prompts)
        completed = 0

        print(f"\nStarting benchmark suite: {total} runs ({len(model_configs)} models x {len(prompts)} prompts)\n")

        for model_config in model_configs:
            for prompt_template in prompts:
                completed += 1
                print(f"\n[{completed}/{total}] ", end="")
                self.run_benchmark(model_config, prompt_template)

        return self.results

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON."""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp}.json"

        output_path = self.results_dir / filename

        # Calculate summary statistics
        summary = self._generate_summary()

        output = {
            "metadata": {
                "benchmark_tool": "Local LLM Inference Benchmark Suite",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "llama_cpp_path": self.llama_cpp_path,
                "total_runs": len(self.results),
            },
            "summary": summary,
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def _generate_summary(self) -> dict:
        """Generate summary statistics from results."""
        if not self.results:
            return {}

        by_model = {}
        for r in self.results:
            key = f"{r.model} ({r.quantization})"
            if key not in by_model:
                by_model[key] = []
            by_model[key].append(r)

        summary = {}
        for model_key, results in by_model.items():
            tps_values = [r.tokens_per_second for r in results]
            summary[model_key] = {
                "runs": len(results),
                "avg_tokens_per_second": round(sum(tps_values) / len(tps_values), 2),
                "max_tokens_per_second": round(max(tps_values), 2),
                "min_tokens_per_second": round(min(tps_values), 2),
                "avg_ttf_ms": round(sum(r.time_to_first_token_ms for r in results) / len(results), 2),
                "total_output_tokens": sum(r.output_tokens for r in results),
            }

        return summary

    def print_comparison_table(self):
        """Print a comparison table of results."""
        if not self.results:
            print("No results to display.")
            return

        by_model = {}
        for r in self.results:
            key = f"{r.model} ({r.quantization})"
            if key not in by_model:
                by_model[key] = []
            by_model[key].append(r)

        print(f"\n{'='*80}")
        print("BENCHMARK COMPARISON")
        print(f"{'='*80}")
        print(f"{'Model':<35} {'Avg tok/s':<12} {'Max tok/s':<12} {'Avg TTF ms':<12} {'Runs':<6}")
        print(f"{'-'*80}")

        for model_key, results in sorted(by_model.items(), key=lambda x: sum(r.tokens_per_second for r in x[1]) / len(x[1]), reverse=True):
            tps_values = [r.tokens_per_second for r in results]
            avg_tps = sum(tps_values) / len(tps_values)
            max_tps = max(tps_values)
            avg_ttf = sum(r.time_to_first_token_ms for r in results) / len(results)

            print(f"{model_key:<35} {avg_tps:<12.2f} {max_tps:<12.2f} {avg_ttf:<12.2f} {len(results):<6}")

        print(f"{'='*80}")


def load_model_configs(config_path: str) -> list[ModelConfig]:
    """Load model configurations from JSON file."""
    with open(config_path) as f:
        configs_data = json.load(f)

    return [
        ModelConfig(
            name=c["name"],
            path=c["path"],
            quantization=c["quantization"],
            gpu_layers=c.get("gpu_layers", -1),
            thread_count=c.get("thread_count", 8),
            batch_size=c.get("batch_size", 2048),
            context_size=c.get("context_size", 4096),
            flash_attention=c.get("flash_attention", False),
            mmap=c.get("mmap", True),
            mlock=c.get("mlock", False),
        )
        for c in configs_data
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM Inference Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python benchmark.py -c configs/models.json

  # Run specific model
  python benchmark.py --model /path/to/model.gguf --name "Qwen 3.6 27B" --quant Q4_K_M

  # Run single prompt category
  python benchmark.py -c configs/models.json --category coding
        """,
    )

    parser.add_argument("-c", "--config", help="Path to model configs JSON file")
    parser.add_argument("--model", help="Path to single model file")
    parser.add_argument("--name", help="Model display name", default="custom_model")
    parser.add_argument("--quant", help="Quantization label", default="unknown")
    parser.add_argument("--llama-path", help="Path to llama.cpp binary", default="llama-cli")
    parser.add_argument("--gpu-layers", type=int, help="Number of layers to offload to GPU (-1 for all)", default=-1)
    parser.add_argument("--threads", type=int, help="Number of CPU threads", default=8)
    parser.add_argument("--batch", type=int, help="Batch size", default=2048)
    parser.add_argument("--context", type=int, help="Context size", default=4096)
    parser.add_argument("--category", help="Run only prompts from specific category")
    parser.add_argument("--output", help="Output filename for results JSON")
    parser.add_argument("--results-dir", help="Directory to save results", default="results")
    parser.add_argument("--flash-attention", action="store_true", help="Enable flash attention")
    parser.add_argument("--predict", type=int, help="Max tokens to generate", default=1024)

    args = parser.parse_args()

    benchmark = LLMBenchmark(
        llama_cpp_path=args.llama_path,
        results_dir=args.results_dir,
    )

    # Load model configs
    if args.config:
        model_configs = load_model_configs(args.config)
    elif args.model:
        model_configs = [
            ModelConfig(
                name=args.name,
                path=args.model,
                quantization=args.quant,
                gpu_layers=args.gpu_layers,
                thread_count=args.threads,
                batch_size=args.batch,
                context_size=args.context,
                flash_attention=args.flash_attention,
            )
        ]
    else:
        parser.error("Either --config or --model is required")
        return

    # Filter prompts by category if specified
    prompts = DEFAULT_PROMPTS
    if args.category:
        prompts = [p for p in DEFAULT_PROMPTS if p.category == args.category]
        if not prompts:
            print(f"No prompts found for category: {args.category}")
            return

    # Run benchmarks
    results = benchmark.run_suite(model_configs, prompts)

    # Save and display results
    if results:
        benchmark.save_results(args.output)
        benchmark.print_comparison_table()
    else:
        print("No benchmark results were collected.")


if __name__ == "__main__":
    main()
