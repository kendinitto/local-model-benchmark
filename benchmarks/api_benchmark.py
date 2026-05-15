#!/usr/bin/env python3
"""
API Benchmark Module — Benchmarks llama.cpp via HTTP server.
Connects to any running llama.cpp server regardless of platform/backend.
"""

import requests
import time
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path


@dataclass
class ApiBenchmarkResult:
    """Results from an API benchmark run."""
    model: str
    prompt_name: str
    prompt_category: str
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float
    time_to_first_token_ms: float
    total_time_ms: float
    server_url: str
    n_predict: int
    temperature: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class PromptTemplate:
    """Benchmark prompt."""
    name: str
    text: str
    expected_min_tokens: int
    category: str


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


class ApiBenchmark:
    """Benchmark llama.cpp server via HTTP API."""

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        results_dir: str = "results",
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ):
        self.server_url = server_url.rstrip("/")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.results: list[ApiBenchmarkResult] = []
        self.model_name = None

    def get_model_info(self) -> str:
        """Get the currently loaded model name from the server."""
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=10)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                if models:
                    self.model_name = models[0].get("id", "unknown_model")
                    return self.model_name
        except Exception:
            pass
        self.model_name = "api_model"
        return self.model_name

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split()) * 1.3

    def run_completion(
        self,
        prompt: str,
        n_predict: int = 1024,
    ) -> Optional[dict]:
        """Run a single completion request against the API."""
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "stream": True,
        }

        token_count = 0
        first_token_time = None
        prompt_tokens = 0
        final_data = None
        start_time = time.perf_counter()

        try:
            with requests.post(
                f"{self.server_url}/completion",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    print(f"  API error: {resp.status_code} — {resp.text[:200]}")
                    return None

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    try:
                        data = json.loads(data_str)
                        if first_token_time is None and data.get("content"):
                            first_token_time = (time.perf_counter() - start_time) * 1000
                        if data.get("tokens"):
                            token_count += len(data["tokens"])
                        if data.get("tokens_evaluated"):
                            prompt_tokens = data["tokens_evaluated"]
                        if data.get("stop"):
                            final_data = data
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000

            if final_data and "timing" in final_data:
                timing = final_data["timing"]
                tps_predict = timing.get("predicted_per_second", 0)
                tps = tps_predict if tps_predict > 0 else (token_count / (total_time_ms / 1000))
            else:
                tps = token_count / (total_time_ms / 1000) if total_time_ms > 0 else 0

            return {
                "prompt_tokens": prompt_tokens or int(self.count_tokens_approx(prompt)),
                "generated_tokens": token_count,
                "tokens_per_second": round(tps, 2),
                "time_to_first_token_ms": round(first_token_time, 2) if first_token_time else 0,
                "total_time_ms": round(total_time_ms, 2),
            }

        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to {self.server_url}")
            return None
        except requests.exceptions.Timeout:
            print("  ✗ Request timed out after 300 seconds")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def run_openai_completion(
        self,
        prompt: str,
        n_predict: int = 1024,
    ) -> Optional[dict]:
        """Run completion using OpenAI-compatible API endpoint."""
        payload = {
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }

        token_count = 0
        first_token_time = None
        start_time = time.perf_counter()

        try:
            with requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    print(f"  API error: {resp.status_code} — {resp.text[:200]}")
                    return None

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and choices[0].get("delta", {}).get("content"):
                            if first_token_time is None:
                                first_token_time = (time.perf_counter() - start_time) * 1000
                            token_count += 1
                        usage = data.get("usage")
                        if usage:
                            return {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "generated_tokens": usage.get("completion_tokens", token_count),
                                "tokens_per_second": 0,  # Will calculate below
                                "time_to_first_token_ms": round(first_token_time, 2) if first_token_time else 0,
                                "total_time_ms": 0,  # Will calculate below
                                "_raw_token_count": token_count,
                            }
                    except json.JSONDecodeError:
                        continue

            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            tps = token_count / (total_time_ms / 1000) if total_time_ms > 0 else 0

            return {
                "prompt_tokens": int(self.count_tokens_approx(prompt)),
                "generated_tokens": token_count,
                "tokens_per_second": round(tps, 2),
                "time_to_first_token_ms": round(first_token_time, 2) if first_token_time else 0,
                "total_time_ms": round(total_time_ms, 2),
            }

        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to {self.server_url}")
            return None
        except requests.exceptions.Timeout:
            print("  ✗ Request timed out after 300 seconds")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def run_benchmark(
        self,
        prompt_template: PromptTemplate,
        n_predict: int = 1024,
        api_mode: str = "auto",
    ) -> Optional[ApiBenchmarkResult]:
        """Run a single benchmark prompt."""
        print(f"\n  Prompt: {prompt_template.name} [{prompt_template.category}]")
        print(f"  Predict: {n_predict} tokens")

        if api_mode == "auto":
            api_mode = self._detect_api_mode()

        if api_mode == "native":
            completion = self.run_completion(prompt_template.text, n_predict)
        else:
            completion = self.run_openai_completion(prompt_template.text, n_predict)

        if not completion:
            return None

        result = ApiBenchmarkResult(
            model=self.model_name or "unknown",
            prompt_name=prompt_template.name,
            prompt_category=prompt_template.category,
            prompt_tokens=completion["prompt_tokens"],
            generated_tokens=completion["generated_tokens"],
            tokens_per_second=completion["tokens_per_second"],
            time_to_first_token_ms=completion["time_to_first_token_ms"],
            total_time_ms=completion["total_time_ms"],
            server_url=self.server_url,
            n_predict=n_predict,
            temperature=self.temperature,
        )

        self.results.append(result)
        print(f"  ✓ {result.tokens_per_second} tok/s | TTF: {result.time_to_first_token_ms}ms | {result.generated_tokens} tokens")
        return result

    def _detect_api_mode(self) -> str:
        """Detect whether server supports native or OpenAI API."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            if resp.status_code in (200, 503):
                return "native"
        except Exception:
            pass
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                return "openai"
        except Exception:
            pass
        return "native"

    def run_suite(
        self,
        prompts: Optional[list[PromptTemplate]] = None,
        n_predict: int = 1024,
        api_mode: str = "auto",
    ) -> list[ApiBenchmarkResult]:
        """Run full benchmark suite."""
        prompts = prompts or DEFAULT_PROMPTS

        print(f"\n{'='*60}")
        print(f"API Benchmark Suite")
        print(f"Server: {self.server_url}")
        print(f"Mode: {api_mode}")
        print(f"{'='*60}")

        self.get_model_info()
        print(f"Model: {self.model_name}")
        print(f"Running {len(prompts)} prompts...\n")

        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] ", end="")
            result = self.run_benchmark(prompt, n_predict, api_mode)
            if result:
                results.append(result)

        return results

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON."""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"api_benchmark_{timestamp}.json"

        output_path = self.results_dir / filename

        summary = self._generate_summary()

        output = {
            "metadata": {
                "benchmark_tool": "Local LLM API Benchmark Suite",
                "server_url": self.server_url,
                "model": self.model_name,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
        """Generate summary statistics."""
        if not self.results:
            return {}

        tps_values = [r.tokens_per_second for r in self.results]
        return {
            "model": self.model_name,
            "runs": len(self.results),
            "avg_tokens_per_second": round(sum(tps_values) / len(tps_values), 2),
            "max_tokens_per_second": round(max(tps_values), 2),
            "min_tokens_per_second": round(min(tps_values), 2),
            "avg_ttf_ms": round(sum(r.time_to_first_token_ms for r in self.results) / len(self.results), 2),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self.results),
            "total_generated_tokens": sum(r.generated_tokens for r in self.results),
        }

    def print_summary(self):
        """Print results summary."""
        if not self.results:
            print("No results to display.")
            return

        tps_values = [r.tokens_per_second for r in self.results]

        print(f"\n{'='*70}")
        print(f"RESULTS — {self.model_name} via {self.server_url}")
        print(f"{'='*70}")
        print(f"{'Prompt':<25} {'tok/s':<10} {'TTF ms':<10} {'Tokens':<10} {'Category':<10}")
        print(f"{'-'*70}")

        for r in self.results:
            print(f"{r.prompt_name:<25} {r.tokens_per_second:<10.2f} {r.time_to_first_token_ms:<10.2f} {r.generated_tokens:<10} {r.prompt_category:<10}")

        print(f"{'-'*70}")
        print(f"{'AVERAGE':<25} {sum(tps_values)/len(tps_values):<10.2f} {sum(r.time_to_first_token_ms for r in self.results)/len(self.results):<10.2f} {sum(r.generated_tokens for r in self.results):<10}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="API Benchmark for llama.cpp server")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--predict", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--category", help="Run only prompts from specific category")
    parser.add_argument("--api-mode", choices=["auto", "native", "openai"], default="auto", help="API mode")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--results-dir", default="results", help="Results directory")

    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.category:
        prompts = [p for p in DEFAULT_PROMPTS if p.category == args.category]

    benchmark = ApiBenchmark(
        server_url=args.server,
        results_dir=args.results_dir,
    )

    results = benchmark.run_suite(prompts, args.predict, args.api_mode)

    if results:
        benchmark.save_results(args.output)
        benchmark.print_summary()
    else:
        print("No benchmark results collected.")


if __name__ == "__main__":
    main()
