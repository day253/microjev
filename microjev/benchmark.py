"""Short, reproducible GPT-2 decision-training benchmark. No downloads."""

import argparse
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

from .demo import SCHEMA
from .mlx_backend import GPT2Decision, make_step, parameter_count
from .schema import cardinality


def benchmark(mode, batch_size=4, sequence_length=128, warmup=3, steps=10):
    if mode not in ("full", "heads") or min(batch_size, sequence_length, warmup, steps) < 1:
        raise ValueError("invalid benchmark mode or nonpositive dimensions/steps")
    if sequence_length > 1024:
        raise ValueError("GPT-2 supports at most 1024 positions")
    mx.random.seed(42)
    model = GPT2Decision(SCHEMA)
    model.set_train_mode(mode)
    mx.eval(model.parameters())
    tokens = mx.random.randint(0, 50257, (batch_size, sequence_length))
    lengths = mx.full((batch_size,), sequence_length, dtype=mx.int32)
    labels = [mx.eye(cardinality(q))[mx.arange(batch_size) % cardinality(q)] for q in SCHEMA.values()]
    mx.eval(tokens, lengths, labels)
    step = make_step(model, 2e-5 if mode == "full" else 1e-3)
    mx.reset_peak_memory()
    durations, losses = [], []
    for i in range(warmup + steps):
        started = time.perf_counter()
        loss = step(tokens, lengths, labels)
        duration = time.perf_counter() - started
        if i >= warmup:
            durations.append(duration)
            losses.append(loss)
        print(f"{mode} step {i + 1}/{warmup + steps}: {duration:.3f}s loss={loss:.4f}", file=sys.stderr, flush=True)
    mean = statistics.mean(durations)
    result = {"mode": mode, "parameters": parameter_count(model),
              "trainable_parameters": parameter_count(model, True),
              "batch_size": batch_size, "sequence_length": sequence_length,
              "dtype": "float32", "warmup_steps": warmup, "measured_steps": steps,
              "seconds_per_step_mean": mean, "seconds_per_step_median": statistics.median(durations),
              "examples_per_second": batch_size / mean,
              "padded_tokens_per_second": batch_size * sequence_length / mean,
              "peak_active_memory_gib": mx.get_peak_memory() / 1024 ** 3,
              "final_loss": losses[-1], "step_seconds": durations,
              "mlx_version": importlib.metadata.version("mlx"),
              "mlx_lm_version": importlib.metadata.version("mlx-lm"),
              "notes": "Random weights and synthetic IDs; typed decision loss, not language-model pretraining. GPU updates synchronized. No feature cache or compile. Excludes loading, tokenization, validation, saving, thermal drift."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["heads", "full"], default="full")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = benchmark(args.mode, args.batch_size, args.sequence_length, args.warmup, args.steps)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
