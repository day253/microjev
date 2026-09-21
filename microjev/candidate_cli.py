"""Offline inference with runtime questions and a shared candidate scorer."""

import argparse
import json
import sys
from pathlib import Path

from .schema import validate_schema


def interactive(emit):
    terminal = sys.stdin.isatty()
    if terminal:
        print("输入文本，每行一条；exit / quit、Ctrl-D 或 Ctrl-C 退出。", file=sys.stderr)
    try:
        while True:
            if terminal:
                print("state> ", end="", file=sys.stderr, flush=True)
            text = input()
            if text.strip().lower() in {"exit", "quit"}:
                return
            if not text.strip():
                continue
            try:
                emit(text)
            except ValueError as error:
                # A rejected input (for example, too long) must not end the session.
                print(f"error: {error}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        if terminal:
            print(file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True, help="JSON file containing a question schema")
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--state")
    state.add_argument("--state-file")
    state.add_argument("--interactive", action="store_true",
                       help="Read one state per line; default when no --state or --state-file is supplied")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--prefix-cache", action="store_true", help="Reuse common token-prefix K/V within this request")
    args = parser.parse_args()
    try:
        questions = validate_schema(json.loads(Path(args.questions).read_text(encoding="utf-8")))
        if args.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        from .candidate import CandidateDecision, predict
        model, tokenizer = CandidateDecision.load(args.model)
        if not 2 <= args.max_length <= model.args.n_positions:
            raise ValueError(f"max_length must be between 2 and {model.args.n_positions}")

        def emit(text):
            result = predict(model, tokenizer, text, questions, args.max_length, args.candidate_batch_size, args.prefix_cache)
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)

        if args.state is not None:
            emit(args.state)
        elif args.state_file is not None:
            emit(Path(args.state_file).read_text(encoding="utf-8"))
        else:
            interactive(emit)
    except (ValueError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
