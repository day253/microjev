"""Offline inference with runtime questions and a shared candidate scorer."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--questions", required=True, help="JSON file containing a question schema")
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--state")
    state.add_argument("--state-file")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        from .candidate import CandidateDecision, predict
        model, tokenizer = CandidateDecision.load(args.model)
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        text = args.state if args.state is not None else Path(args.state_file).read_text(encoding="utf-8")
        result = predict(model, tokenizer, text, questions, args.max_length, args.candidate_batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
