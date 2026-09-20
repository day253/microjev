"""Compare serial, batched and prefix-cached scoring on the same checkpoint."""

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.candidate_batch_size < 1:
        parser.error("warmup, repeats and candidate-batch-size must be positive")
    from .autograd import softmax
    from .candidate import CandidateDecision, encode_requests, score_sequences
    from .sst5 import SCHEMA
    model, tokenizer = CandidateDecision.load(args.model)
    model.eval()
    base = "A wonderful film with moving performances and a brilliant story."
    sentence = " The acting is thoughtful and the pacing is steady."
    cases = {"short": base, "medium": base + sentence * 12, "long": base + sentence * 55}
    report = {"checkpoint": args.model, "config": vars(args), "cases": {},
              "scope": "GPU-synchronized scoring only; excludes tokenization and JSON construction. Long repeated text tests speed/equivalence, not language quality. Prefix caching is per request, not cross-request reuse."}
    for name, state in cases.items():
        groups = encode_requests(model, tokenizer, [{"state": state, "questions": SCHEMA}], max_length=1024, labeled=False)
        sequences = [seq for group in groups for seq in group.sequences]

        def run(mode):
            if mode == "serial_questions":
                values = []
                for group in groups:
                    scores, _ = score_sequences(model, group.sequences, tokenizer.eos_token_id, args.candidate_batch_size)
                    values.extend(scores)
                return values, 0
            return score_sequences(model, sequences, tokenizer.eos_token_id, args.candidate_batch_size,
                                   prefix_cache=mode == "batched_prefix_cache")

        modes = ("serial_questions", "batched", "batched_prefix_cache")
        results, times = {}, {mode: [] for mode in modes}
        for mode in modes:
            for _ in range(args.warmup):
                results[mode] = run(mode)
        # Rotate mode order to reduce order/thermal bias between variants.
        for repeat in range(args.repeats):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                start = time.perf_counter()
                results[mode] = run(mode)
                times[mode].append(time.perf_counter() - start)
        reference, _ = results["serial_questions"]
        item = {"state_tokens": len(tokenizer.encode(state, add_special_tokens=False)),
                "candidate_count": len(sequences), "logical_tokens": sum(map(len, sequences)), "modes": {}}
        for mode in modes:
            scores, prefix = results[mode]
            offset, max_probability_difference = 0, 0.0
            for group in groups:
                size = len(group.sequences)
                expected = softmax(reference[offset:offset + size])
                actual = softmax(scores[offset:offset + size])
                max_probability_difference = max(max_probability_difference, max(abs(a - b) for a, b in zip(expected, actual)))
                offset += size
            if max_probability_difference > 1e-4:
                raise ValueError(f"{name} {mode}: probability equivalence failed ({max_probability_difference})")
            item["modes"][mode] = {"mean_seconds": statistics.mean(times[mode]), "median_seconds": statistics.median(times[mode]),
                                    "samples_seconds": times[mode], "shared_prefix_tokens": prefix,
                                    "forward_tokens": item["logical_tokens"] - prefix * (len(sequences) - 1),
                                    "max_logit_difference": max(abs(a - b) for a, b in zip(reference, scores)),
                                    "max_probability_difference": max_probability_difference}
        report["cases"][name] = item
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
