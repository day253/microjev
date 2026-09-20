"""Calibrate and test an already selected candidate checkpoint without retraining."""

import argparse
import json
import time
from pathlib import Path

from .candidate_sst5 import choice_probes, instruction_probes, requests_from_rows
from .sst5 import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint selected using selection data only")
    parser.add_argument("--output", required=True, help="New directory for the final calibrated checkpoint")
    parser.add_argument("--data-dir", default="data/sst5")
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output must be a new directory; preserve existing finalized models")
    from .candidate import CandidateDecision, calibrate, evaluate
    started = time.perf_counter()
    splits, manifest = prepare(args.data_dir, seed=42)
    model, tokenizer = CandidateDecision.load(args.model)
    calibration = calibrate(model, tokenizer, requests_from_rows(splits["calibration"]), args.max_length,
                            scope="Temperature fitted on SST-5 canonical sentiment/positive/rating; no general calibration guarantee for other questions")
    test = evaluate(model, tokenizer, requests_from_rows(splits["test"]), args.max_length)
    instruction_test = evaluate(model, tokenizer, instruction_probes(splits["test"]), args.max_length)
    choice_test = evaluate(model, tokenizer, choice_probes(splits["test"]), args.max_length)
    model.save(output, tokenizer)
    report = {"config": vars(args), "dataset": manifest, "calibration": calibration, "test": test,
              "instruction_test": instruction_test, "dynamic_choice_test": choice_test,
              "seconds": time.perf_counter() - started, "checkpoint": str(output),
              "selection_policy": "The input checkpoint must be selected before running this command. Test scores are not for further tuning."}
    (output / "final_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
