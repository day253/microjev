"""Selection-only SST-5 experiments for the question-conditioned shared scorer."""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

from .sst5 import SCHEMA, prepare


PROPOSITION_PAIRS = (
    (
        ("Is the sentiment positive or very positive? Neutral counts as false.",
         "Is the sentiment not positive? Negative and neutral both count as true."),
        ("Does this movie review have positive sentiment?",
         "Does this movie review lack positive sentiment?"),
        ("Is the author's opinion favorable toward the film?",
         "Is the author's opinion NOT favorable toward the film?"),
        ("Would you classify the review as a positive evaluation?",
         "Would you classify the review as anything other than a positive evaluation?"),
    ),
    (
        ("Is the sentiment negative or very negative? Neutral counts as false.",
         "Is the sentiment not negative? Neutral and positive count as true."),
        ("Does this movie review have negative sentiment?",
         "Does this movie review lack negative sentiment?"),
        ("Is the author's opinion unfavorable toward the film?",
         "Is the author's opinion NOT unfavorable toward the film?"),
        ("Would you classify the review as a negative evaluation?",
         "Would you classify the review as anything other than a negative evaluation?"),
    ),
    (
        ("Is the sentiment neutral?",
         "Is the sentiment not neutral? Positive or negative counts as true."),
        ("Does this movie review have neutral sentiment?",
         "Does this movie review lack neutral sentiment?"),
        ("Is the author's opinion neither positive nor negative?",
         "Is the author's opinion positive or negative rather than neutral?"),
        ("Would you classify the review as a neutral evaluation?",
         "Would you classify the review as anything other than a neutral evaluation?"),
    ),
)


def requests_from_rows(rows, augment=False, augmentation="legacy"):
    if augmentation not in ("legacy", "paired", "boundary"):
        raise ValueError("augmentation must be legacy, paired or boundary")
    requests = []
    for i, row in enumerate(rows):
        questions, labels = copy.deepcopy(SCHEMA), dict(row["labels"])
        if augment:
            # ID is deliberately unchanged as semantics change; IDs never enter the model.
            variant, rating = i % 4, labels["rating"]
            propositions = (
                ("Is the sentiment positive or very positive? Neutral counts as false.", rating >= 3),
                ("Is the sentiment negative or very negative? Neutral counts as false.", rating <= 1),
                ("Is the sentiment neutral?", rating == 2),
                ("Is the sentiment not negative? Neutral and positive count as true.", rating >= 2),
            )
            questions["positive"]["instructions"], labels["positive"] = propositions[variant]
            if variant % 2:
                questions["rating"]["criteria"].reverse()
                labels["rating"] = 4 - rating
                questions["rating"]["instructions"] = "Rate the movie review using the supplied sentiment descriptions."
            if variant >= 2:
                questions["sentiment"]["instructions"] = "Which description best matches the review's emotional tone?"
            # Random candidate order is unnecessary for this architecture, but exercise it.
            if variant % 2:
                questions["sentiment"]["criteria"] = dict(reversed(list(questions["sentiment"]["criteria"].items())))
            if augmentation in ("paired", "boundary"):
                predicate, phrasing = i % 3, (i // 3) % 4
                positive, complement = PROPOSITION_PAIRS[predicate][phrasing]
                expected = (rating >= 3, rating <= 1, rating == 2)[predicate]
                questions["positive"]["instructions"] = positive
                labels["positive"] = expected
                questions["complement"] = {"type": "noul", "instructions": complement}
                labels["complement"] = not expected
                if augmentation == "boundary":
                    # Train compatible boundary clauses on all predicates, not just
                    # the wording that failed in the development probes.
                    if (i // 12) % 4:
                        category = (i // 48) % 3
                        name = ("positive", "negative", "neutral")[category]
                        templates = ("For a {name} review, the answer is {truth}.",
                                     "A {name} opinion counts as {truth}.",
                                     "If the text is {name}, return {truth}.")
                        template = templates[(i // 144) % len(templates)]
                        for key, true_on_category in (("positive", predicate == category),
                                                      ("complement", predicate != category)):
                            clause = template.format(name=name, truth=str(true_on_category).lower())
                            original = questions[key]["instructions"]
                            questions[key]["instructions"] = (clause + " " + original if i % 2
                                                               else original + " " + clause)
                    if (i // 6) % 2:
                        for key in ("positive", "complement"):
                            questions[key]["instructions"] = questions[key]["instructions"].replace(" not ", " NOT ")
        requests.append({"state": row["text"], "questions": questions, "labels": labels})
    return requests


def instruction_probes(rows):
    """Held-out training phrasings, repeatedly used for model selection, not final test."""
    result = []
    for row in rows:
        rating = row["labels"]["rating"]
        result.append({"state": row["text"], "questions": {
            "positive_paraphrase": {"type": "noul", "instructions": "Does the review express a favorable opinion of the movie?"},
            "negative_paraphrase": {"type": "noul", "instructions": "Does the review express an unfavorable opinion of the movie?"},
            "not_positive": {"type": "noul", "instructions": "Is the review NOT positive? Count neutral reviews as true."},
        }, "labels": {"positive_paraphrase": rating >= 3, "negative_paraphrase": rating <= 1,
                       "not_positive": rating < 3}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openai-community/gpt2")
    parser.add_argument("--source-kind", choices=("pretrained", "fixed", "candidate"), default="pretrained")
    parser.add_argument("--data-dir", default="data/sst5")
    parser.add_argument("--output", default="runs/candidate-pilot")
    parser.add_argument("--train-limit", type=int, default=1024, help="0 means all training rows")
    parser.add_argument("--selection-limit", type=int, default=550)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4, help="questions per update, each expanded into candidates")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=("legacy", "paired", "boundary"), default="legacy")
    parser.add_argument("--selection-objective", choices=("canonical", "instruction-balanced"), default="canonical")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate", action="store_true", help="Only after choosing the final experiment")
    parser.add_argument("--evaluate-test", action="store_true", help="One-time final test after all selection")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.train_limit < 0 or args.selection_limit < 1:
        parser.error("invalid epochs, batch size or dataset limit")
    if not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1:
        parser.error("label-smoothing must be finite and in [0, 1)")
    if args.evaluate_test and not args.calibrate:
        parser.error("final test requires --calibrate")
    output = Path(args.output)
    if (output / "progress.json").exists() or (output / "training_report.json").exists():
        parser.error("output already contains an experiment; choose a new output directory")
    import mlx.core as mx
    from .candidate import CandidateDecision, calibrate, evaluate, fit, predict
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Fix selection/calibration membership across all model seeds and experiments.
    splits, manifest = prepare(args.data_dir, seed=42)
    rows = list(splits["train"])
    random.Random(args.seed).shuffle(rows)
    if args.train_limit:
        rows = rows[:args.train_limit]
    train = requests_from_rows(rows, augment=True, augmentation=args.augmentation)
    selection_rows = splits["selection"][:args.selection_limit]
    selection = requests_from_rows(selection_rows)
    probes = instruction_probes(selection_rows)
    mx.random.seed(args.seed)
    if args.source_kind == "candidate":
        model, tokenizer = CandidateDecision.load(args.base_model)
        model.source = str(args.base_model)
    else:
        model, tokenizer = CandidateDecision.from_pretrained(args.base_model, decision_checkpoint=args.source_kind == "fixed")
    total_questions = sum(len(request["questions"]) for request in train)
    total_steps = math.ceil(total_questions / args.batch_size) * args.epochs
    best, reports = {"score": float("inf"), "epoch": None}, []
    training_started = time.perf_counter()

    def progress(step, epoch, loss):
        if step == 1 or step % 50 == 0:
            elapsed = time.perf_counter() - training_started
            print(f"epoch={epoch}/{args.epochs} step={step}/{total_steps} loss={loss:.4f} elapsed={elapsed:.1f}s eta_train={elapsed / step * (total_steps - step):.1f}s",
                  file=sys.stderr, flush=True)

    def finish_epoch(epoch, loss):
        metrics = evaluate(model, tokenizer, selection, args.max_length)
        nll = sum(q["nll"] for q in metrics["questions"].values()) / len(SCHEMA)
        probe_metrics = None
        score = nll
        if args.selection_objective == "instruction-balanced":
            probe_metrics = evaluate(model, tokenizer, probes, args.max_length)
            probe_nll = sum(q["nll"] for q in probe_metrics["questions"].values()) / len(probe_metrics["questions"])
            score = (nll + probe_nll) / 2
        model.save(output / "checkpoints" / f"epoch-{epoch}", tokenizer)
        if score < best["score"]:
            best.update(score=score, epoch=epoch)
        reports.append({"epoch": epoch, "training_loss": loss, "selection_nll": nll, "selection": metrics,
                        "selection_score": score, "instruction_probes": probe_metrics})
        (output / "progress.json").write_text(json.dumps({"config": vars(args), "best": best, "epochs": reports}, indent=2) + "\n")
        print(f"saved epoch {epoch}; selection_nll={nll:.4f}; selection_score={score:.4f}; best_epoch={best['epoch']}", file=sys.stderr, flush=True)

    print(json.dumps({"train_rows": len(train), "selection_rows": len(selection), "steps": total_steps, "test_enabled": args.evaluate_test}), file=sys.stderr, flush=True)
    training = fit(model, tokenizer, train, epochs=args.epochs, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, max_length=args.max_length, seed=args.seed,
                   callback=progress, epoch_callback=finish_epoch, label_smoothing=args.label_smoothing)
    del model
    mx.clear_cache()
    model, tokenizer = CandidateDecision.load(output / "checkpoints" / f"epoch-{best['epoch']}")
    # Probes use only selection rows. Do not use held-out test or calibration to tune.
    probe_metrics = evaluate(model, tokenizer, probes, args.max_length)
    calibration_report, test = None, None
    if args.calibrate:
        calibration_report = calibrate(model, tokenizer, requests_from_rows(splits["calibration"]), args.max_length,
                                       scope="SST-5 canonical sentiment/positive/rating questions; other questions unvalidated")
    if args.evaluate_test:
        test = evaluate(model, tokenizer, requests_from_rows(splits["test"]), args.max_length)
    model.save(output / "best", tokenizer)
    example = "A wonderful film with moving performances and a brilliant story."
    predict(model, tokenizer, example, SCHEMA, args.max_length)  # warmup
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        prediction = predict(model, tokenizer, example, SCHEMA, args.max_length)
        timings.append(time.perf_counter() - start)
    report = {"dataset": manifest, "config": vars(args), "train_rows": len(train), "selection_rows": len(selection),
              "training": training, "epochs": reports, "best_epoch": best["epoch"], "instruction_probes": probe_metrics,
              "calibration": calibration_report, "test": test, "prediction": {"state": example, "result": prediction},
              "inference_seconds_mean": sum(timings) / len(timings), "inference_seconds_samples": timings,
              "total_seconds": time.perf_counter() - started, "checkpoint": str(output / "best"),
              "limitations": "English sentiment, derived questions from one source label. Runtime conditioning does not establish general instruction following. Candidate batching repeats state; no shared-prefix cache or Jev RLCD."}
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
