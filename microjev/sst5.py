"""Reproducible real-data GPT-2 decision fine-tuning on Stanford SST-5."""

import argparse
import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

REVISION = "e51bdcd8cd3a30da231967c1a249ba59361279a3"
SOURCE = "https://huggingface.co/datasets/SetFit/sst5"
SCHEMA = {
    "sentiment": {"type": "choice", "instructions": "Classify the sentiment of the English movie review.",
                  "criteria": {"negative": "Very negative or negative", "neutral": "Neutral", "positive": "Positive or very positive"}},
    "positive": {"type": "noul", "instructions": "Is the sentiment positive or very positive? Neutral counts as false."},
    "rating": {"type": "score", "instructions": "Rate sentiment on the Stanford SST-5 scale.",
               "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]},
}


def convert(row):
    label = row["label"]
    if type(label) is not int or not 0 <= label <= 4 or not isinstance(row.get("text"), str):
        raise ValueError("unexpected SST-5 record")
    return {"text": row["text"], "labels": {
        "sentiment": "negative" if label < 2 else "neutral" if label == 2 else "positive",
        "positive": label >= 3, "rating": label}}


def prepare(directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    splits, hashes = {}, {}
    for split in ("train", "dev", "test"):
        path = directory / f"{split}.raw.jsonl"
        if not path.exists():
            url = f"{SOURCE}/resolve/{REVISION}/{split}.jsonl"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            # Verify syntax before promoting the download to the cache.
            [convert(json.loads(line)) for line in payload.decode("utf-8").splitlines() if line.strip()]
            path.write_bytes(payload)
        payload = path.read_bytes()
        hashes[split] = hashlib.sha256(payload).hexdigest()
        splits[split] = [convert(json.loads(line)) for line in payload.decode("utf-8").splitlines() if line.strip()]
    # Separate model selection from post-training temperature fitting.
    dev = list(splits.pop("dev"))
    random.Random(seed).shuffle(dev)
    midpoint = len(dev) // 2
    splits["selection"], splits["calibration"] = dev[:midpoint], dev[midpoint:]
    for split, rows in splits.items():
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"source": SOURCE, "revision": REVISION, "raw_sha256": hashes,
                "counts": {key: len(rows) for key, rows in splits.items()}, "dev_shuffle_seed": seed,
                "labels": "All three heads are supervised projections of the same SST-5 label; not independent tasks."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return splits, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openai-community/gpt2")
    parser.add_argument("--data-dir", default="data/sst5")
    parser.add_argument("--output", default="runs/gpt2-sst5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    import mlx.core as mx
    from .mlx_backend import GPT2Decision, calibrate, evaluate, fit, predict
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    splits, manifest = prepare(args.data_dir, args.seed)
    print(json.dumps({"data": manifest["counts"]}), file=sys.stderr, flush=True)
    mx.random.seed(args.seed)
    model, tokenizer = GPT2Decision.from_pretrained(SCHEMA, args.base_model)
    total_steps = math.ceil(len(splits["train"]) / args.batch_size) * args.epochs
    epoch_reports, best = [], {"nll": float("inf"), "epoch": None}
    training_started = time.perf_counter()

    def progress(step, epoch, loss):
        if step == 1 or step % 100 == 0:
            elapsed = time.perf_counter() - training_started
            remaining = elapsed / step * (total_steps - step)
            print(f"epoch={epoch}/{args.epochs} step={step}/{total_steps} loss={loss:.4f} elapsed={elapsed:.1f}s eta_train={remaining:.1f}s",
                  file=sys.stderr, flush=True)

    def finish_epoch(epoch, training_loss):
        metrics = evaluate(model, tokenizer, splits["selection"], args.max_length)
        nll = sum(q["nll"] for q in metrics["questions"].values()) / len(SCHEMA)
        checkpoint = output / "checkpoints" / f"epoch-{epoch}"
        model.save(checkpoint, tokenizer)
        if nll < best["nll"]:
            best.update(nll=nll, epoch=epoch)
        epoch_reports.append({"epoch": epoch, "training_loss": training_loss, "selection_nll": nll, "selection": metrics})
        (output / "progress.json").write_text(json.dumps({"best": best, "epochs": epoch_reports}, indent=2))
        print(f"saved epoch {epoch}; selection_nll={nll:.4f}; best_epoch={best['epoch']}", file=sys.stderr, flush=True)

    training = fit(model, tokenizer, splits["train"], mode="full", epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate,
                   max_length=args.max_length, seed=args.seed, callback=progress,
                   epoch_callback=finish_epoch)
    del model
    mx.clear_cache()
    model, tokenizer = GPT2Decision.load(output / "checkpoints" / f"epoch-{best['epoch']}")
    # Never consult test labels for training, checkpoint choice, or calibration.
    calibration_before = evaluate(model, tokenizer, splits["calibration"], args.max_length)
    temperatures = calibrate(model, tokenizer, splits["calibration"], args.max_length)
    calibration_after = evaluate(model, tokenizer, splits["calibration"], args.max_length)
    test = evaluate(model, tokenizer, splits["test"], args.max_length)
    model.save(output / "best", tokenizer)
    examples = [
        "A wonderful film with moving performances and a brilliant story.",
        "A tedious and disappointing movie that wastes its talented cast.",
        "The movie has some good scenes, but the rest is ordinary.",
    ]
    predictions = [{"text": text, "result": predict(model, tokenizer, text, args.max_length)} for text in examples]
    report = {"dataset": manifest, "config": vars(args), "training": training,
              "epochs": epoch_reports, "best_epoch": best["epoch"], "temperatures": temperatures,
              "calibration_before": calibration_before, "calibration_after": calibration_after,
              "test": test, "predictions": predictions,
              "total_seconds": time.perf_counter() - started, "checkpoint": str(output / "best"),
              "limitations": "English movie sentiment only. Fixed schema. Three labels derived from one source label. Separate heads need not be logically consistent. Test results are not Jev capability claims."}
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
