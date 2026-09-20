import argparse
import json
import sys
from pathlib import Path

from .demo import SCHEMA, dataset
from .model import MicroJev


def read_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    return rows


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser():
    result = argparse.ArgumentParser(description="Train tiny typed decision transformers, using only Python.")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        command = commands.add_parser(name, help="train a model" if name == "train" else "train and evaluate a synthetic demo")
        command.add_argument("--steps", type=int, default=300)
        command.add_argument("--learning-rate", type=float, default=0.01)
        command.add_argument("--n-embd", type=int, default=16)
        command.add_argument("--n-head", type=int, default=4)
        command.add_argument("--n-layer", type=int, default=1)
        command.add_argument("--block-size", type=int, default=64)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", default="runs/demo.json" if name == "demo" else None, required=name == "train")
        if name == "train":
            command.add_argument("--schema", required=True)
            command.add_argument("--data", required=True)
    predict = commands.add_parser("predict", help="load a checkpoint and return typed decisions")
    predict.add_argument("--model", required=True)
    text = predict.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file")
    for name in ("evaluate", "calibrate"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--data", required=True)
        if name == "calibrate":
            command.add_argument("--output", required=True)
    return result


def run(args):
    if args.command in ("demo", "train"):
        schema = SCHEMA if args.command == "demo" else json.loads(Path(args.schema).read_text(encoding="utf-8"))
        rows = dataset() if args.command == "demo" else read_rows(args.data)
        model = MicroJev.from_rows(schema, rows, n_embd=args.n_embd, n_head=args.n_head,
                                  n_layer=args.n_layer, block_size=args.block_size, seed=args.seed)

        def progress(step, loss):
            if step == 1 or step % 25 == 0 or step == args.steps:
                print(f"step {step}/{args.steps} loss={loss:.4f}", file=sys.stderr, flush=True)

        losses = model.fit(rows, steps=args.steps, learning_rate=args.learning_rate,
                           seed=args.seed, callback=progress)
        report = {"checkpoint": args.output, "parameters": len(model.params),
                  "initial_step_loss": losses[0], "final_step_loss": losses[-1]}
        if args.command == "demo":
            report["note"] = "Synthetic keyword/template task; not evidence of natural-language understanding."
            report["temperatures"] = model.calibrate(dataset("validation"))
            report["test"] = model.evaluate(dataset("test"))
            report["prediction"] = model.predict("工单：加急，退款，影响严重")
        model.save(args.output)
        emit(report)
        return
    model = MicroJev.load(args.model)
    if args.command == "predict":
        text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
        emit(model.predict(text))
    elif args.command == "evaluate":
        emit(model.evaluate(read_rows(args.data)))
    else:
        rows = read_rows(args.data)
        before = model.evaluate(rows)
        temperatures = model.calibrate(rows)
        model.save(args.output)
        emit({"temperatures": temperatures, "validation_before": before,
              "validation_after": model.evaluate(rows), "checkpoint": args.output})


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as error:
        arguments.exit(2, f"error: {error}\n")
