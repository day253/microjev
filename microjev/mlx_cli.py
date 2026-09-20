"""GPT-2 124M + typed decision heads on Apple Silicon."""

import argparse
import json
import sys
from pathlib import Path

from .cli import emit, read_rows
from .gpt2_demo import SCHEMA, dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "train"):
        command = commands.add_parser(name)
        command.add_argument("--base-model", default="openai-community/gpt2")
        command.add_argument("--mode", choices=["heads", "full"], default="full")
        command.add_argument("--epochs", type=int, default=3)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--learning-rate", type=float)
        command.add_argument("--max-length", type=int, default=128)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", default="runs/gpt2-demo" if name == "demo" else None, required=name == "train")
        if name == "train":
            command.add_argument("--schema", required=True)
            command.add_argument("--data", required=True)
            command.add_argument("--validation")
    for name in ("predict", "evaluate", "calibrate"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--max-length", type=int, default=128)
        if name == "predict":
            command.add_argument("--text", required=True)
        else:
            command.add_argument("--data", required=True)
        if name == "calibrate":
            command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        import mlx.core as mx
        from .mlx_backend import GPT2Decision, calibrate, evaluate, fit, parameter_count, predict
    except ImportError as error:
        parser.exit(2, f"MLX backend requires Apple Silicon and the mlx extra: pip install '.[mlx]'\n{error}\n")
    try:
        if args.command in ("demo", "train"):
            schema = SCHEMA if args.command == "demo" else json.loads(Path(args.schema).read_text(encoding="utf-8"))
            rows = dataset() if args.command == "demo" else read_rows(args.data)
            mx.random.seed(args.seed)
            model, tokenizer = GPT2Decision.from_pretrained(schema, args.base_model)
            learning_rate = args.learning_rate if args.learning_rate is not None else (2e-5 if args.mode == "full" else 1e-3)
            print(f"loaded GPT-2: {parameter_count(model):,} parameters; mode={args.mode}; rows={len(rows)}", file=sys.stderr, flush=True)

            def progress(step, epoch, loss):
                if step == 1 or step % 20 == 0:
                    print(f"epoch {epoch}/{args.epochs} step={step} loss={loss:.4f}", file=sys.stderr, flush=True)

            report = {"backend": "mlx", "mode": args.mode, "rows": len(rows)}
            report["training"] = fit(model, tokenizer, rows, mode=args.mode, epochs=args.epochs,
                                     batch_size=args.batch_size, learning_rate=learning_rate,
                                     max_length=args.max_length, seed=args.seed, callback=progress)
            if args.command == "demo":
                validation = dataset("validation")
            else:
                validation = read_rows(args.validation) if args.validation else None
            if validation:
                report["temperatures"] = calibrate(model, tokenizer, validation, args.max_length)
                report["validation"] = evaluate(model, tokenizer, validation, args.max_length)
            if args.command == "demo":
                report["test"] = evaluate(model, tokenizer, dataset("test"), args.max_length)
                report["prediction"] = predict(model, tokenizer, "I want my payment refunded. Please handle this urgently. All work is blocked and there is no workaround.", args.max_length)
                report["note"] = "Synthetic English paraphrase task, 18 held-out examples; not a real-world benchmark or Jev reproduction."
            model.save(args.output, tokenizer)
            report["checkpoint"] = args.output
            (Path(args.output) / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            emit(report)
        else:
            model, tokenizer = GPT2Decision.load(args.model)
            if args.command == "predict":
                emit(predict(model, tokenizer, args.text, args.max_length))
            elif args.command == "evaluate":
                emit(evaluate(model, tokenizer, read_rows(args.data), args.max_length))
            else:
                rows = read_rows(args.data)
                temperatures = calibrate(model, tokenizer, rows, args.max_length)
                model.save(args.output, tokenizer)
                emit({"temperatures": temperatures, "validation": evaluate(model, tokenizer, rows, args.max_length)})
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
