"""GPT-2 backbone and typed heads on Apple Silicon (optional MLX dependency)."""

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load as load_pretrained
from mlx_lm.models.gpt2 import GPT2Model, ModelArgs
from transformers import AutoTokenizer

from .autograd import cross_entropy, softmax
from .schema import answer, cardinality, targets, validate_schema


def gpt2_args():
    return ModelArgs(model_type="gpt2", n_ctx=1024, n_embd=768, n_head=12,
                     n_layer=12, n_positions=1024, layer_norm_epsilon=1e-5,
                     vocab_size=50257)


class GPT2Decision(nn.Module):
    def __init__(self, schema, args=None, backbone=None):
        super().__init__()
        self.schema = validate_schema(schema)
        self.args = args or gpt2_args()
        self.backbone = GPT2Model(self.args) if backbone is None else backbone
        self.heads = [nn.Linear(self.args.n_embd, cardinality(q)) for q in self.schema.values()]
        self.temperatures = [1.0] * len(self.heads)
        self.calibration = "unfitted"
        self.source = "random"

    def __call__(self, tokens, lengths):
        hidden = self.backbone(tokens)
        # Right padding is causally after the pooled EOS; it cannot affect the result.
        pooled = hidden[mx.arange(tokens.shape[0]), lengths - 1]
        return [head(pooled) for head in self.heads]

    def set_train_mode(self, mode):
        if mode not in ("heads", "full"):
            raise ValueError("mode must be heads or full")
        self.unfreeze()
        if mode == "heads":
            self.backbone.freeze()
        self.train()

    @classmethod
    def from_pretrained(cls, schema, source="openai-community/gpt2"):
        base, tokenizer = load_pretrained(source, tokenizer_config={"trust_remote_code": False})
        if getattr(base, "model_type", None) != "gpt2":
            raise ValueError("this backend only supports GPT-2 checkpoints")
        model = cls(schema, args=base.args, backbone=base.model)
        model.source = source
        # Float32 parameter updates avoid losing small fine-tuning updates to bf16 rounding.
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
        return model, tokenizer

    def save(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_weights(str(directory / "model.safetensors"))
        tokenizer.save_pretrained(str(directory))
        metadata = dict(format="microjev-gpt2-mlx", version=1, schema=self.schema,
                        args=asdict(self.args), source=self.source,
                        temperatures=self.temperatures, calibration=self.calibration)
        (directory / "microjev.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "microjev.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "microjev-gpt2-mlx" or metadata.get("version") != 1:
            raise ValueError("unsupported MLX decision checkpoint")
        model = cls(metadata["schema"], ModelArgs(**metadata["args"]))
        temperatures = metadata["temperatures"]
        if len(temperatures) != len(model.heads) or any(not math.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError("invalid checkpoint temperatures")
        model.temperatures = temperatures
        model.calibration = metadata["calibration"]
        model.source = metadata["source"]
        model.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(model.parameters())
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        return model, tokenizer


def encode_rows(model, tokenizer, rows, max_length=128, labeled=True):
    if not rows:
        raise ValueError("dataset must not be empty")
    if not 2 <= max_length <= model.args.n_positions:
        raise ValueError(f"max_length must be between 2 and {model.args.n_positions}")
    encoded = []
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer requires an EOS token")
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"row {i}: expected a text string")
        tokens = tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
        if len(tokens) > max_length:
            raise ValueError(f"row {i}: {len(tokens)} tokens exceeds max_length={max_length}; no silent truncation")
        labels = targets(model.schema, row.get("labels")) if labeled else None
        encoded.append((tokens, labels))
    return encoded


def batchify(encoded, schema, pad_id):
    length = max(len(tokens) for tokens, _ in encoded)
    tokens = mx.array([tokens + [pad_id] * (length - len(tokens)) for tokens, _ in encoded], dtype=mx.int32)
    lengths = mx.array([len(tokens) for tokens, _ in encoded], dtype=mx.int32)
    labels = ([mx.array([labels[key] for _, labels in encoded]) for key in schema]
              if encoded[0][1] is not None else None)
    return tokens, lengths, labels


def decision_loss(model, tokens, lengths, labels):
    logits = model(tokens, lengths)
    return sum(nn.losses.cross_entropy(values, target, reduction="mean")
               for values, target in zip(logits, labels)) / len(logits)


def make_step(model, learning_rate):
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, decision_loss)

    def step(tokens, lengths, labels):
        loss, gradients = value_and_grad(model, tokens, lengths, labels)
        gradients, _ = optim.clip_grad_norm(gradients, 1.0)
        optimizer.update(model, gradients)
        # Synchronize both loss and updates; timing only loss would undercount work.
        mx.eval(loss, model.parameters(), optimizer.state)
        return float(loss.item())
    return step


def fit(model, tokenizer, rows, mode="full", epochs=3, batch_size=4,
        learning_rate=2e-5, max_length=128, seed=42, callback=None):
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    encoded = encode_rows(model, tokenizer, rows, max_length)
    model.set_train_mode(mode)
    model.temperatures = [1.0] * len(model.heads)
    model.calibration = "unfitted"
    step = make_step(model, learning_rate)
    rng, losses, start = random.Random(seed), [], time.perf_counter()
    for epoch in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [encoded[i] for i in order[offset:offset + batch_size]]
            loss = step(*batchify(batch, model.schema, tokenizer.eos_token_id))
            if not math.isfinite(loss):
                raise ValueError("non-finite training loss; reduce learning_rate")
            losses.append(loss)
            if callback:
                callback(len(losses), epoch + 1, loss)
    return {"steps": len(losses), "seconds": time.perf_counter() - start,
            "first_loss": losses[0], "last_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses)}


def logits_for_rows(model, tokenizer, rows, max_length=128, batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoded = encode_rows(model, tokenizer, rows, max_length)
    model.eval()
    results = []
    for start in range(0, len(encoded), batch_size):
        tokens, lengths, labels = batchify(encoded[start:start + batch_size], model.schema, tokenizer.eos_token_id)
        outputs = model(tokens, lengths)
        mx.eval(outputs)
        values = [output.tolist() for output in outputs]
        for i in range(tokens.shape[0]):
            results.append([head[i] for head in values])
    return results


def evaluate(model, tokenizer, rows, max_length=128):
    logits = logits_for_rows(model, tokenizer, rows, max_length)
    report = {key: {"accuracy": 0.0, "nll": 0.0, "brier": 0.0} for key in model.schema}
    for row, outputs in zip(rows, logits):
        expected = targets(model.schema, row["labels"])
        for i, key in enumerate(model.schema):
            scaled = [v / model.temperatures[i] for v in outputs[i]]
            probabilities = softmax(scaled)
            report[key]["nll"] += cross_entropy(scaled, expected[key])
            report[key]["brier"] += sum((p - y) ** 2 for p, y in zip(probabilities, expected[key]))
            report[key]["accuracy"] += float(max(range(len(probabilities)), key=probabilities.__getitem__) ==
                                             max(range(len(expected[key])), key=expected[key].__getitem__))
    return {"rows": len(rows), "questions": {key: {k: v / len(rows) for k, v in metrics.items()}
                                              for key, metrics in report.items()}}


def calibrate(model, tokenizer, rows, max_length=128):
    logits = logits_for_rows(model, tokenizer, rows, max_length)
    expected = [targets(model.schema, row["labels"]) for row in rows]
    for i, key in enumerate(model.schema):
        def objective(temperature):
            return sum(cross_entropy([v / temperature for v in out[i]], target[key])
                       for out, target in zip(logits, expected))
        model.temperatures[i] = min((1.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 10.0), key=objective)
    model.calibration = "temperature_fitted"
    return dict(zip(model.schema, model.temperatures))


def predict(model, tokenizer, text, max_length=128):
    encoded = encode_rows(model, tokenizer, [{"text": text}], max_length, labeled=False)
    tokens, lengths, _ = batchify(encoded, model.schema, tokenizer.eos_token_id)
    model.eval()
    logits = model(tokens, lengths)
    mx.eval(logits)
    return {"model": "gpt2-jevlite", "answers": {
        key: answer(question, softmax([v / model.temperatures[i] for v in logits[i][0].tolist()]))
        for i, (key, question) in enumerate(model.schema.items())},
        "metadata": {"source": model.source, "backend": "mlx", "parameters": parameter_count(model), "input_tokens": len(encoded[0][0]),
                     "calibration": model.calibration, "confidence_method": "one_minus_normalized_entropy"}}


def parameter_count(model, trainable=False):
    return sum(value.size for _, value in tree_flatten(model.trainable_parameters() if trainable else model.parameters()))
