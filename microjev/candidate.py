"""Question-conditioned GPT-2 scalar scoring with runtime candidate sets.

Each candidate is encoded independently with its state and question. The shared
scalar head has no task, question-ID, candidate-index or cardinality parameters.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.gpt2 import GPT2Model, ModelArgs
from mlx_lm.models.cache import KVCache
from transformers import AutoTokenizer

from .autograd import cross_entropy, softmax
from .mlx_backend import GPT2Decision, gpt2_args, parameter_count
from .schema import answer, targets, validate_schema


class CandidateDecision(nn.Module):
    def __init__(self, args=None, backbone=None):
        super().__init__()
        self.args = args or gpt2_args()
        self.backbone = GPT2Model(self.args) if backbone is None else backbone
        # A shared bias would cancel in softmax; omit it.
        self.scorer = nn.Linear(self.args.n_embd, 1, bias=False)
        self.temperatures = {kind: 1.0 for kind in ("choice", "noul", "score")}
        self.calibration = "unfitted"
        self.calibration_scope = "none"
        self.source = "random"

    def __call__(self, tokens, lengths, cache=None):
        hidden = self.backbone(tokens, cache=cache)
        return self.scorer(hidden[mx.arange(tokens.shape[0]), lengths - 1]).squeeze(-1)

    @classmethod
    def from_pretrained(cls, source="openai-community/gpt2", decision_checkpoint=False):
        if decision_checkpoint:
            base, tokenizer = GPT2Decision.load(source)
        else:
            base, tokenizer = GPT2Decision.from_pretrained({"unused": {"type": "noul"}}, source)
        model = cls(base.args, base.backbone)
        model.source = str(source)
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
        return model, tokenizer

    def save(self, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_weights(str(directory / "model.safetensors"))
        tokenizer.save_pretrained(str(directory))
        metadata = dict(format="microjev-candidate-mlx", version=1, args=asdict(self.args),
                        temperatures=self.temperatures, source=self.source,
                        calibration=self.calibration, calibration_scope=self.calibration_scope)
        (directory / "microjev.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory / "microjev.json").read_text(encoding="utf-8"))
        if meta.get("format") != "microjev-candidate-mlx" or meta.get("version") != 1:
            raise ValueError("unsupported candidate checkpoint")
        model = cls(ModelArgs(**meta["args"]))
        if set(meta["temperatures"]) != set(model.temperatures) or any(
                not math.isfinite(t) or t <= 0 for t in meta["temperatures"].values()):
            raise ValueError("invalid checkpoint temperatures")
        model.temperatures = meta["temperatures"]
        model.source = meta["source"]
        model.calibration = meta["calibration"]
        model.calibration_scope = meta["calibration_scope"]
        model.load_weights(str(directory / "model.safetensors"), strict=True)
        mx.eval(model.parameters())
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        return model, tokenizer


def candidate_texts(question):
    kind = question["type"]
    if kind == "choice":
        return [name + (": " + description if description else "")
                for name, description in question["criteria"].items()]
    if kind == "score":
        # Numeric indices are output positions, not learned semantic features.
        return list(question["criteria"])
    return ["False: The proposition is false.", "True: The proposition is true."]


@dataclass
class EncodedQuestion:
    key: str
    question: dict
    sequences: list
    target: object


def encode_requests(model, tokenizer, requests, max_length=192, labeled=True):
    if not requests:
        raise ValueError("requests must not be empty")
    if not 2 <= max_length <= model.args.n_positions:
        raise ValueError(f"max_length must be between 2 and {model.args.n_positions}")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires an EOS token")
    result = []
    for i, request in enumerate(requests, 1):
        if not isinstance(request, dict) or not isinstance(request.get("state"), str):
            raise ValueError(f"request {i}: expected a state string")
        schema = validate_schema(request.get("questions"))
        labels = targets(schema, request.get("labels")) if labeled else None
        for key, question in schema.items():
            sequences = []
            for candidate in candidate_texts(question):
                prompt = (f"State:\n{request['state']}\nQuestion:\n{question.get('instructions', '')}"
                          f"\nCandidate:\n{candidate}\nCompatibility:")
                sequence = tokenizer.encode(prompt, add_special_tokens=False) + [tokenizer.eos_token_id]
                if len(sequence) > max_length:
                    raise ValueError(f"request {i}, {key}: {len(sequence)} tokens exceeds max_length={max_length}; no silent truncation")
                sequences.append(sequence)
            result.append(EncodedQuestion(key, question, sequences, labels[key] if labeled else None))
    return result


def batchify(groups, pad_id):
    sequences = [seq for group in groups for seq in group.sequences]
    length = max(map(len, sequences))
    tokens = mx.array([seq + [pad_id] * (length - len(seq)) for seq in sequences], dtype=mx.int32)
    lengths = mx.array(list(map(len, sequences)), dtype=mx.int32)
    sizes = [len(group.sequences) for group in groups]
    labels = [mx.array(group.target) for group in groups] if groups[0].target is not None else None
    return tokens, lengths, sizes, labels


def decision_loss(model, tokens, lengths, sizes, labels):
    scores = model(tokens, lengths)
    start, losses = 0, []
    for size, target in zip(sizes, labels):
        losses.append(nn.losses.cross_entropy(scores[start:start + size][None, :], target[None, :], reduction="mean"))
        start += size
    return sum(losses) / len(losses)


def fit(model, tokenizer, requests, epochs=1, batch_size=4, learning_rate=1e-5,
        max_length=192, seed=42, callback=None, epoch_callback=None, label_smoothing=0.0):
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be finite and in [0, 1)")
    encoded = encode_requests(model, tokenizer, requests, max_length)
    if label_smoothing:
        for group in encoded:
            # Only training targets change; caller labels and evaluation stay hard.
            count = len(group.target)
            group.target = [(1 - label_smoothing) * p + label_smoothing / count for p in group.target]
    model.unfreeze()
    model.temperatures = {kind: 1.0 for kind in model.temperatures}
    model.calibration, model.calibration_scope = "unfitted", "none"
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    value_and_grad = nn.value_and_grad(model, decision_loss)
    rng, losses, started = random.Random(seed), [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(encoded)))
        rng.shuffle(order)
        epoch_start = len(losses)
        for offset in range(0, len(order), batch_size):
            groups = [encoded[i] for i in order[offset:offset + batch_size]]
            loss, gradients = value_and_grad(model, *batchify(groups, tokenizer.eos_token_id))
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            optimizer.update(model, gradients)
            mx.eval(loss, model.parameters(), optimizer.state)
            value = float(loss.item())
            if not math.isfinite(value):
                raise ValueError("non-finite training loss")
            losses.append(value)
            if callback:
                callback(len(losses), epoch, value)
        if epoch_callback:
            epoch_losses = losses[epoch_start:]
            epoch_callback(epoch, sum(epoch_losses) / len(epoch_losses))
    return {"steps": len(losses), "questions": len(encoded), "seconds": time.perf_counter() - started,
            "first_loss": losses[0], "last_loss": losses[-1], "mean_loss": sum(losses) / len(losses)}


def logits_for_requests(model, tokenizer, requests, max_length=192, batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoded = encode_requests(model, tokenizer, requests, max_length)
    model.eval()
    results = []
    for offset in range(0, len(encoded), batch_size):
        groups = encoded[offset:offset + batch_size]
        tokens, lengths, sizes, _ = batchify(groups, tokenizer.eos_token_id)
        scores = model(tokens, lengths)
        mx.eval(scores)
        values, start = scores.tolist(), 0
        for group, size in zip(groups, sizes):
            results.append((group, values[start:start + size]))
            start += size
    return results


def metrics_from_logits(model, results):
    metrics = {}
    for group, values in results:
        key = group.key
        if key not in metrics:
            metrics[key] = {"count": 0, "accuracy": 0.0, "nll": 0.0, "brier": 0.0}
        scaled = [v / model.temperatures[group.question["type"]] for v in values]
        probabilities, expected = softmax(scaled), group.target
        item = metrics[key]
        item["count"] += 1
        item["accuracy"] += float(max(range(len(values)), key=probabilities.__getitem__) ==
                                   max(range(len(values)), key=expected.__getitem__))
        item["nll"] += cross_entropy(scaled, expected)
        item["brier"] += sum((p - y) ** 2 for p, y in zip(probabilities, expected))
    return {"questions": {key: {k: v if k == "count" else v / item["count"] for k, v in item.items()}
                          for key, item in metrics.items()}}


def evaluate(model, tokenizer, requests, max_length=192, batch_size=8):
    return metrics_from_logits(model, logits_for_requests(model, tokenizer, requests, max_length, batch_size))


def calibrate(model, tokenizer, requests, max_length=192, scope="supplied calibration data"):
    results = logits_for_requests(model, tokenizer, requests, max_length)
    before = metrics_from_logits(model, results)
    for kind in model.temperatures:
        selected = [(group, values) for group, values in results if group.question["type"] == kind]
        if selected:
            def objective(temperature):
                return sum(cross_entropy([v / temperature for v in values], group.target) for group, values in selected)
            model.temperatures[kind] = min((1.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 10.0), key=objective)
    model.calibration, model.calibration_scope = "temperature_fitted_by_type", scope
    return {"temperatures": dict(model.temperatures), "before": before, "after": metrics_from_logits(model, results)}


def common_prefix_length(sequences):
    # Leave at least the final EOS to score, including for identical candidates.
    limit = min(map(len, sequences)) - 1
    for position in range(limit):
        token = sequences[0][position]
        if any(sequence[position] != token for sequence in sequences[1:]):
            return position
    return limit


def score_sequences(model, sequences, pad_id, batch_size=16, prefix_cache=False):
    """Batched scoring; caches are local to this call and never cross requests."""
    if batch_size < 1 or not sequences or any(not sequence for sequence in sequences):
        raise ValueError("nonempty sequences and a positive batch_size are required")
    prefix_length = common_prefix_length(sequences) if prefix_cache else 0
    shared_cache = None
    if prefix_length:
        shared_cache = [KVCache() for _ in range(model.args.n_layer)]
        model.backbone(mx.array([sequences[0][:prefix_length]], dtype=mx.int32), cache=shared_cache)
        mx.eval([cache.state for cache in shared_cache])
    values = []
    for offset in range(0, len(sequences), batch_size):
        batch = [seq[prefix_length:] for seq in sequences[offset:offset + batch_size]]
        chunk = EncodedQuestion("", {}, batch, None)
        tokens, lengths, _, _ = batchify([chunk], pad_id)
        cache = None
        if shared_cache is not None:
            cache = []
            for shared in shared_cache:
                branch = KVCache()
                branch.state = tuple(mx.repeat(value, len(batch), axis=0) for value in shared.state)
                cache.append(branch)
        scores = model(tokens, lengths, cache=cache)
        mx.eval(scores)
        values.extend(scores.tolist())
    return values, prefix_length


def predict(model, tokenizer, state, questions, max_length=192, candidate_batch_size=16, prefix_cache=False):
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    groups = encode_requests(model, tokenizer, [{"state": state, "questions": questions}], max_length, labeled=False)
    model.eval()
    sequences = [seq for group in groups for seq in group.sequences]
    all_values, prefix_length = score_sequences(model, sequences, tokenizer.eos_token_id, candidate_batch_size, prefix_cache)
    answers, offset = {}, 0
    for group in groups:
        values = all_values[offset:offset + len(group.sequences)]
        offset += len(group.sequences)
        scaled = [v / model.temperatures[group.question["type"]] for v in values]
        answers[group.key] = answer(group.question, softmax(scaled))
    logical_tokens = sum(map(len, sequences))
    tokens_processed = logical_tokens - prefix_length * (len(sequences) - 1)
    return {"model": "gpt2-jevlite-candidate", "answers": answers,
            "metadata": {"backend": "mlx", "source": model.source, "parameters": parameter_count(model),
                         "candidate_count": len(sequences), "tokens_processed": tokens_processed,
                         "logical_input_tokens": logical_tokens, "shared_prefix_tokens": prefix_length,
                         "prefix_cache": bool(prefix_length),
                         "calibration": model.calibration, "calibration_scope": model.calibration_scope,
                         "confidence_method": "one_minus_normalized_entropy",
                         "scoring": "independent_state_question_candidate_shared_scalar"}}
