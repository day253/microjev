"""A tiny causal transformer with fixed typed decision heads."""

import json
import math
import random
from pathlib import Path

from .autograd import Value, cross_entropy, number, softmax
from .schema import answer, cardinality, targets, validate_schema


def linear(vector, weights):
    return [sum(a * b for a, b in zip(vector, row)) for row in weights]


def rmsnorm(vector):
    scale = (sum(x * x for x in vector) / len(vector) + 1e-5) ** -0.5
    return [x * scale for x in vector]


class MicroJev:
    """Fixed schemas are learned from labels; instruction strings are metadata.

    UNK=0, BOS=1, EOS=2. EOS attends to the entire input. Every decision head
    reads the same final hidden state; no output token is generated.
    """

    def __init__(self, schema, chars, n_embd=16, n_head=4, n_layer=1,
                 block_size=64, seed=42):
        for key, value in {"n_embd": n_embd, "n_head": n_head,
                           "n_layer": n_layer, "block_size": block_size}.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if n_embd % n_head or block_size < 2:
            raise ValueError("n_embd must be divisible by n_head; block_size must be >= 2")
        if not isinstance(chars, (list, tuple)) or any(not isinstance(c, str) or len(c) != 1 for c in chars):
            raise ValueError("chars must be a list of single-character strings")
        if len(set(chars)) != len(chars):
            raise ValueError("chars must not contain duplicates")
        self.schema = validate_schema(schema)
        self.chars = list(chars)
        self.token_ids = {char: i + 3 for i, char in enumerate(chars)}
        self.config = dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                           block_size=block_size, seed=seed)
        self.temperatures = {key: 1.0 for key in self.schema}
        self.calibration = "unfitted"
        rng = random.Random(seed)

        def matrix(rows, columns):
            return [[Value(rng.gauss(0, 0.08)) for _ in range(columns)] for _ in range(rows)]

        self.weights = {
            "token": matrix(len(chars) + 3, n_embd),
            "position": matrix(block_size, n_embd),
        }
        for layer in range(n_layer):
            for name in ("q", "k", "v", "o"):
                self.weights[f"{layer}.{name}"] = matrix(n_embd, n_embd)
            self.weights[f"{layer}.up"] = matrix(4 * n_embd, n_embd)
            self.weights[f"{layer}.down"] = matrix(n_embd, 4 * n_embd)
        for i, question in enumerate(self.schema.values()):
            self.weights[f"head.{i}"] = matrix(cardinality(question), n_embd)
        self.params = [p for matrix_ in self.weights.values() for row in matrix_ for p in row]

    @classmethod
    def from_rows(cls, schema, rows, **config):
        if not rows:
            raise ValueError("training data must not be empty")
        texts = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("each row must contain a string text and object labels")
            texts.append(row["text"])
        model = cls(schema, sorted(set("".join(texts))), **config)
        model.validate_rows(rows)
        return model

    def tokenize(self, text):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        limit = self.config["block_size"] - 2
        if len(text) > limit:
            raise ValueError(f"input has {len(text)} characters; maximum is {limit} (no truncation)")
        return [1] + [self.token_ids.get(char, 0) for char in text] + [2]

    def validate_rows(self, rows):
        if not rows:
            raise ValueError("dataset must not be empty")
        for i, row in enumerate(rows, 1):
            try:
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                self.tokenize(row.get("text"))
                targets(self.schema, row.get("labels"))
            except ValueError as error:
                raise ValueError(f"row {i}: {error}") from error

    def logits(self, text, training=False):
        tokens = self.tokenize(text)
        weights = self.weights if training else self.numeric_weights()
        width, heads, layers = (self.config[k] for k in ("n_embd", "n_head", "n_layer"))
        head_width = width // heads
        keys, values = [[] for _ in range(layers)], [[] for _ in range(layers)]
        for position, token in enumerate(tokens):
            hidden = rmsnorm([a + b for a, b in zip(weights["token"][token], weights["position"][position])])
            for layer in range(layers):
                normalized = rmsnorm(hidden)
                query = linear(normalized, weights[f"{layer}.q"])
                keys[layer].append(linear(normalized, weights[f"{layer}.k"]))
                values[layer].append(linear(normalized, weights[f"{layer}.v"]))
                attended = []
                for head in range(heads):
                    start, end = head * head_width, (head + 1) * head_width
                    scores = [sum(query[j] * key[j] for j in range(start, end)) / math.sqrt(head_width)
                              for key in keys[layer]]
                    attention = softmax(scores)
                    attended.extend(sum(p * value[j] for p, value in zip(attention, values[layer]))
                                    for j in range(start, end))
                hidden = [a + b for a, b in zip(hidden, linear(attended, weights[f"{layer}.o"]))]
                expanded = linear(rmsnorm(hidden), weights[f"{layer}.up"])
                activated = [x.relu() if isinstance(x, Value) else max(0.0, x) for x in expanded]
                hidden = [a + b for a, b in zip(hidden, linear(activated, weights[f"{layer}.down"]))]
        hidden = rmsnorm(hidden)
        return {key: linear(hidden, weights[f"head.{i}"]) for i, key in enumerate(self.schema)}

    def loss(self, row):
        logits = self.logits(row["text"], training=True)
        expected = targets(self.schema, row["labels"])
        return sum(cross_entropy(logits[key], expected[key]) for key in self.schema) / len(self.schema)

    def fit(self, rows, steps=300, learning_rate=0.01, seed=42, callback=None):
        self.validate_rows(rows)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        # A new fit starts a new Adam schedule; checkpoints deliberately omit optimizer state.
        self.temperatures = {key: 1.0 for key in self.schema}
        self.calibration = "unfitted"
        first, second = [0.0] * len(self.params), [0.0] * len(self.params)
        rng, order, losses = random.Random(seed), list(range(len(rows))), []
        for step in range(steps):
            if step % len(rows) == 0:
                rng.shuffle(order)
            loss = self.loss(rows[order[step % len(rows)]])
            loss.backward()
            if not math.isfinite(loss.data) or any(not math.isfinite(p.grad) for p in self.params):
                raise ValueError("non-finite loss or gradients; reduce the learning rate")
            norm = math.sqrt(sum(p.grad * p.grad for p in self.params))
            clip = min(1.0, 1.0 / max(norm, 1e-12))
            rate = learning_rate * (1.0 - step / steps)
            for i, p in enumerate(self.params):
                gradient = p.grad * clip
                first[i] = 0.9 * first[i] + 0.1 * gradient
                second[i] = 0.999 * second[i] + 0.001 * gradient * gradient
                m = first[i] / (1.0 - 0.9 ** (step + 1))
                v = second[i] / (1.0 - 0.999 ** (step + 1))
                p.data -= rate * m / (math.sqrt(v) + 1e-8)
                p.grad = 0.0
            losses.append(loss.data)
            if callback:
                callback(step + 1, loss.data)
        return losses

    def predict(self, text):
        logits = self.logits(text)
        return {
            "model": "microjev",
            "answers": {key: answer(question, softmax([v / self.temperatures[key] for v in logits[key]]))
                        for key, question in self.schema.items()},
            "metadata": {
                "confidence_method": "one_minus_normalized_entropy",
                "calibration": self.calibration,
                "unknown_characters": sum(c not in self.token_ids for c in text),
            },
        }

    def evaluate(self, rows):
        self.validate_rows(rows)
        totals = {key: {"nll": 0.0, "brier": 0.0, "accuracy": 0.0} for key in self.schema}
        for row in rows:
            logits, expected = self.logits(row["text"]), targets(self.schema, row["labels"])
            for key in self.schema:
                scaled = [v / self.temperatures[key] for v in logits[key]]
                probabilities = softmax(scaled)
                totals[key]["nll"] += cross_entropy(scaled, expected[key])
                totals[key]["brier"] += sum((p - y) ** 2 for p, y in zip(probabilities, expected[key]))
                predicted = max(range(len(probabilities)), key=lambda i: probabilities[i])
                actual = max(range(len(expected[key])), key=lambda i: expected[key][i])
                totals[key]["accuracy"] += float(predicted == actual)
        return {"rows": len(rows), "questions": {key: {metric: value / len(rows) for metric, value in stats.items()}
                                                 for key, stats in totals.items()}}

    def calibrate(self, rows):
        """Fit per-head temperatures on validation data; this does not certify calibration."""
        self.validate_rows(rows)
        cached = [(self.logits(row["text"]), targets(self.schema, row["labels"])) for row in rows]
        grid = (1.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0, 10.0)
        for key in self.schema:
            def objective(temperature):
                return sum(cross_entropy([v / temperature for v in logits[key]], expected[key])
                           for logits, expected in cached)
            self.temperatures[key] = min(grid, key=objective)
        self.calibration = "temperature_fitted"
        return dict(self.temperatures)

    def numeric_weights(self):
        return {key: [[number(v) for v in row] for row in matrix] for key, matrix in self.weights.items()}

    def save(self, path):
        payload = dict(format="microjev", version=1, schema=self.schema, chars=self.chars,
                       config=self.config, weights=self.numeric_weights(),
                       temperatures=self.temperatures, calibration=self.calibration)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != "microjev" or payload.get("version") != 1:
            raise ValueError("unsupported checkpoint format/version")
        try:
            model = cls(payload["schema"], payload["chars"], **payload["config"])
            incoming = payload["weights"]
            if not isinstance(incoming, dict) or incoming.keys() != model.weights.keys():
                raise ValueError("checkpoint weight names do not match architecture")
            for key, matrix in model.weights.items():
                saved = incoming[key]
                if not isinstance(saved, list) or len(saved) != len(matrix):
                    raise ValueError(f"{key}: checkpoint matrix shape mismatch")
                for row, numbers in zip(matrix, saved):
                    if not isinstance(numbers, list) or len(numbers) != len(row):
                        raise ValueError(f"{key}: checkpoint row shape mismatch")
                    for p, value in zip(row, numbers):
                        if type(value) not in (int, float) or not math.isfinite(value):
                            raise ValueError(f"{key}: weights must be finite numbers")
                        p.data = float(value)
            temperatures = payload["temperatures"]
            if not isinstance(temperatures, dict) or temperatures.keys() != model.schema.keys():
                raise ValueError("checkpoint temperatures must match question IDs")
            if any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in temperatures.values()):
                raise ValueError("checkpoint temperatures must be finite and positive")
            if payload["calibration"] not in ("unfitted", "temperature_fitted"):
                raise ValueError("invalid calibration status")
            model.temperatures = dict(temperatures)
            model.calibration = payload["calibration"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"malformed checkpoint: {error}") from error
        return model
