import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from microjev import MicroJev
from microjev.autograd import Value, cross_entropy, softmax
from microjev.demo import SCHEMA, dataset
from microjev.schema import answer, targets, validate_schema


class AutogradTests(unittest.TestCase):
    def test_shared_graph_and_repeated_backward(self):
        x = Value(2)
        shared = x * x
        result = shared * shared + shared
        result.backward()
        self.assertAlmostEqual(x.grad, 36)
        result.backward()
        self.assertAlmostEqual(x.grad, 36)

    def test_deep_graph_without_recursion(self):
        x = Value(1)
        result = x
        for _ in range(2000):
            result = result + 1
        result.backward()
        self.assertEqual(x.grad, 1)

    def test_extreme_logits_are_finite(self):
        logits = [Value(1000), Value(-1000)]
        loss = cross_entropy(logits, [0.0, 1.0])
        loss.backward()
        self.assertEqual(loss.data, 2000)
        self.assertEqual([x.grad for x in logits], [1.0, -1.0])


class ModelTests(unittest.TestCase):
    def tiny_model(self):
        return MicroJev(SCHEMA, ["a", "b"], n_embd=4, n_head=2, block_size=8)

    def test_full_transformer_gradients_match_finite_differences(self):
        model = self.tiny_model()
        row = {"text": "ab", "labels": {"intent": "bug", "urgent": True, "severity": 2}}
        model.loss(row).backward()
        expected = targets(model.schema, row["labels"])

        def numeric_loss():
            logits = model.logits(row["text"])
            return sum(cross_entropy(logits[k], expected[k]) for k in expected) / len(expected)

        for name in ("token", "position", "0.q", "0.k", "0.v", "0.o", "0.up", "0.down", "head.0", "head.1", "head.2"):
            p = model.weights[name][1][1]
            analytic, original, epsilon = p.grad, p.data, 1e-5
            p.data = original + epsilon
            above = numeric_loss()
            p.data = original - epsilon
            below = numeric_loss()
            p.data = original
            self.assertAlmostEqual(analytic, (above - below) / (2 * epsilon), delta=2e-5, msg=name)

    def test_training_learns_a_separable_task(self):
        schema = {"label": {"type": "choice", "criteria": {"a": "", "b": ""}}}
        rows = [{"text": c, "labels": {"label": c}} for c in "ab"]
        model = MicroJev.from_rows(schema, rows, n_embd=4, n_head=1, block_size=4, seed=9)
        before = model.evaluate(rows)["questions"]["label"]["nll"]
        model.fit(rows, steps=70, learning_rate=0.03, seed=9)
        after = model.evaluate(rows)["questions"]["label"]
        self.assertLess(after["nll"], before * 0.3)
        self.assertEqual(after["accuracy"], 1.0)

    def test_inference_matches_training_forward(self):
        model = self.tiny_model()
        tracked, plain = model.logits("ab", training=True), model.logits("ab")
        for key in tracked:
            for x, y in zip(tracked[key], plain[key]):
                self.assertAlmostEqual(x.data, y, places=12)

    def test_round_trip_checkpoint_includes_calibration(self):
        model = self.tiny_model()
        model.temperatures["intent"] = 2.0
        model.calibration = "temperature_fitted"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = MicroJev.load(path)
            self.assertEqual(loaded.predict("ab"), model.predict("ab"))

    def test_calibration_cannot_worsen_validation_nll_from_default(self):
        model = self.tiny_model()
        rows = [{"text": "ab", "labels": {"intent": "bug", "urgent": False, "severity": 1}}]
        before = model.evaluate(rows)
        model.calibrate(rows)
        after = model.evaluate(rows)
        for key in model.schema:
            self.assertLessEqual(after["questions"][key]["nll"], before["questions"][key]["nll"] + 1e-12)

    def test_unknown_characters_and_length(self):
        model = self.tiny_model()
        self.assertEqual(model.predict("x")["metadata"]["unknown_characters"], 1)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            model.predict("a" * 7)

    def test_bad_checkpoint_fails(self):
        model = self.tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            payload = json.loads(path.read_text())
            for mutate in (
                    lambda p: p["weights"]["head.0"].pop(),
                    lambda p: p["temperatures"].update(intent=0),
                    lambda p: p["weights"]["token"][0].__setitem__(0, float("nan"))):
                broken = copy.deepcopy(payload)
                mutate(broken)
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    MicroJev.load(path)

    def test_invalid_labels_and_config(self):
        model = self.tiny_model()
        with self.assertRaises(ValueError):
            targets(model.schema, {"intent": "missing", "urgent": True, "severity": 1})
        with self.assertRaises(ValueError):
            targets(model.schema, {"intent": "bug", "urgent": float("nan"), "severity": 1})
        with self.assertRaises(ValueError):
            targets(model.schema, {"intent": "bug", "urgent": True, "severity": True})
        with self.assertRaises(ValueError):
            MicroJev(SCHEMA, [], n_embd=3, n_head=2)
        with self.assertRaises(ValueError):
            validate_schema({"bad": {"type": "score", "criteria": ["one"]}})

    def test_typed_answer_semantics(self):
        probabilities = [0.2, 0.3, 0.5]
        result = answer(SCHEMA["severity"], probabilities)
        self.assertAlmostEqual(result["score"], 1.3)
        self.assertEqual(answer(SCHEMA["urgent"], [0.2, 0.8]), {"type": "noul", "noul": 0.8})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1)

    def test_synthetic_splits_have_no_duplicate_texts(self):
        train, valid, test = (set(row["text"] for row in dataset(split)) for split in ("train", "validation", "test"))
        self.assertFalse(train & valid or train & test or valid & test)

    def test_cli_json_and_error_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.tiny_model().save(path)
            command = [sys.executable, "-m", "microjev", "predict", "--model", str(path), "--text"]
            success = subprocess.run(command + ["ab"], capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["model"], "microjev")
            failure = subprocess.run(command + ["a" * 20], capture_output=True, text=True)
            self.assertEqual(failure.returncode, 2)
            self.assertIn("maximum", failure.stderr)
            self.assertNotIn("Traceback", failure.stderr)


if __name__ == "__main__":
    unittest.main()
