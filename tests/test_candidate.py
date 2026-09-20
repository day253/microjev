import copy
import importlib.util
import math
import tempfile
import unittest
from dataclasses import replace

from microjev.candidate_sst5 import instruction_probes, requests_from_rows
from microjev.schema import targets
from microjev.sst5 import convert


class CandidateDataTests(unittest.TestCase):
    def test_augmented_targets_follow_changed_question_and_reversed_scale(self):
        rows = [convert({"text": "review", "label": label}) for label in range(5)] * 4
        requests = requests_from_rows(rows, augment=True)
        for i, request in enumerate(requests):
            target = rows[i]["labels"]["rating"]
            variant = i % 4
            self.assertEqual(request["labels"]["rating"], 4 - target if variant % 2 else target)
            self.assertEqual(request["labels"]["positive"], (target >= 3, target <= 1, target == 2, target >= 2)[variant])
            targets(request["questions"], request["labels"])
        # Source labels must not be mutated by augmentation.
        self.assertEqual(rows[1]["labels"]["rating"], 1)

    def test_instruction_probe_negation_is_complementary(self):
        rows = [convert({"text": "review", "label": label}) for label in range(5)]
        for request in instruction_probes(rows):
            labels = request["labels"]
            self.assertEqual(labels["not_positive"], not labels["positive_paraphrase"])
            targets(request["questions"], labels)


HAS_MLX = importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None


@unittest.skipUnless(HAS_MLX, "optional MLX dependencies are not installed")
class CandidateMLXTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from microjev.candidate import CandidateDecision
        from microjev.mlx_backend import gpt2_args
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        mx.random.seed(10)
        words = ["<eos>", "<unk>", "a", "b", "c", "State", "Question", "Candidate", "Compatibility",
                 ":", ".", "True", "False", "The", "proposition", "is", "true", "false", "red", "blue", "green"]
        core = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", unk_token="<unk>")
        args = replace(gpt2_args(), n_embd=16, n_head=2, n_layer=1, vocab_size=len(words), n_positions=64, n_ctx=64)
        self.model = CandidateDecision(args)
        self.schema = {"pick": {"type": "choice", "instructions": "a", "criteria": {"red": "a", "blue": "b", "green": "c"}},
                       "truth": {"type": "noul", "instructions": "b"},
                       "rating": {"type": "score", "instructions": "c", "criteria": ["a", "b"]}}
        self.requests = [{"state": "a", "questions": self.schema,
                          "labels": {"pick": "red", "truth": True, "rating": 1}},
                         {"state": "b b b", "questions": self.schema,
                          "labels": {"pick": "blue", "truth": False, "rating": 0}}]

    def test_candidate_permutation_question_id_and_reversed_score(self):
        from microjev.candidate import predict
        first = predict(self.model, self.tokenizer, "a", self.schema, max_length=64)
        changed = copy.deepcopy(self.schema)
        changed["renamed"] = changed.pop("pick")
        changed["renamed"]["criteria"] = dict(reversed(list(changed["renamed"]["criteria"].items())))
        changed["rating"]["criteria"].reverse()
        second = predict(self.model, self.tokenizer, "a", changed, max_length=64, candidate_batch_size=1)
        for name, p in first["answers"]["pick"]["probabilities"].items():
            self.assertAlmostEqual(p, second["answers"]["renamed"]["probabilities"][name], delta=1e-5)
        self.assertAlmostEqual(first["answers"]["rating"]["score"], 1 - second["answers"]["rating"]["score"], delta=1e-5)
        self.assertAlmostEqual(sum(first["answers"]["pick"]["probabilities"].values()), 1.0)

    def test_instructions_are_encoded_and_change_scores(self):
        from microjev.candidate import encode_requests, predict
        schema = copy.deepcopy(self.schema)
        schema["pick"]["instructions"] = "b b"
        before = encode_requests(self.model, self.tokenizer, self.requests[:1], max_length=64)
        after = encode_requests(self.model, self.tokenizer, [{**self.requests[0], "questions": schema}], max_length=64)
        self.assertNotEqual(before[0].sequences, after[0].sequences)
        a = predict(self.model, self.tokenizer, "a", self.schema, max_length=64)
        b = predict(self.model, self.tokenizer, "a", schema, max_length=64)
        self.assertNotEqual(a["answers"]["pick"], b["answers"]["pick"])

    def test_mixed_candidate_counts_loss_matches_manual_nll(self):
        from microjev.candidate import batchify, decision_loss, encode_requests
        from microjev.autograd import cross_entropy
        groups = encode_requests(self.model, self.tokenizer, self.requests, max_length=64)
        tokens, lengths, sizes, labels = batchify(groups, 0)
        scores = self.model(tokens, lengths).tolist()
        offset, total = 0, 0.0
        for group, size in zip(groups, sizes):
            total += cross_entropy(scores[offset:offset + size], group.target)
            offset += size
        self.assertAlmostEqual(float(decision_loss(self.model, tokens, lengths, sizes, labels).item()), total / len(groups), places=5)

    def test_padding_and_batch_size_do_not_change_logits(self):
        from microjev.candidate import logits_for_requests
        alone = logits_for_requests(self.model, self.tokenizer, self.requests[:1], max_length=64, batch_size=1)
        padded = logits_for_requests(self.model, self.tokenizer, self.requests, max_length=64, batch_size=6)
        for (_, a), (_, b) in zip(alone, padded):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, delta=1e-5)

    def test_training_calibration_and_offline_roundtrip(self):
        from microjev.candidate import CandidateDecision, calibrate, evaluate, fit, predict
        history = []
        report = fit(self.model, self.tokenizer, self.requests, epochs=2, batch_size=3,
                     learning_rate=0.001, max_length=64, epoch_callback=lambda epoch, loss: history.append((epoch, loss)))
        self.assertEqual([epoch for epoch, _ in history], [1, 2])
        self.assertTrue(math.isfinite(report["last_loss"]))
        before = evaluate(self.model, self.tokenizer, self.requests, max_length=64)
        calibration = calibrate(self.model, self.tokenizer, self.requests, max_length=64, scope="unit test")
        for key in self.schema:
            self.assertLessEqual(calibration["after"]["questions"][key]["nll"], before["questions"][key]["nll"] + 1e-8)
        expected = predict(self.model, self.tokenizer, "a", self.schema, max_length=64)
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory, self.tokenizer)
            loaded, tokenizer = CandidateDecision.load(directory)
            actual = predict(loaded, tokenizer, "a", self.schema, max_length=64)
        self.assertEqual(actual, expected)

    def test_rejects_long_state_and_invalid_schema(self):
        from microjev.candidate import predict
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            predict(self.model, self.tokenizer, "a " * 60, self.schema, max_length=64)
        with self.assertRaises(ValueError):
            predict(self.model, self.tokenizer, "a", {"invalid": {"type": "choice", "criteria": {"only": "a"}}}, max_length=64)
