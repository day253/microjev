import importlib.util
import tempfile
import unittest
from dataclasses import replace


HAS_MLX = importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None


@unittest.skipUnless(HAS_MLX, "optional MLX dependencies are not installed")
class MLXTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from microjev.mlx_backend import GPT2Decision, gpt2_args
        from microjev.demo import SCHEMA
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        mx.random.seed(5)
        core = Tokenizer(models.WordLevel({"<eos>": 0, "<unk>": 1, "a": 2, "b": 3}, unk_token="<unk>"))
        core.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", unk_token="<unk>")
        args = replace(gpt2_args(), n_embd=8, n_head=2, n_layer=2, vocab_size=4, n_positions=16, n_ctx=16)
        self.model = GPT2Decision(SCHEMA, args)
        self.rows = [{"text": "a", "labels": {"intent": "refund", "urgent": True, "severity": 2}},
                     {"text": "a b b", "labels": {"intent": "bug", "urgent": False, "severity": 0}}]

    def test_padding_does_not_change_predictions(self):
        from microjev.mlx_backend import logits_for_rows
        alone = logits_for_rows(self.model, self.tokenizer, self.rows[:1], max_length=16)[0]
        padded = logits_for_rows(self.model, self.tokenizer, self.rows, max_length=16)[0]
        for first, second in zip(alone, padded):
            for a, b in zip(first, second):
                self.assertAlmostEqual(a, b, delta=1e-5)

    def test_heads_mode_freezes_backbone_full_mode_updates_it(self):
        import mlx.core as mx
        from microjev.mlx_backend import fit
        original = mx.array(self.model.backbone.wte.weight)
        mx.eval(original)
        fit(self.model, self.tokenizer, self.rows, mode="heads", epochs=1, learning_rate=0.001, max_length=16)
        self.assertTrue(mx.array_equal(original, self.model.backbone.wte.weight).item())
        fit(self.model, self.tokenizer, self.rows, mode="full", epochs=1, learning_rate=0.001, max_length=16)
        self.assertFalse(mx.array_equal(original, self.model.backbone.wte.weight).item())

    def test_checkpoint_round_trip_is_offline_and_preserves_answers(self):
        from microjev.mlx_backend import GPT2Decision, calibrate, predict
        calibrate(self.model, self.tokenizer, self.rows, max_length=16)
        expected = predict(self.model, self.tokenizer, "a b", max_length=16)
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory, self.tokenizer)
            loaded, tokenizer = GPT2Decision.load(directory)
            actual = predict(loaded, tokenizer, "a b", max_length=16)
        self.assertEqual(actual, expected)

    def test_long_inputs_are_rejected(self):
        from microjev.mlx_backend import encode_rows
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            encode_rows(self.model, self.tokenizer, self.rows, max_length=2)


if __name__ == "__main__":
    unittest.main()
