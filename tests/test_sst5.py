import unittest
from microjev.sst5 import SCHEMA, convert
from microjev.schema import targets


class SST5Tests(unittest.TestCase):
    def test_all_five_labels_have_consistent_typed_targets(self):
        for label in range(5):
            converted = convert({"text": "review", "label": label})
            labels = converted["labels"]
            self.assertEqual(labels["rating"], label)
            self.assertEqual(labels["positive"], label >= 3)
            self.assertEqual(labels["sentiment"], "negative" if label < 2 else "neutral" if label == 2 else "positive")
            expected = targets(SCHEMA, labels)
            self.assertEqual(expected["rating"][label], 1.0)

    def test_invalid_labels_rejected(self):
        for label in (-1, 5, True, "4"):
            with self.assertRaises(ValueError):
                convert({"text": "review", "label": label})
