"""Tests for the training pipeline that don't need a GPU or the real tokenizer.

A stand-in tokenizer (special tokens atomic, everything else one token per
character) exercises the exact code path in training/train.py.
"""
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

import export_gguf  # noqa: E402
import train  # noqa: E402

from fincopilot.datagen.generate import Generator  # noqa: E402

_SPECIAL = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in _SPECIAL) + ")")


class FakeTokenizer:
    """Special tokens -> ids 0..3, every other character -> 1000 + ord(c)."""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in _SPLIT.split(text):
            if part in _SPECIAL:
                ids.append(_SPECIAL.index(part))
            else:
                ids.extend(1000 + ord(c) for c in part)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(_SPECIAL[i] if i < 4 else chr(i - 1000) for i in ids)


class TestTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = Generator(7).build_train(scale=0.05)
        cls.tok = FakeTokenizer()

    def test_tokenize_all_and_mask(self):
        feats, dropped = train.tokenize_all(self.rows, self.tok, max_len=20_000)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(feats), len(self.rows))
        for row, feat in zip(self.rows, feats):
            self.assertEqual(len(feat["input_ids"]), len(feat["labels"]))
            learned = self.tok.decode([t for t in feat["labels"] if t != -100])
            expected = "".join(m["content"].strip() + "<|eot_id|>" for m in row["messages"]
                               if m["role"] == "assistant" and m.get("train", True))
            self.assertEqual(learned, expected, row["id"])
            # Nothing from the system prompt, user or tool messages is ever learned.
            self.assertNotIn("You are Financial Copilot", learned)

    def test_over_length_examples_are_dropped(self):
        feats, dropped = train.tokenize_all(self.rows, self.tok, max_len=500)
        self.assertEqual(len(feats) + dropped, len(self.rows))
        self.assertGreater(dropped, 0)

    def test_pad_batch(self):
        feats, _ = train.tokenize_all(self.rows[:3], self.tok, max_len=20_000)
        batch = train.pad_batch(feats, pad_id=9)
        width = max(len(f["input_ids"]) for f in feats)
        for key in ("input_ids", "attention_mask", "labels"):
            self.assertTrue(all(len(r) == width for r in batch[key]))
        for f, ids, mask, labels in zip(feats, batch["input_ids"], batch["attention_mask"], batch["labels"]):
            n = len(f["input_ids"])
            self.assertEqual(ids[n:], [9] * (width - n))
            self.assertEqual(mask[n:], [0] * (width - n))
            self.assertEqual(labels[n:], [-100] * (width - n))

    def test_describe_and_preview_run(self):
        feats, _ = train.tokenize_all(self.rows[:20], self.tok, max_len=20_000)
        stats = train.describe(feats, self.tok, "unit-test")
        self.assertGreater(stats["trained_tokens"], 0)
        self.assertLess(stats["trained_tokens"], stats["tokens"])

    def test_rejects_non_llama_tokenizer(self):
        class Plain(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1000 + ord(c) for c in text]}
        with self.assertRaises(SystemExit):
            train.tokenize_all(self.rows[:1], Plain(), max_len=20_000)

    def test_modelfile(self):
        with tempfile.TemporaryDirectory() as d:
            gguf = os.path.join(d, "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf")
            open(gguf, "w", encoding="utf-8").close()
            with open(export_gguf.write_modelfile(gguf), encoding="utf-8") as f:
                text = f.read()
        self.assertIn("FROM ./Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf", text)
        self.assertIn("{{ if .System }}", text)
        self.assertIn('PARAMETER stop "<|eot_id|>"', text)
        self.assertIn("compound_interest(principal", text)


if __name__ == "__main__":
    unittest.main()
