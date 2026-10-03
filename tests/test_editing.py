import contextlib
import copy
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from embedit.editing import edit_embeddings
from embedit.pipelines import encoders, generate


class TinyTokenizer:
    vocabulary = {"rose": 2, "blue": 3, "tree": 4, "green": 5}

    def encode(self, text, add_special_tokens=True):
        return [0] + [self.vocabulary[word] for word in text.split()] + [1]

    def __call__(self, texts, *, max_length, **kwargs):
        rows = [self.encode(text) for text in texts]
        return SimpleNamespace(input_ids=torch.tensor([row + [1] * (max_length - len(row)) for row in rows]))


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.projection = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(max_position_embeddings=77)
        self.text_model = SimpleNamespace(embeddings=SimpleNamespace(token_embedding=self.embedding))

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, ids):
        # Context mixing makes changes to source tokens affect the shared target.
        hidden = self.projection(self.embedding(ids)).cumsum(dim=1)
        return (hidden,)


class EditingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = TinyEncoder()
        self.tokenizer = TinyTokenizer()

    def load_reference(self, name):
        path = Path(__file__).resolve().parents[1] / "reference" / name
        spec = importlib.util.spec_from_file_location("reference_variant", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_source_rows_change_and_flags_are_restored(self):
        before = copy.deepcopy(self.encoder.state_dict())
        edit_embeddings(self.encoder, self.tokenizer, [("rose", "blue rose")], iterations=5)
        self.assertFalse(torch.equal(before["embedding.weight"][2], self.encoder.embedding.weight[2]))
        unchanged = [0, 1, 3, 4, 5, 6, 7]
        self.assertTrue(torch.equal(before["embedding.weight"][unchanged], self.encoder.embedding.weight[unchanged]))
        for key in ("projection.weight", "projection.bias"):
            self.assertTrue(torch.equal(before[key], self.encoder.state_dict()[key]))
        self.assertTrue(all(p.requires_grad for p in self.encoder.parameters()))
        self.assertTrue(self.encoder.training)

    def test_single_edit_matches_original_sd14_and_sdxl(self):
        for name, factor in [("sd14_mse.py", 0.35), ("sdxl_mse.py", 0.3)]:
            with self.subTest(name=name):
                reference = self.load_reference(name)
                original = copy.deepcopy(self.encoder)
                cleaned = copy.deepcopy(self.encoder)
                with contextlib.redirect_stdout(io.StringIO()):
                    reference.change_emd(original, self.tokenizer, {"old": "rose", "new": "blue rose"}, num_iterations=5)
                edit_embeddings(cleaned, self.tokenizer, [("rose", "blue rose")], iterations=5, factor=factor)
                torch.testing.assert_close(original.embedding.weight, cleaned.embedding.weight, rtol=0, atol=1e-7)

    def test_joint_edit_matches_original(self):
        original = copy.deepcopy(self.encoder)
        pairs = [("rose", "blue rose"), ("tree", "green tree")]
        reference = self.load_reference("sd14_joint_mse.py")
        with contextlib.redirect_stdout(io.StringIO()):
            reference.multi_change_emd(original, self.tokenizer, pairs, iters=5, mse_factor=0.2)
        edit_embeddings(self.encoder, self.tokenizer, pairs, iterations=5, joint=True, factor=0.2)
        torch.testing.assert_close(original.embedding.weight, self.encoder.embedding.weight, rtol=0, atol=1e-7)

    def test_early_stopping_retains_original_minimum_steps(self):
        result = edit_embeddings(self.encoder, self.tokenizer, [("rose", "blue rose")], iterations=8, factor=100)
        self.assertEqual(len(result["history"]), 2)

    def test_empty_and_overlong_edits_fail(self):
        for text in ["", " ".join(["rose"] * 80)]:
            with self.assertRaises(ValueError):
                edit_embeddings(self.encoder, self.tokenizer, [(text, "blue")])

    def test_invalid_hyperparameters_fail(self):
        for kwargs in [{"learning_rate": float("nan")}, {"iterations": 0}, {"factor": -1}]:
            with self.assertRaises(ValueError):
                edit_embeddings(self.encoder, self.tokenizer, [("rose", "blue")], **kwargs)

    def test_sdxl_second_encoder_uses_its_own_tokenizer(self):
        pipe = SimpleNamespace(text_encoder=object(), text_encoder_2=object(), tokenizer=object(), tokenizer_2=object())
        self.assertIs(encoders(pipe, "sdxl")[1][2], pipe.tokenizer_2)

    def test_seed_and_guidance_reach_base_and_refiner(self):
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=["image"])

        self.assertEqual(generate(fake, fake, "rose", 42, 20, 6.0, "cpu"), "image")
        for call in calls:
            self.assertEqual(call["generator"].initial_seed(), 42)
            self.assertEqual(call["guidance_scale"], 6.0)
        self.assertEqual(calls[0]["denoising_end"], calls[1]["denoising_start"])


if __name__ == "__main__":
    unittest.main()
