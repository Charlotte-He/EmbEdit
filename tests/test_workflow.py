"""Exercise save/reload and row independence without downloading a model."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from embedit.cli import main
from test_editing import TinyEncoder, TinyTokenizer


class SavedEncoder(TinyEncoder):
    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        # JSON tensors are sufficient for this tiny fixture; production uses safetensors.
        (path / "fixture.json").write_text(json.dumps({key: value.tolist() for key, value in self.state_dict().items()}))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        model = cls()
        model.load_state_dict({key: torch.tensor(value) for key, value in json.loads((path / "fixture.json").read_text()).items()})
        return model


class WorkflowTests(unittest.TestCase):
    def test_saved_encoders_reload_and_independent_edits_restore_rows(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            csv = root / "input.csv"
            csv.write_text("old,new\nrose,blue rose\ntree,green tree\n")
            for model, mode in [("sd14", "independent"), ("sd14", "joint"), ("sdxl", "sequential")]:
                with self.subTest(model=model, mode=mode):
                    torch.manual_seed(11)
                    encoder = SavedEncoder()
                    original = encoder.embedding.weight.detach().clone()
                    pipe = SimpleNamespace(text_encoder=encoder, tokenizer=TinyTokenizer())
                    if model == "sdxl":
                        pipe.text_encoder_2 = SavedEncoder()
                        pipe.tokenizer_2 = TinyTokenizer()
                    output = root / f"{model}_{mode}"
                    with patch("embedit.pipelines.load_pipeline", return_value=(pipe, None)), \
                         patch("embedit.cli.importlib.metadata.version", return_value="test-fixture"), \
                         contextlib.redirect_stdout(io.StringIO()):
                        main(["--model", model, "--mode", mode, "--csv", str(csv),
                              "--output", str(output), "--iterations", "3", "--skip-generation", "--device", "cpu"])
                    saved = output / ("edit_0001" if mode == "independent" else "edited")
                    self.assertTrue((saved / "edit.json").is_file())
                    reloaded = SavedEncoder.from_pretrained(saved / "text_encoder")
                    torch.testing.assert_close(reloaded.embedding.weight, pipe.text_encoder.embedding.weight)
                    if mode == "independent":
                        # The first row's rose edit must not leak into the tree edit.
                        torch.testing.assert_close(reloaded.embedding.weight[2], original[2], rtol=0, atol=0)
                    if model == "sdxl":
                        self.assertTrue((saved / "text_encoder_2" / "fixture.json").is_file())

                    pipe.text_encoder = SavedEncoder()
                    seen = []

                    def fake_generate(current, *args):
                        seen.append(current.text_encoder.embedding.weight.detach().clone())
                        return SimpleNamespace(save=lambda path: path.write_bytes(b"fixture"))

                    with patch("embedit.pipelines.load_pipeline", return_value=(pipe, None)), \
                         patch("embedit.pipelines.generate", side_effect=fake_generate), \
                         patch("embedit.cli.importlib.metadata.version", return_value="test-fixture"):
                        main(["--model", model, "--load-edit", str(saved), "--prompt", "rose",
                              "--output", str(root / f"reload_{model}_{mode}"), "--device", "cpu"])
                    torch.testing.assert_close(seen[0], reloaded.embedding.weight)


if __name__ == "__main__":
    unittest.main()
