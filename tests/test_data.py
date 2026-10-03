import contextlib
import io
import unittest
from unittest.mock import mock_open, patch

from embedit.cli import main
from embedit.data import prompts_for, read_edits


class DataTests(unittest.TestCase):
    def read(self, text, **kwargs):
        with patch("builtins.open", mock_open(read_data=text)):
            return read_edits("user.csv", **kwargs)

    def test_minimal_csv_and_row_slice(self):
        rows = self.read("old,new\nrose,blue rose\ntree,green tree\n", begin=1)
        self.assertEqual(rows, [{"old": "tree", "new": "green tree"}])

    def test_malformed_csv_fails(self):
        for text in ["a,b\nrose,blue\n", "old,new\nrose,\n", "old,new\nrose,blue,extra\n"]:
            with self.assertRaises(ValueError):
                self.read(text)

    def test_invalid_range_fails(self):
        with self.assertRaises(ValueError):
            self.read("old,new\nrose,blue\n", end=2)

    def test_prompt_columns_are_not_inferred_from_filename(self):
        result = prompts_for({"old": "rose", "new": "blue rose", "ex1": "a rose", "positive1": "rose", "gt1": "target"})
        self.assertEqual([label for label, _ in result], ["base", "ex1", "positive1"])

    def test_help_exits_without_loading_models(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
