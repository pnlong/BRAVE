import tempfile
import unittest
from pathlib import Path

from rave.core import (
    find_wandb_run_id,
    write_wandb_run_id,
    _resolve_wandb_run_id,
)


class WandbRunIdTest(unittest.TestCase):

    def test_write_and_read_id_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_wandb_run_id(tmp, "abc123")
            self.assertEqual(find_wandb_run_id(tmp), "abc123")

    def test_parse_from_wandb_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            wandb_dir = Path(tmp) / "wandb"
            wandb_dir.mkdir()
            (wandb_dir / "run-20260627_010702-k3oozmv3").mkdir()
            self.assertEqual(find_wandb_run_id(tmp), "k3oozmv3")

    def test_resolve_wandb_run_id_from_callable(self):
        class _Run:
            def id(self):
                return "abc123"

        self.assertEqual(_resolve_wandb_run_id(_Run()), "abc123")


if __name__ == "__main__":
    unittest.main()
