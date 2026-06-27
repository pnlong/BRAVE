import tempfile
import unittest
from pathlib import Path

import yaml

from rave.preprocess_metadata import read_stored_sec_from_metadata


class PreprocessStoredLimitTest(unittest.TestCase):

    def test_read_stored_sec_from_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "metadata.yaml"
            meta.write_text(yaml.safe_dump({"n_seconds": 123.45}))
            self.assertAlmostEqual(read_stored_sec_from_metadata(tmp), 123.45)

    def test_missing_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_stored_sec_from_metadata(tmp)


if __name__ == "__main__":
    unittest.main()
