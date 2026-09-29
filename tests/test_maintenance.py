"""Storage reset tests runnable without pandas or scikit-learn."""
import json
from pathlib import Path
import tempfile
import unittest

from filelock import FileLock, Timeout
from pipeline_maintenance import reset_generated_state


class ResetTests(unittest.TestCase):
    def test_reset_removes_versions_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            original = workspace / 'churn_model.pkl'
            original.write_bytes(b'original')
            root = workspace / 'state'
            for kind in ('runs', 'datasets'):
                folder = root / kind / 'example'
                folder.mkdir(parents=True)
                (folder / 'metadata.json').write_text('{}')
                (folder / 'artifact').write_bytes(b'dummy')
            for name in ('active.json', 'dataset.json', 'holdout.json', 'rollback_old.json'):
                (root / name).write_text('{"version":"example"}')
            result = reset_generated_state(root)
            self.assertEqual(result, {'runs': 1, 'datasets': 1})
            for name in ('active.json', 'dataset.json'):
                self.assertIsNone(json.loads((root / name).read_text())['version'])
            for name in ('runs', 'datasets', 'holdout.json', 'rollback_old.json'):
                self.assertFalse((root / name).exists())
            self.assertEqual(original.read_bytes(), b'original')
            self.assertEqual(reset_generated_state(root), {'runs': 0, 'datasets': 0})

    def test_reset_does_not_interrupt_a_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'dataset.json').write_text('{"version":"keep"}')
            with FileLock(str(root / '.write.lock')):
                with self.assertRaises(Timeout):
                    reset_generated_state(root)
            self.assertEqual(json.loads((root / 'dataset.json').read_text())['version'], 'keep')


if __name__ == '__main__':
    unittest.main()
