"""Reset local generated state without importing the machine-learning runtime."""
from pathlib import Path
import json
import os
import shutil
import uuid

from filelock import FileLock


def reset_generated_state(root):
    """Physically remove generated versions; leave bundled source files alone."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.write.lock'), timeout=1):
        targets = [root / name for name in ('runs', 'datasets', 'holdout.json')]
        targets += list(root.glob('rollback_*.json'))
        # Validate every target before changing pointers or deleting anything.
        for target in targets:
            if target.resolve().parent != root or target.is_symlink() or target.is_junction():
                raise ValueError(f'Unsafe storage path: {target}')
            if target.is_dir():
                for child in target.rglob('*'):
                    if child.is_symlink() or child.is_junction():
                        raise ValueError(f'Linked storage path cannot be reset: {child}')
        counts = {name: len(list((root / name).glob('*/metadata.json')))
                  for name in ('runs', 'datasets')}
        # Stop serving generated artifacts before removing them. An interrupted
        # cleanup can safely be retried; pointers already select the originals.
        for name in ('active.json', 'dataset.json'):
            temporary = root / f'{uuid.uuid4().hex}.tmp'
            try:
                with temporary.open('w', encoding='utf-8') as handle:
                    json.dump({'version': None}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, root / name)
            finally:
                temporary.unlink(missing_ok=True)
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        return counts
