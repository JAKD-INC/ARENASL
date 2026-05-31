from collections import defaultdict
from pathlib import Path
import numpy as np


def load_templates(directory) -> dict[str, list[np.ndarray]]:
    """Load reference templates grouped by gloss.

    Expects files named ``<GLOSS>__<index>.npy`` in `directory`.

    Raises:
        FileNotFoundError: if the directory does not exist.
        ValueError: if it contains no template files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"template directory not found: {directory}")

    templates: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(directory.glob("*__*.npy")):
        gloss = path.stem.rsplit("__", 1)[0]
        array = np.load(path)
        if array.shape[0] == 0:
            raise ValueError(f"template has 0 frames: {path}")
        templates[gloss].append(array)

    if not templates:
        raise ValueError(f"no templates (*__*.npy) found in {directory}")

    for gloss, exemplars in templates.items():
        shapes = {exemplar.shape[1:] for exemplar in exemplars}
        if len(shapes) > 1:
            raise ValueError(
                f"inconsistent feature shape for gloss {gloss}: {sorted(shapes)}"
            )
    return dict(templates)
