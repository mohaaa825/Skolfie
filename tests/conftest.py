import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from skolfie.features.stylometric import FeatureExtractor
from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a fresh module (``scripts/`` is not a package).

    The module is registered in ``sys.modules`` because dataclasses look their
    module up there while the script is executed.
    """
    module_name = f"skolfie_script_{name}"
    spec = importlib.util.spec_from_file_location(module_name,
                                                  REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture(scope="session")
def seed_texts() -> dict[str, list[str]]:
    """Synthetic seed texts (a test fixture only, not training data)."""
    return {"human": list(HUMAN_TEXTS), "ai": list(AI_TEXTS)}
