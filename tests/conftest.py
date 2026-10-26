import pytest

from skolfie.features.stylometric import FeatureExtractor
from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS


@pytest.fixture(scope="session")
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture(scope="session")
def seed_texts() -> dict[str, list[str]]:
    """Synthetic seed texts (a test fixture only, not training data)."""
    return {"human": list(HUMAN_TEXTS), "ai": list(AI_TEXTS)}
