import json
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def strict_basic():
    return json.loads((FIXTURES_DIR / "strict_basic.json").read_text())


@pytest.fixture
def strict_mismatch():
    return json.loads((FIXTURES_DIR / "strict_mismatch.json").read_text())


@pytest.fixture
def strict_rounding():
    return json.loads((FIXTURES_DIR / "strict_rounding.json").read_text())
