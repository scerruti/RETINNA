"""Smoke tests against the real services. Run with: pytest -m network"""

import pytest

from burnmap import perimeters

pytestmark = pytest.mark.network


def test_frap_finds_2024_park_fire():
    fires = perimeters.search("Park", 2024)
    park = max(fires, key=lambda f: f.acres)
    assert park.acres > 400_000 and park.start.isoformat().startswith("2024-07")
    full = perimeters.get(park.id)
    assert full.geometry.is_valid and full.geometry.bounds[0] < -121
