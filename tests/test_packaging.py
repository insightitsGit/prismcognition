from __future__ import annotations

from pathlib import Path

import prismcognition

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.1.0"' in text
    assert prismcognition.__version__ == "2.1.0"


def test_license_and_typed_marker_exist():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert (ROOT / "prismcognition" / "py.typed").is_file()


def test_pyproject_has_publish_metadata():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "prismcognition"' in text
    assert 'license = "MIT"' in text
    assert "Homepage = " in text
    assert 'prismcognition = "prismcognition.cli:main"' in text
