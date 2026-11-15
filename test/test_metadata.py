"""Tests for the setuptools-scm based versioning scheme."""
from pathlib import Path

import pytest

import pysndlib

REPO_ROOT = Path(__file__).parent.parent


def test_version_string_is_derived():
    v = pysndlib.__version__
    assert v and v != "0.0.0.dev0"
    # setuptools-scm formats: exact ("0.8.0") or dev ("0.8.1.dev3+g1a2b3c4")
    assert v[0].isdigit()


def test_changelog_exists_and_has_version_section():
    if not (REPO_ROOT / "CHANGELOG.md").exists():
        pytest.skip("CHANGELOG.md is a local-only file, not present in CI checkouts")
    text = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert text.startswith("# Changelog")
    assert "## 0.8.0" in text


def test_release_doc_exists_with_tagging_instructions():
    if not (REPO_ROOT / "RELEASE.md").exists():
        pytest.skip("RELEASE.md is a local-only file, not present in CI checkouts")
    release_doc = (REPO_ROOT / "RELEASE.md").read_text()
    assert "git push origin v" in release_doc
    assert "Choose a tag" in release_doc  # GitHub web UI instructions


def test_pyproject_has_no_static_version():
    import re
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert "setuptools-scm" in text
    assert not re.search(r"^version\s*=", text, re.M)  # no static version anywhere
