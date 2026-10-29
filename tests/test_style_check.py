"""The style check's transforms do exactly what docs/decisions.md says."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("style_check", ROOT / "scripts" / "style_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


@pytest.mark.parametrize("text, a, b", [
    ("i i think", "i think", "i think"),
    ("it is it is good", "it is good", "it is good"),
    ("no no no no", "no", "no"),
    ("i went to the the shop", "i went to the shop", "i went to the shop"),
    ("yeah so i mean you know it is like good", "yeah so i mean you know it is like good", "it is good"),
    ("i like i like it", "i like it", "i it"),          # B is aggressive on purpose: "like" goes even as a verb
    ("okay okay right", "okay right", ""),
    ("the cat sat", "the cat sat", "the cat sat"),
])
def test_transforms(text, a, b):
    assert sc.transform(text, "official") == text
    assert sc.transform(text, "A") == a
    assert sc.transform(text, "B") == b
