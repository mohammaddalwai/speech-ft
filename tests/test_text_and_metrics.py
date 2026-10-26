"""Normalizer and WER checks. The normalizer tests download Whisper's tokenizer files once (needs internet)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import metrics, text  # noqa: E402


def test_training_text_lowercases_and_drops_tags_only():
    assert text.training_text("YEAH <LAUGH> THERE'S THERE'S  A <OVERLAP> UH CAT") == "yeah there's there's a uh cat"


def test_normalizer_makes_edacc_and_whisper_styles_equal():
    pairs = [
        ("MISTER QUILTER IS THE APOSTLE", "Mr. Quilter is the apostle."),   # case, punctuation, titles
        ("<LAUGH> I DON'T THINK SO", "I don't think so."),                   # tags, contractions
        ("UH SO MM HMM WE WENT", "So we went."),                             # fillers (Whisper's list)
        ("I HAVE TWENTY FIVE EUROS", "I have 25 euros."),                    # numbers
        ("HMMM UHH I MEAN AH YES", "I mean, yes."),                          # v2: more filler spellings
        ("ALRIGHT OKAY I CAN'T", "All right, OK, I cannot."),               # v2: spelling conventions
        ("IT'S KINDA FUNNY CAUSE I DUNNO", "It's kind of funny because I do not know."),
    ]
    for edacc, whisper in pairs:
        assert text.normalize(edacc) == text.normalize(whisper), (edacc, whisper)


def test_normalizer_keeps_real_words():
    assert text.normalize("Huh? Oh, I am here.") == "huh 0 i am here"  # Whisper's normalizer writes "oh" as 0
    assert text.normalize("A hum and a hymn") == "a hum and a hymn"


def test_filler_loop_normalizes_to_nothing():
    assert text.normalize("m" * 300) == ""


def test_utterance_errors_and_pooling():
    a = metrics.utterance_errors("the cat sat on the mat", "the cat sat on a mat mat")
    assert (a["n_ref"], a["sub"], a["del"], a["ins"]) == (6, 1, 0, 1)
    b = metrics.utterance_errors("hello there", "")
    assert (b["n_ref"], b["sub"], b["del"], b["ins"]) == (2, 0, 2, 0)
    p = metrics.pooled([a, b])
    assert p["ref_words"] == 8 and abs(p["wer"] - 4 / 8) < 1e-9  # pooled over words, not averaged per utterance


def test_report_groups():
    utts = [{**metrics.utterance_errors("a b c d", "a b c d"), "group": "x"},
            {**metrics.utterance_errors("a b", "a c"), "group": "y"}]
    r = metrics.report(utts)
    assert r["by_group"]["x"]["wer"] == 0 and r["by_group"]["y"]["wer"] == 0.5
    assert abs(r["macro_over_groups"] - 0.25) < 1e-9
