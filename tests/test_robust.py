"""
tests/test_robust.py
---------------------
Roadmap step 9: the evasion attacks (skolfie/robust/attacks.py), the attack
set builder (scripts/make_attacks.py), the robustness report
(scripts/eval_robust.py) and evaluate.py --extra-train, all offline: the LLM
attacks use the fake provider and the T5 attack a fake paraphraser.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from skolfie.data.generation.providers import FakeProvider, ProviderError
from skolfie.mixed.chunking import split_sentences
from skolfie.models.baselines import LengthBaseline, TfidfLRBaseline
from skolfie.robust import attacks as A
from tests.test_evaluate import make_corpus, make_splits
from tests.test_stacker import _load_script

TEXT = (
    "The committee will consider the important proposal tomorrow morning. However, numerous "
    "members believe it is a crucial step for the whole society! Critics argue that the plan "
    "is too expensive and too slow.\nIn the end, the vote will show whether the approach "
    "works. Many people hope for a clear result, and the mayor promised a quick decision."
)


def _frame(texts, ids=None, source="s1", domain="news") -> pd.DataFrame:
    ids = ids or [f"x:{i}" for i in range(len(texts))]
    return pd.DataFrame({"id": ids, "text": texts, "source": source, "domain": domain})


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_metadata() -> None:
    assert list(A.ATTACKS) == ["typo_5", "typo_15", "synonym_15", "homoglyph", "zero_width",
                               "t5_paraphrase", "llm_paraphrase", "humanizer", "temp_low",
                               "temp_high", "human_edit_30"]
    for name in ("typo_5", "synonym_15", "homoglyph", "zero_width"):
        assert A.ATTACKS[name].applies_to == "both"
    for name in ("t5_paraphrase", "llm_paraphrase", "humanizer", "temp_low", "human_edit_30"):
        assert A.ATTACKS[name].applies_to == "ai"
    assert A.ATTACKS["t5_paraphrase"].needs_gpu and not A.ATTACKS["t5_paraphrase"].needs_api
    assert all(A.ATTACKS[n].needs_api for n in ("llm_paraphrase", "humanizer", "temp_low",
                                                 "temp_high"))
    assert not any(A.ATTACKS[n].needs_api or A.ATTACKS[n].needs_gpu
                   for n in ("typo_5", "synonym_15", "homoglyph", "zero_width", "human_edit_30"))
    assert A.get_attack("typo_40").rate == 0.4
    assert A.get_attack("homoglyph").rate == A.get_attack("homoglyph_20").rate == 0.2
    assert A.get_attack("temp_high").params["temperature"] == 1.3
    for bad in ("typo", "nope", "typo_101", "human_edit_100", "human_edit_0", "typo_x"):
        with pytest.raises(ValueError):
            A.get_attack(bad)


# --------------------------------------------------------------------------- #
# Character and word noise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["typo_10", "synonym_20", "homoglyph", "zero_width"])
def test_noise_attacks_are_deterministic(name: str) -> None:
    frame = _frame([TEXT, TEXT.upper(), "Short text here."])
    ctx = A.AttackContext(seed=3, synonyms=A.TableSynonyms())
    first = A.apply_attack(name, frame, ctx)
    assert first == A.apply_attack(name, frame, A.AttackContext(seed=3))
    assert first[0] != TEXT
    # Per text: the same text is attacked the same way whatever the batch holds.
    assert A.apply_attack(name, frame.iloc[[0]], ctx)[0] == first[0]
    assert A.apply_attack(name, frame, A.AttackContext(seed=4))[0] != first[0]


def test_typo_rate() -> None:
    rng = A.text_rng(0, "t", TEXT)
    assert A.typo_text(TEXT, 0.0, rng) == TEXT
    long_text = " ".join([TEXT] * 20)
    words = long_text.split()
    for rate in (0.05, 0.15, 0.4):
        new = A.typo_text(long_text, rate, A.text_rng(1, "t", long_text))
        eligible = sum(1 for w in words if A._TYPO_WORD.search(w))
        changed = sum(a != b for a, b in zip(words, new.split()))
        assert len(new.split()) == len(words)  # typos never add or remove words
        assert abs(changed / eligible - rate) < 0.02
        assert [w[0] for w in new.split()] == [w[0] for w in words]  # first letters kept
    assert "\n" in A.typo_text(TEXT, 0.5, A.text_rng(0, "t", TEXT))


def test_keyboard_neighbours() -> None:
    assert set("qwsz") <= set(A.KEYBOARD_NEIGHBOURS["a"])
    assert "g" not in A.KEYBOARD_NEIGHBOURS["g"]
    assert all(len(v) >= 2 for v in A.KEYBOARD_NEIGHBOURS.values())


def test_synonym_attack_with_injected_table() -> None:
    table = A.TableSynonyms({"proposal": ["plan"], "important": ["key"], "the": ["thee"],
                             "committee": ["panel"], "members": ["delegates"]})
    rng = A.text_rng(0, "s", TEXT)
    new = A.synonym_text(TEXT, 1.0, rng, table)
    assert "important proposal" not in new and "key plan" in new
    assert "panel" in new and "thee" not in new  # stop words are never replaced
    capital = A.synonym_text("Important news.", 1.0, rng, table)
    assert capital == "Key news."
    assert A.synonym_text("IMPORTANT news.", 1.0, rng, table) == "KEY news."
    assert A.synonym_text(TEXT, 0.0, rng, table) == TEXT
    # The rate is a share of the content words; only words with a synonym can change.
    single = A.TableSynonyms({k: [w for w in v if " " not in w]
                              for k, v in A.SYNONYM_TABLE.items()})
    long_text = " ".join([TEXT] * 20)
    new = A.synonym_text(long_text, 0.1, A.text_rng(1, "s", long_text), single)
    changed = sum(a != b for a, b in zip(long_text.split(), new.split()))
    content = [w for w in A._WORD.findall(long_text) if w.lower() not in A._stop_words()]
    assert changed == round(0.1 * len(content))


def test_wordnet_synonyms_selection() -> None:
    class Lemma:
        def __init__(self, name, count):
            self._name, self._count = name, count

        def name(self):
            return self._name

        def count(self):
            return self._count

    class Synset:
        def __init__(self, pos, lemmas):
            self._pos, self._lemmas = pos, lemmas

        def pos(self):
            return self._pos

        def lemmas(self):
            return [Lemma(n, c) for n, c in self._lemmas]

    class FakeWordNet:
        senses = {"big": [Synset("a", [("big", 9), ("large", 5), ("full-grown", 1)]),
                          Synset("a", [("bad", 1), ("big", 2)]),
                          Synset("a", [("heavy", 3)]),
                          Synset("v", [("bulk", 4)])],
                  "houses": [Synset("n", [("house", 5), ("home", 3)])]}

        def synsets(self, word):
            return self.senses.get(word, [])

        def morphy(self, word, pos):
            return {"houses": "house"}.get(word, word)

    syn = A.WordNetSynonyms.__new__(A.WordNetSynonyms)
    syn.wn, syn.max_candidates, syn._cache = FakeWordNet(), 3, {}
    # Most common sense's part of speech, top two senses, one-word lemmas, by frequency.
    assert syn("big") == ("large", "bad")
    assert syn("houses") == ()  # inflected forms are left alone
    assert syn("unknown") == ()


def test_synonym_source_falls_back_to_table(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise LookupError("no wordnet")

    monkeypatch.setattr(A, "WordNetSynonyms", missing)
    assert isinstance(A.load_synonyms("auto"), A.TableSynonyms)
    assert isinstance(A.load_synonyms("table"), A.TableSynonyms)
    with pytest.raises(LookupError):
        A.load_synonyms("wordnet")
    with pytest.raises(ValueError):
        A.load_synonyms("thesaurus")


def test_unicode_attacks_change_bytes_but_look_the_same() -> None:
    for fn in (A.homoglyph_text, A.zero_width_text):
        new = fn(TEXT, 0.3, A.text_rng(0, fn.__name__, TEXT))
        assert new != TEXT and new.encode() != TEXT.encode()
        assert A.undo_unicode_tricks(new) == TEXT
        assert fn(TEXT, 0.0, A.text_rng(0, "x", TEXT)) == TEXT
    glyph = A.homoglyph_text(TEXT, 1.0, A.text_rng(0, "h", TEXT))
    assert len(glyph) == len(TEXT) and not glyph.isascii()
    width = A.zero_width_text(TEXT, 1.0, A.text_rng(0, "z", TEXT))
    assert width.count(A.ZERO_WIDTH) == sum(len(w) >= 2 for w in TEXT.split())
    assert width.replace(A.ZERO_WIDTH, "").split() == TEXT.split()


# --------------------------------------------------------------------------- #
# Human-edit splice
# --------------------------------------------------------------------------- #
def test_human_edit_splices_from_same_source_and_domain() -> None:
    donors = A.DonorPool(
        ["Same group one. Same group two. Same group three.", "Other domain sentence here.",
         "Other source sentence here."],
        ["s1", "s1", "s2"], ["news", "casual", "news"])
    out = A.apply_attack("human_edit_40", _frame([TEXT]), A.AttackContext(seed=0, donors=donors))
    sents = [out[0][s:e] for s, e in split_sentences(out[0])]
    orig = [TEXT[s:e] for s, e in split_sentences(TEXT)]
    assert len(sents) == len(orig) == 5
    replaced = [s for s, o in zip(sents, orig) if s != o]
    assert len(replaced) == 2  # round(0.4 * 5)
    assert all(s.startswith("Same group") for s in replaced)
    assert "\n" in out[0]  # the layout of the AI text is kept
    # No donor of the same (source, domain), or a single sentence: not attacked.
    ctx = A.AttackContext(seed=0, donors=donors)
    assert A.apply_attack("human_edit_30", _frame([TEXT], domain="legal"), ctx) == [None]
    assert A.apply_attack("human_edit_30", _frame(["Just one sentence."]), ctx) == [None]
    with pytest.raises(ValueError):
        A.apply_attack("human_edit_30", _frame([TEXT]), A.AttackContext())


def test_human_edit_never_replaces_every_sentence() -> None:
    donors = A.DonorPool(["Human one. Human two. Human three."], ["s1"], ["news"])
    two = "First AI sentence. Second AI sentence."
    out = A.human_edit_text(two, "s1", "news", 0.99, A.text_rng(0, "h", two), donors)
    assert out is not None and ("First AI sentence." in out or "Second AI sentence." in out)


# --------------------------------------------------------------------------- #
# T5 paraphrase (fake paraphraser)
# --------------------------------------------------------------------------- #
def test_t5_paraphrase_with_injected_paraphraser() -> None:
    calls = []

    def fake(sentences: list[str]) -> list[str]:
        calls.append(list(sentences))
        return [f"Reworded {len(s.split())} words." if "mayor" not in s else ""
                for s in sentences]

    texts = [TEXT, "Ok. Too short to matter."]
    out = A.apply_attack("t5_paraphrase", _frame(texts), A.AttackContext(paraphraser=fake))
    assert len(calls) == 1  # one batched call for every sentence of every text
    assert len(calls[0]) == 5 + 1  # "Ok." has fewer than 3 words and is kept as is
    assert out[0].count("Reworded") == 4 and "mayor promised" in out[0]  # empty output: kept
    assert "\n" in out[0]
    assert out[1].startswith("Ok. Reworded")
    with pytest.raises(ValueError):
        A.paraphrase_texts([TEXT], lambda s: s[:-1])


def test_t5_paraphraser_loads_lazily() -> None:
    para = A.T5Paraphraser("some/model", device="cpu", batch_size=4)
    assert para._model is None and para.generate_kwargs["num_beams"] == 5


# --------------------------------------------------------------------------- #
# LLM attacks (fake provider)
# --------------------------------------------------------------------------- #
def test_llm_paraphrase_and_humanizer() -> None:
    def respond(prompt: str) -> str:
        if "REFUSE" in prompt:
            return "I'm sorry, but I can't help with that."
        return "Sure! Here is the rewritten text:\n" + " ".join(["word"] * 30)

    provider = FakeProvider(responses=respond)
    frame = _frame([TEXT, TEXT + " REFUSE"])
    ctx = A.AttackContext(provider=provider, workers=2)
    out = A.apply_attack("llm_paraphrase", frame, ctx)
    assert out == [" ".join(["word"] * 30), None]
    assert ctx.meta["llm_paraphrase"][0]["model"] == "fake-model"
    A.apply_attack("humanizer", frame, ctx)
    prompts = [c["prompt"] for c in provider.calls]
    assert "Paraphrase the text below" in prompts[0] and TEXT in prompts[0]
    assert "contractions" in prompts[2] and "sentence length" in prompts[2]
    assert {c["temperature"] for c in provider.calls} == {A.LLM_TEMPERATURE}
    with pytest.raises(ValueError):
        A.apply_attack("humanizer", frame, A.AttackContext())


def test_temperature_attacks() -> None:
    provider = FakeProvider(responses=lambda p: " ".join(["more"] * 25))
    fresh_id = "fresh:claude-x:tech-explainer.con09.persona-teacher.w600.t06:0"
    frame = pd.DataFrame({"id": [fresh_id, "hc3:a:1:chatgpt:0"], "text": [TEXT, TEXT],
                          "source": "s", "domain": "news"})
    ctx = A.AttackContext(provider=provider)
    out = A.apply_attack("temp_high", frame, ctx)
    assert [c["temperature"] for c in provider.calls] == [1.3, 1.3]
    assert provider.calls[0]["prompt"].startswith("Write an explainer")  # the original prompt
    assert "Continue the text" in provider.calls[1]["prompt"]
    assert [m["prompt"] for m in ctx.meta["temp_high"]] == ["original", "continuation"]
    assert out[0] == " ".join(["more"] * 25)
    opening = TEXT[:split_sentences(TEXT)[0][1]]
    assert out[1].startswith(opening) and out[1].endswith("more")
    A.apply_attack("temp_low", frame.iloc[:1], ctx)
    assert provider.calls[-1]["temperature"] == 0.2


def test_temperature_problems() -> None:
    class NoTemp:
        model_name = "m"
        send_temperature = False

    class Anthropic:
        model_name = "claude-old"
        provider_name = "anthropic"

    assert A.temperature_problem(FakeProvider(), 1.3) is None
    assert "temperature" in A.temperature_problem(NoTemp(), 0.2)
    assert A.temperature_problem(Anthropic(), 0.2) is None
    assert "up to 1" in A.temperature_problem(Anthropic(), 1.3)
    with pytest.raises(ValueError):
        A.apply_attack("temp_low", _frame([TEXT]), A.AttackContext(provider=NoTemp()))


def test_fresh_prompt_parsing() -> None:
    assert A.fresh_prompt("hc3:x:1") is None
    assert A.fresh_prompt("fresh:gen:not-a-template.x.plain.w100.t06:0") is None
    assert A.fresh_prompt("fresh:gen:tech-explainer.con99.plain.w100.t06:0") is None
    text = A.fresh_prompt("fresh:llama3.3:70b:tech-explainer.con09.plain.w100.t10:2")
    assert text is not None and "about 100 words" in text


def test_provider_errors() -> None:
    class Flaky:
        model_name = "m"

        def __init__(self, status):
            self.status = status

        def generate(self, prompt, *, temperature, max_tokens):
            raise ProviderError("boom", status=self.status)

    assert A.run_prompts(Flaky(500), ["a", "b"], temperature=0.5, max_tokens=[10, 10],
                         log=lambda *_: None) == [None, None]
    with pytest.raises(ProviderError):
        A.run_prompts(Flaky(401), ["a"], temperature=0.5, max_tokens=[10])


def test_clean_llm_output() -> None:
    body = " ".join(["word"] * 25)
    assert A.clean_llm_output(f"Here's the paraphrased text:\n{body}") == body
    assert A.clean_llm_output(f"<think>hmm</think>\"{body}\"") == body
    assert A.clean_llm_output("too short") is None
    assert A.clean_llm_output("") is None
    assert A.clean_llm_output("As an AI, I cannot rewrite this text. " + body) is None


# --------------------------------------------------------------------------- #
# Scripts
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("robustdata")
    corpus = make_corpus(n_per_cell=60, seed=5)
    splits = make_splits(corpus, seed=5)
    corpus.to_parquet(root / "corpus.parquet", index=False)
    splits.to_parquet(root / "splits.parquet", index=False)
    train = corpus[corpus["id"].isin(splits.loc[splits["split"] == "train", "id"])]
    for name, model in (("tfidf", TfidfLRBaseline()), ("length", LengthBaseline())):
        joblib.dump(model.fit(train["text"].tolist(), train["label"].tolist()),
                    root / f"{name}.joblib")
    return {"root": root, "corpus": root / "corpus.parquet", "splits": root / "splits.parquet",
            "tfidf": root / "tfidf.joblib", "length": root / "length.joblib"}


def _make_argv(data: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
            "--out-dir", str(out), "--min-words", "20", "--synonym-source", "table", *extra]


def test_make_attacks_list(capsys) -> None:
    assert _load_script("make_attacks").main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "t5_paraphrase" in out and "human_edit_30" in out


def test_api_attacks_skip_without_keys(monkeypatch) -> None:
    script = _load_script("make_attacks")
    for _, key in script.API_KEYS:
        monkeypatch.delenv(key, raising=False)
    args = script.build_parser().parse_args(["--provider", "auto"])
    provider, why = script.make_llm_provider(args)
    assert provider is None and "ANTHROPIC_API_KEY" in why
    monkeypatch.setattr(script, "_cuda_available", lambda: False)
    run, skipped = script.select_attacks(args, None, why, log=lambda *_: None)
    assert set(skipped) == {"t5_paraphrase", "llm_paraphrase", "humanizer", "temp_low",
                            "temp_high"}
    assert run == ["typo_5", "typo_15", "synonym_15", "homoglyph", "zero_width",
                   "human_edit_30"]
    args = script.build_parser().parse_args(["--provider", "anthropic"])
    assert "ANTHROPIC_API_KEY" in script.make_llm_provider(args)[1]
    # Named explicitly, the T5 attack runs even without a GPU.
    args = script.build_parser().parse_args(["--attacks", "t5_paraphrase,humanizer"])
    run, skipped = script.select_attacks(args, None, why, log=lambda *_: None)
    assert run == ["t5_paraphrase"] and list(skipped) == ["humanizer"]


def test_make_attacks_eval_sets(data, capsys) -> None:
    script = _load_script("make_attacks")
    out = data["root"] / "attacks-test"
    argv = _make_argv(data, out, "--n", "15", "--provider", "fake", "--max-words", "60",
                      "--attacks", "typo_10,synonym_15,zero_width,human_edit_30,llm_paraphrase,"
                                   "temp_low")
    assert script.main(argv) == 0
    df = pd.read_parquet(out / "attacks.parquet")
    assert list(df.columns) == script.OUT_COLUMNS
    assert df["id"].is_unique
    assert (df["id"] == df["attack"] + ":" + df["orig_id"]).all()
    clean = df[df["attack"] == "clean"]
    assert clean["label"].value_counts().to_dict() == {"ai": 15, "human": 15}
    counts = df.groupby(["attack", "label"]).size().to_dict()
    for name in ("typo_10", "synonym_15", "zero_width"):
        assert counts[(name, "ai")] == counts[(name, "human")] == 15
    for name in ("human_edit_30", "llm_paraphrase", "temp_low"):
        assert (name, "human") not in counts and counts[(name, "ai")] > 0
    assert df["n_words"].max() <= 60
    # Attacked texts are paired with their clean versions.
    merged = df.merge(clean[["orig_id", "text", "label"]], on="orig_id", suffixes=("", "_clean"))
    assert (merged["label"] == merged["label_clean"]).all()
    typo = merged[merged["attack"] == "typo_10"]
    assert (typo["text"] != typo["text_clean"]).all()
    assert typo["word_change"].between(0.03, 0.2).mean() > 0.8
    temp = df[df["attack"] == "temp_low"]
    assert (temp["generator"] == "fake-model").all()
    assert json.loads(temp["meta"].iloc[0])["temperature"] == 0.2
    manifest = json.loads((out / "attacks_manifest.json").read_text())
    assert manifest["mode"] == "eval" and manifest["clean"] == {"n_ai": 15, "n_human": 15}
    assert manifest["attacks"]["typo_10"]["n"] == 30 and manifest["skipped"] == {}

    # Rerun: every attack is reused from parts/ and the output is identical.
    capsys.readouterr()
    assert script.main(argv) == 0
    assert capsys.readouterr().out.count("reused") == 6
    pd.testing.assert_frame_equal(pd.read_parquet(out / "attacks.parquet"), df)


def test_make_attacks_skips_api_attacks(data, monkeypatch) -> None:
    script = _load_script("make_attacks")
    out = data["root"] / "attacks-noapi"
    assert script.main(_make_argv(data, out, "--n", "5", "--provider", "none",
                                  "--attacks", "typo_5,humanizer")) == 0
    manifest = json.loads((out / "attacks_manifest.json").read_text())
    assert list(manifest["attacks"]) == ["typo_5"]
    assert "--provider none" in manifest["skipped"]["humanizer"]
    assert script.main(_make_argv(data, out, "--provider", "none",
                                  "--attacks", "humanizer")) == 2
    with pytest.raises(SystemExit):
        script.main(_make_argv(data, out, "--attacks", "typo"))


def test_make_attacks_augmentation(data) -> None:
    script = _load_script("make_attacks")
    out = data["root"] / "attacks-train"
    with pytest.raises(SystemExit):  # never build training data from an evaluation split
        script.main(_make_argv(data, out, "--share", "0.5", "--split", "test"))
    with pytest.raises(SystemExit):
        script.main(_make_argv(data, out, "--share", "0.5", "--split", "train",
                               "--attacks", "typo_5", "--weight", "homoglyph=2"))
    assert script.main(_make_argv(
        data, out, "--split", "train", "--share", "0.5", "--attacks",
        "typo_10,homoglyph,human_edit_30", "--weight", "human_edit_30=2")) == 0
    df = pd.read_parquet(out / "attacks.parquet")
    assert "clean" not in set(df["attack"])
    assert df["orig_id"].is_unique  # one attack per sampled text
    manifest = json.loads((out / "attacks_manifest.json").read_text())
    assert manifest["mode"] == "augment" and manifest["weights"] == {"human_edit_30": 2.0}
    splits = pd.read_parquet(data["splits"])
    train_ids = set(splits.loc[splits["split"] == "train", "id"])
    assert set(df["orig_id"]) <= train_ids
    stats = manifest["attacks"]
    for name in ("typo_10", "homoglyph"):  # human copies in the train split's proportion
        assert stats[name]["n_human"] > 0
        assert abs(stats[name]["n_human"] - stats[name]["n_ai"]) <= 0.5 * stats[name]["n_ai"] + 2
    assert stats["human_edit_30"]["n_human"] == 0
    assert stats["human_edit_30"]["n_input"] > stats["typo_10"]["n_ai"]
    assert sum(s["n_ai"] for s in stats.values()) <= round(0.5 * 200)


def test_threshold_at_fpr() -> None:
    script = _load_script("eval_robust")
    p = np.linspace(0, 1, 101)
    t = script.threshold_at_fpr(p, 0.05)
    assert np.mean(p >= t) <= 0.05 and np.mean(p >= t) == pytest.approx(0.05, abs=0.011)
    assert np.mean(p >= script.threshold_at_fpr(p, 0.0)) == 0
    assert script.threshold_at_fpr([0.2, 0.2, 0.2], 0.01) > 0.2


def test_eval_robust_end_to_end(data, capsys) -> None:
    make = _load_script("make_attacks")
    attacks_dir = data["root"] / "attacks-eval"
    assert make.main(_make_argv(data, attacks_dir, "--n", "20", "--provider", "fake",
                                "--attacks", "typo_15,homoglyph,llm_paraphrase,human_edit_30")) == 0
    script = _load_script("eval_robust")
    out = data["root"] / "robust"
    argv = ["--attacks", str(attacks_dir / "attacks.parquet"),
            "--model", f"tfidf={data['tfidf']}", "--model", str(data["length"]),
            "--out-dir", str(out)]
    assert script.main(argv) == 0
    assert "Roadmap check: length vs tfidf" in capsys.readouterr().out

    report = json.loads((out / "robust_report.json").read_text())
    assert list(report["models"]) == ["tfidf", "length"]
    tfidf = report["models"]["tfidf"]
    manifest = json.loads((attacks_dir / "attacks_manifest.json").read_text())
    assert tfidf["clean"]["n_ai"] == manifest["clean"]["n_ai"] > 10
    assert tfidf["clean"]["n_human"] == manifest["clean"]["n_human"] == 20
    assert tfidf["clean"]["fpr_thr"] <= 0.01 + 1e-9
    assert list(tfidf["attacks"]) == ["typo_15", "homoglyph", "llm_paraphrase", "human_edit_30"]
    typo = tfidf["attacks"]["typo_15"]
    assert typo["applies_to"] == "both" and typo["n_human"] == 20
    assert typo["n_ai"] == tfidf["clean"]["n_ai"]
    for key in ("auroc", "auroc_both", "auroc_drop", "recall_05", "recall_thr",
                "recall_thr_drop", "fpr_05", "fpr_thr", "fpr_thr_rise"):
        assert key in typo
    assert typo["auroc_drop"] == pytest.approx(typo["auroc_clean"] - typo["auroc"])
    para = tfidf["attacks"]["llm_paraphrase"]
    assert para["applies_to"] == "ai" and "fpr_05" not in para
    check = report["comparison"]["length"]
    assert set(check["paraphrase"]) == {"llm_paraphrase"}
    assert isinstance(check["pass"], bool)

    preds = pd.read_parquet(out / "robust_predictions.parquet")
    n_rows = len(pd.read_parquet(attacks_dir / "attacks.parquet"))
    assert len(preds) == 2 * n_rows and set(preds["model"]) == {"tfidf", "length"}
    assert preds["p_ai"].between(0, 1).all()
    md = (out / "robust_report.md").read_text()
    assert md.startswith("# Robustness") and "| typo_15 |" in md

    # One model, a subset of attacks; errors for unknown attacks or files.
    assert script.main(["--attacks", str(attacks_dir / "attacks.parquet"), "--model",
                        str(data["tfidf"]), "--only", "homoglyph",
                        "--out-dir", str(out / "one")]) == 0
    one = json.loads((out / "one" / "robust_report.json").read_text())
    assert list(one["models"]["tfidf"]["attacks"]) == ["homoglyph"]
    assert one["comparison"] is None
    with pytest.raises(SystemExit):
        script.main(["--attacks", str(attacks_dir / "attacks.parquet"), "--model",
                     str(data["tfidf"]), "--only", "nope"])
    with pytest.raises(SystemExit):
        script.main(["--attacks", str(attacks_dir / "attacks.parquet"), "--model",
                     str(data["root"] / "missing.joblib")])


def test_eval_robust_scores_a_stacked_model(data) -> None:
    # The product model is a StackedModel that loads its members itself; the
    # Unicode attacks no longer move any score (the models normalise text).
    from skolfie.models.stacker import StackedModel, Stacker, operating_points

    make = _load_script("make_attacks")
    attacks_dir = data["root"] / "attacks-unicode"
    assert make.main(_make_argv(data, attacks_dir, "--n", "20", "--provider", "none",
                                "--attacks", "homoglyph,zero_width")) == 0
    rows = pd.read_parquet(attacks_dir / "attacks.parquet")
    members = {"tfidf": data["tfidf"], "length": data["length"]}
    texts, labels = rows["text"].tolist(), rows["label"].tolist()
    P = np.column_stack([joblib.load(path).predict_proba(texts) for path in members.values()])
    stacker = Stacker().fit(P, labels)
    joblib.dump(StackedModel(members, stacker,
                             operating_points(labels, stacker.predict_proba(P))),
                data["root"] / "stacked.joblib")

    script = _load_script("eval_robust")
    out = data["root"] / "robust-stacked"
    assert script.main(["--attacks", str(attacks_dir / "attacks.parquet"), "--model",
                        f"stacked={data['root'] / 'stacked.joblib'}",
                        "--out-dir", str(out)]) == 0
    report = json.loads((out / "robust_report.json").read_text())
    stacked = report["models"]["stacked"]
    assert list(stacked["attacks"]) == ["homoglyph", "zero_width"]
    for r in stacked["attacks"].values():
        assert r["n_ai"] > 0 and r["n_human"] > 0
        assert r["auroc_drop"] == pytest.approx(0, abs=1e-9)
        assert r["recall_thr_drop"] == r["fpr_thr_rise"] == 0
        assert r["mean_p_ai"] == pytest.approx(r["mean_p_ai_clean"])


def test_roadmap_check() -> None:
    script = _load_script("eval_robust")
    ref = {"clean": {"fpr_05": 0.02, "fpr_at_95tpr": 0.1},
           "attacks": {"t5_paraphrase": {"auroc_drop": 0.2}, "typo_5": {"auroc_drop": 0.1}}}
    good = {"clean": {"fpr_05": 0.022, "fpr_at_95tpr": 0.1},
            "attacks": {"t5_paraphrase": {"auroc_drop": 0.08}}}
    res = script.roadmap_check(ref, good, A.PARAPHRASE_ATTACKS, 0.005)
    assert res["pass"] and res["paraphrase"]["t5_paraphrase"]["ratio"] == pytest.approx(0.4)
    worse_fpr = {**good, "clean": {"fpr_05": 0.03, "fpr_at_95tpr": 0.1}}
    assert not script.roadmap_check(ref, worse_fpr, A.PARAPHRASE_ATTACKS, 0.005)["pass"]
    not_halved = {**good, "attacks": {"t5_paraphrase": {"auroc_drop": 0.15}}}
    assert not script.roadmap_check(ref, not_halved, A.PARAPHRASE_ATTACKS, 0.005)["pass"]


def test_evaluate_extra_train(data, capsys) -> None:
    corpus = pd.read_parquet(data["corpus"])
    splits = pd.read_parquet(data["splits"])
    train_ids = splits.loc[splits["split"] == "train", "id"].tolist()
    test_ids = splits.loc[splits["split"] == "test", "id"].tolist()
    val_ids = splits.loc[splits["split"] == "val", "id"].tolist()
    text = corpus.set_index("id")["text"]
    extra = pd.DataFrame({
        "text": ["extra one " + text[train_ids[0]], "extra two " + text[train_ids[1]],
                 "  " + text[test_ids[0]] + " ",   # an evaluation text (whitespace aside)
                 "edited " + text[val_ids[0]]],   # attacked copy of an evaluation text
        "label": ["ai", "human", "ai", "ai"],
        "orig_id": [train_ids[0], train_ids[1], test_ids[0], val_ids[0]],
        "attack": ["typo_5", "typo_5", "typo_5", "homoglyph"],
    })
    path = data["root"] / "extra.parquet"
    extra.to_parquet(path, index=False)
    evaluate = _load_script("evaluate")
    out = data["root"] / "adv"
    assert evaluate.main(["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
                          "--model", "length", "--extra-train", str(path),
                          "--out-dir", str(out)]) == 0
    log = capsys.readouterr().out
    assert "dropped 2 extra train texts" in log and "+ 2 extra train texts" in log
    report = json.loads((out / "report.json").read_text())
    info = report["train"]["extra_train"]
    assert info == {"n_given": 4, "n_dropped_eval_overlap": 2, "n": 2, "n_ai": 1, "n_human": 1,
                    "files": [str(path)], "by_file": {str(path): 2},
                    "by_attack": {"typo_5": 2}}
    assert report["train"]["n"] == len(train_ids)  # the split's own count is unchanged
    assert "plus 2 extra texts" in (out / "report.md").read_text()

    bad = data["root"] / "bad.parquet"
    pd.DataFrame({"text": ["x"]}).to_parquet(bad, index=False)
    for extra_path in (bad, data["root"] / "missing.parquet"):
        with pytest.raises(SystemExit):
            evaluate.main(["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
                           "--model", "length", "--extra-train", str(extra_path)])
