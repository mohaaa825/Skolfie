"""
tests/test_metrics.py
----------------------
Hand-computed checks for skolfie/eval/metrics.py.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from skolfie.eval.metrics import (
    BREAKDOWN_COLUMNS,
    binary_metrics,
    breakdown,
    brier_score,
    expected_calibration_error,
    fpr_at_tpr,
    to_binary,
)


def test_perfect_classifier() -> None:
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, p)
    assert (m["n"], m["n_human"], m["n_ai"]) == (4, 2, 2)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["auroc"] == 1.0
    assert m["fpr_at_95tpr"] == 0.0
    # k = ceil(0.95 * 2) = 2 -> the 2nd-highest AI score.
    assert m["threshold_at_95tpr"] == 0.8
    assert m["tpr_at_threshold"] == 1.0
    # Every score sits alone in its bin: (0.1 + 0.2 + 0.2 + 0.1) / 4.
    assert m["ece"] == pytest.approx(0.15)
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert m["notes"] == []

    hard = binary_metrics(y, [0.0, 0.0, 1.0, 1.0])
    assert hard["ece"] == 0.0 and hard["brier"] == 0.0


def test_constant_random_classifier() -> None:
    y = [0, 0, 1, 1]
    m = binary_metrics(y, [0.5] * 4)
    assert m["auroc"] == 0.5
    assert m["accuracy"] == 0.5          # everything is called AI at p >= 0.5
    assert m["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 0))
    assert m["fpr_at_95tpr"] == 1.0       # the threshold is 0.5, every human text >= 0.5
    assert m["ece"] == 0.0                # predicted 0.5, observed 0.5
    assert m["brier"] == 0.25


def test_uninformative_scores_auroc_near_half() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=20_000)
    p = rng.random(20_000)
    m = binary_metrics(y, p)
    assert m["auroc"] == pytest.approx(0.5, abs=0.02)
    # Threshold ~ 0.05 (the 5th percentile of AI scores) -> ~95% of humans flagged.
    assert m["fpr_at_95tpr"] == pytest.approx(0.95, abs=0.02)


def test_known_fpr_at_95_tpr() -> None:
    ai = np.arange(1, 21) / 20  # 0.05, 0.10, ..., 1.00
    human = np.array([0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.3, 0.5, 0.9])
    y = np.r_[np.ones(20), np.zeros(10)].astype(int)
    p = np.r_[ai, human]
    out = fpr_at_tpr(y, p)
    # k = ceil(0.95 * 20) = 19 -> 19th-highest AI score = 0.10; 19/20 AI kept.
    assert out["threshold"] == 0.10
    assert out["tpr"] == 0.95
    # Humans with p >= 0.10: 0.10, 0.12, 0.3, 0.5, 0.9.
    assert out["fpr"] == 0.5
    assert binary_metrics(y, p)["fpr_at_95tpr"] == 0.5


def test_fpr_at_95_tpr_ties_count_as_ai() -> None:
    # All AI scores tie: the threshold is that score and TPR overshoots to 1.0.
    y = [1, 1, 1, 0, 0]
    out = fpr_at_tpr(y, [0.7, 0.7, 0.7, 0.7, 0.6])
    assert out == {"fpr": 0.5, "threshold": 0.7, "tpr": 1.0}


def test_ece_constructed() -> None:
    # Bin [0.2, 0.2667): p = 0.25, 1 of 4 AI -> gap 0.
    # Bin [0.8667, 0.9333): p = 0.9, 6 of 10 AI -> gap 0.3.
    y = [1, 0, 0, 0] + [1] * 6 + [0] * 4
    p = [0.25] * 4 + [0.9] * 10
    assert expected_calibration_error(y, p) == pytest.approx(10 / 14 * 0.3)
    # p = 1.0 falls in the last bin, which also holds 0.95.
    assert expected_calibration_error([1, 0], [1.0, 0.95]) == pytest.approx(abs(0.5 - 0.975))
    # Fewer bins merge the two groups: |7/14 - (4*0.25 + 10*0.9)/14|.
    assert expected_calibration_error(y, p, n_bins=1) == pytest.approx(abs(7 / 14 - 10 / 14))


def test_brier_and_string_labels() -> None:
    assert brier_score(["ai", "human"], [0.75, 0.25]) == pytest.approx(0.0625)
    assert list(to_binary(["AI", "human", 1, 0, True])) == [1, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        to_binary(["robot"])


def test_single_class_and_empty() -> None:
    m = binary_metrics([1, 1, 1], [0.2, 0.6, 0.9])
    assert math.isnan(m["auroc"])
    assert math.isnan(m["fpr_at_95tpr"])  # no humans to measure
    assert m["threshold_at_95tpr"] == 0.2
    assert any("AUROC undefined" in n for n in m["notes"])
    m = binary_metrics([0, 0], [0.2, 0.6])
    assert math.isnan(m["auroc"]) and math.isnan(m["threshold_at_95tpr"])
    assert m["accuracy"] == 0.5
    empty = binary_metrics([], [])
    assert empty["n"] == 0 and math.isnan(empty["auroc"]) and empty["notes"]


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.5])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        binary_metrics([0, 2], [0.5, 0.5])


def test_breakdown_per_group() -> None:
    df = pd.DataFrame({
        "label": ["human", "human", "human", "ai", "ai", "ai"],
        "generator": ["human", "human", "human", "gpt-x", "gpt-x", "llama"],
        "domain": ["news", "news", "qa", "news", "qa", "qa"],
        "source": ["a"] * 6,
    })
    p = [0.1, 0.7, 0.2, 0.9, 0.4, 0.8]
    tables = breakdown(df, p)
    assert set(tables) == {"generator", "domain", "source"}
    gen = tables["generator"].set_index("group")
    assert list(tables["generator"].columns) == BREAKDOWN_COLUMNS
    assert gen.loc["human", "fpr_human"] == pytest.approx(1 / 3)
    assert gen.loc["human", "accuracy"] == pytest.approx(2 / 3)
    assert math.isnan(gen.loc["human", "recall_ai"]) and math.isnan(gen.loc["human", "auroc"])
    assert gen.loc["gpt-x", "recall_ai"] == 0.5
    assert math.isnan(gen.loc["gpt-x", "fpr_human"])
    assert gen.loc["llama", "n"] == 1
    dom = tables["domain"].set_index("group")
    # news: humans 0.1, 0.7 vs AI 0.9 -> AUROC 1.0; qa: human 0.2 vs AI 0.4, 0.8 -> 1.0.
    assert dom.loc["news", "auroc"] == 1.0 and dom.loc["qa", "auroc"] == 1.0
    assert dom.loc["qa", "recall_ai"] == 0.5
    # Missing columns are skipped; the records are JSON-able.
    assert set(breakdown(df.drop(columns="source"), p)) == {"generator", "domain"}
    json.dumps(tables["domain"].to_dict(orient="records"))
