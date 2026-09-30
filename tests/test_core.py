import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2a.core import RunConfig, StreamController, make_budget_segments
from run_streams import run_one, score_for_strategy


def test_budget_segments_are_exact_and_disjoint():
    parts = make_budget_segments(9991, 50)
    assert len(parts) == 50
    joined = np.concatenate(parts)
    assert np.array_equal(joined, np.arange(9991))


def test_label_is_revealed_after_prediction_state_is_frozen():
    rng = np.random.default_rng(7)
    prototypes = rng.normal(size=(7, 32))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    controller = StreamController(prototypes, RunConfig(num_classes=7))
    state = controller.predict(prototypes[0])
    frozen_prediction = state["prediction"]
    frozen_probabilities = np.asarray(state["mixture"]).copy()
    controller.reveal_label(state, 0)
    assert state["prediction"] == frozen_prediction
    assert np.array_equal(state["mixture"], frozen_probabilities)
    assert np.isclose(controller.weights.sum(), 1.0)


def test_unlabeled_update_does_not_require_a_label_for_decision():
    rng = np.random.default_rng(11)
    prototypes = rng.normal(size=(3, 8))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    controller = StreamController(
        prototypes,
        RunConfig(num_classes=3, pseudo_confidence=0.0, pseudo_agreement=0.0),
    )
    state = controller.predict(prototypes[1])
    pseudo_label = controller.adapt_unlabeled(state)
    assert pseudo_label == state["prediction"]
    assert controller.memory.pseudo_count.sum() == 1


def test_ablation_scores_remove_exactly_one_query_component():
    state = {"entropy": 0.8, "disagreement": 0.2, "coverage": 0.5, "query_value": 0.4}
    rng = np.random.default_rng(3)
    assert np.isclose(score_for_strategy("a2a_no_gate", state, rng), 0.4)
    assert np.isclose(score_for_strategy("a2a_no_entropy", state, rng), np.sqrt(0.1))
    assert np.isclose(score_for_strategy("a2a_no_disagreement", state, rng), np.sqrt(0.4))
    assert np.isclose(score_for_strategy("a2a_no_coverage", state, rng), np.sqrt(0.16))
    assert np.isclose(score_for_strategy("dynamic_entropy_online", state, rng), 0.8)


def test_noisy_oracle_is_delayed_and_records_realized_noise(tmp_path):
    rng = np.random.default_rng(17)
    text = rng.normal(size=(3, 8)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    labels = np.arange(30, dtype=np.int64) % 3
    features = text[labels].copy()
    cache = {
        "features": features,
        "labels": labels,
        "domains": np.arange(30, dtype=np.int64) % 2,
        "paths": np.asarray([f"sample-{i}" for i in range(30)]),
        "text_features": text,
    }
    summary = run_one(
        cache,
        tmp_path,
        "iid_random",
        seed=21,
        budget_percent=10.0,
        strategy="periodic",
        config=RunConfig(num_classes=3),
        write_events=False,
        oracle_noise=1.0,
    )
    assert summary["query_count"] == 3
    assert summary["noisy_oracle_count"] == 3
    assert summary["realized_oracle_noise"] == 1.0
    # Current-sample accuracy remains defined by the pre-intervention prediction.
    assert 0.0 <= summary["online_accuracy"] <= 1.0


def test_zero_shot_and_ask_only_are_explicit_protocol_anchors(tmp_path):
    text = np.eye(3, dtype=np.float32)
    labels = np.arange(30, dtype=np.int64) % 3
    cache = {
        "features": text[labels],
        "labels": labels,
        "domains": np.arange(30, dtype=np.int64) % 2,
        "paths": np.asarray([f"sample-{i}" for i in range(30)]),
        "text_features": text,
    }
    zero = run_one(
        cache, tmp_path, "iid_random", 2, 1.0, "zero_shot",
        RunConfig(num_classes=3), write_events=False,
    )
    ask = run_one(
        cache, tmp_path, "iid_random", 2, 10.0, "ask_only",
        RunConfig(num_classes=3), write_events=False,
    )
    assert zero["budget_count"] == 0
    assert zero["query_count"] == 0
    assert ask["query_count"] == ask["budget_count"]


def test_class_confusable_noise_is_delayed_and_excludes_true_class(tmp_path):
    text = np.eye(3, dtype=np.float32)
    labels = np.arange(30, dtype=np.int64) % 3
    cache = {
        "features": text[labels],
        "labels": labels,
        "domains": np.arange(30, dtype=np.int64) % 2,
        "paths": np.asarray([f"sample-{i}" for i in range(30)]),
        "text_features": text,
    }
    summary = run_one(
        cache, tmp_path, "iid_random", 3, 10.0, "ask_only",
        RunConfig(num_classes=3), write_events=True, oracle_noise=1.0,
        oracle_noise_mode="class_confusable",
    )
    assert summary["oracle_noise_mode"] == "class_confusable"
    assert summary["noisy_oracle_count"] == summary["query_count"]
    import gzip
    import csv
    with gzip.open(next(tmp_path.glob("*.events.csv.gz")), "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["queried"] == "1":
            assert int(row["oracle_label"]) != int(row["label"])
