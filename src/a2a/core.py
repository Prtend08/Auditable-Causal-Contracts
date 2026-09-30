from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np


EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(), EPS)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def normalized_entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.maximum(p, EPS))).sum() / np.log(len(p)))


def margin_uncertainty(p: np.ndarray) -> float:
    top = np.partition(p, -2)[-2:]
    return float(1.0 - (top.max() - top.min()))


def kl_to_mixture_disagreement(expert_probs: np.ndarray, mixture: np.ndarray) -> float:
    kl = expert_probs * (
        np.log(np.maximum(expert_probs, EPS))
        - np.log(np.maximum(mixture[None, :], EPS))
    )
    # Mean KL from uniformly averaged experts to the learned weighted mixture,
    # scaled by log(M). This is not generally Jensen-Shannon divergence and is
    # not guaranteed to lie in [0, 1] when the mixture weights are non-uniform.
    return float(np.mean(np.sum(kl, axis=1)) / np.log(expert_probs.shape[0]))


# Backwards-compatible name for archived scripts; arithmetic is unchanged.
js_disagreement = kl_to_mixture_disagreement


def make_budget_segments(length: int, budget: int) -> List[np.ndarray]:
    """Return exactly ``budget`` non-empty chronological segments."""
    if budget <= 0:
        return []
    if budget > length:
        raise ValueError(f"budget {budget} exceeds stream length {length}")
    return [part for part in np.array_split(np.arange(length), budget) if len(part)]


@dataclass(frozen=True)
class RunConfig:
    num_classes: int
    weight_learning_rate: float = 0.5
    weight_floor_mixing: float = 0.02
    pseudo_confidence: float = 0.80
    pseudo_agreement: float = 0.75
    prototype_prior_count: float = 2.0
    pseudo_momentum: float = 0.95
    logit_scale: float = 100.0
    persistence_switch_rate: float = 0.35
    risk_entropy_threshold: float = 0.50

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class PrototypeMemory:
    def __init__(self, text_prototypes: np.ndarray, config: RunConfig):
        self.text = normalize_rows(text_prototypes.astype(np.float64))
        self.config = config
        classes, dim = self.text.shape
        self.labeled_sum = np.zeros((classes, dim), dtype=np.float64)
        self.labeled_count = np.zeros(classes, dtype=np.int64)
        self.pseudo = self.text.copy()
        self.pseudo_count = np.zeros(classes, dtype=np.int64)

    def labeled_prototypes(self) -> np.ndarray:
        numer = self.labeled_sum + self.config.prototype_prior_count * self.text
        denom = self.labeled_count[:, None] + self.config.prototype_prior_count
        return normalize_rows(numer / denom)

    def add_label(self, feature: np.ndarray, label: int) -> None:
        self.labeled_sum[label] += feature
        self.labeled_count[label] += 1

    def add_pseudo(self, feature: np.ndarray, label: int) -> None:
        momentum = self.config.pseudo_momentum if self.pseudo_count[label] else 0.0
        self.pseudo[label] = normalize_rows(
            (momentum * self.pseudo[label] + (1.0 - momentum) * feature)[None, :]
        )[0]
        self.pseudo_count[label] += 1

    def expert_probabilities(self, feature: np.ndarray) -> np.ndarray:
        labeled = self.labeled_prototypes()
        hybrid = normalize_rows(0.5 * labeled + 0.5 * self.pseudo)
        prototypes = [self.text, labeled, self.pseudo, hybrid]
        return np.stack(
            [softmax(self.config.logit_scale * (feature @ proto.T)) for proto in prototypes]
        )


class StreamController:
    """Four-expert controller with delayed selective supervision.

    The public ``predict`` method never receives the ground-truth label. The
    caller records the prediction before invoking ``adapt_unlabeled`` or
    ``reveal_label``.
    """

    expert_names = (
        "zero_shot_text",
        "labeled_prototype",
        "pseudo_labeled_prototype",
        "conservative_hybrid",
    )

    def __init__(self, text_prototypes: np.ndarray, config: RunConfig):
        self.config = config
        self.memory = PrototypeMemory(text_prototypes, config)
        self.weights = np.full(len(self.expert_names), 1.0 / len(self.expert_names))

    def predict(self, feature: np.ndarray) -> Dict[str, object]:
        feature = feature.astype(np.float64)
        feature = feature / max(np.linalg.norm(feature), EPS)
        expert_probs = self.memory.expert_probabilities(feature)
        mixture = np.sum(self.weights[:, None] * expert_probs, axis=0)
        mixture /= mixture.sum()
        expert_preds = expert_probs.argmax(axis=1)
        pred = int(mixture.argmax())
        agreement = float(np.mean(expert_preds == pred))
        entropy = normalized_entropy(mixture)
        margin = margin_uncertainty(mixture)
        disagreement = kl_to_mixture_disagreement(expert_probs, mixture)
        coverage = float(1.0 / np.sqrt(self.memory.labeled_count[pred] + 1.0))
        query_value = float(max(entropy * disagreement * coverage, 0.0) ** (1.0 / 3.0))
        return {
            "feature": feature,
            "expert_probs": expert_probs,
            "mixture": mixture,
            "prediction": pred,
            "confidence": float(mixture[pred]),
            "agreement": agreement,
            "entropy": entropy,
            "margin": margin,
            "disagreement": disagreement,
            "coverage": coverage,
            "query_value": query_value,
            "weights_before": self.weights.copy(),
        }

    def adapt_unlabeled(self, state: Dict[str, object]) -> Optional[int]:
        if (
            float(state["confidence"]) >= self.config.pseudo_confidence
            and float(state["agreement"]) >= self.config.pseudo_agreement
        ):
            pseudo_label = int(state["prediction"])
            self.memory.add_pseudo(np.asarray(state["feature"]), pseudo_label)
            return pseudo_label
        return None

    def reveal_label(self, state: Dict[str, object], label: int) -> None:
        probs = np.asarray(state["expert_probs"])
        losses = -np.log(np.maximum(probs[:, label], EPS))
        self.weights *= np.exp(-self.config.weight_learning_rate * losses)
        self.weights /= np.maximum(self.weights.sum(), EPS)
        rho = self.config.weight_floor_mixing
        self.weights = (1.0 - rho) * self.weights + rho / len(self.weights)
        self.weights /= self.weights.sum()
        self.memory.add_label(np.asarray(state["feature"]), int(label))

    def audit_state(self) -> Dict[str, object]:
        return {
            "weights": self.weights.tolist(),
            "labeled_count": self.memory.labeled_count.tolist(),
            "pseudo_count": self.memory.pseudo_count.tolist(),
        }
