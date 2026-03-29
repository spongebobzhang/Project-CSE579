from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from multiplexrag.eval_utils import mrr_at_k


ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\d")


def query_features(text: str) -> dict[str, float]:
    tokens = text.split()
    unique_tokens = {token.lower() for token in tokens}
    return {
        "char_len": float(len(text)),
        "token_len": float(len(tokens)),
        "avg_token_len": float(sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0,
        "unique_ratio": float(len(unique_tokens) / len(tokens)) if tokens else 0.0,
        "digit_count": float(len(NUMBER_RE.findall(text))),
        "acronym_count": float(len(ACRONYM_RE.findall(text))),
        "has_question_mark": float("?" in text),
    }


def features_matrix(queries: list[dict]) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "char_len",
        "token_len",
        "avg_token_len",
        "unique_ratio",
        "digit_count",
        "acronym_count",
        "has_question_mark",
    ]
    rows = []
    for query in queries:
        feats = query_features(query["text"])
        rows.append([feats[name] for name in feature_names])
    return np.asarray(rows, dtype=np.float32), feature_names


def best_mode_for_query(results_by_mode: dict[str, list[str]], relevant: set[str], k: int = 10) -> str:
    scored = {
        mode: mrr_at_k(doc_ids, relevant, k)
        for mode, doc_ids in results_by_mode.items()
    }
    return max(scored.items(), key=lambda item: (item[1], item[0] == "hybrid"))[0]


class QueryRouter:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.centroids: dict[str, np.ndarray] = {}

    def fit(self, queries: list[dict], labels: list[str]) -> "QueryRouter":
        x, feature_names = features_matrix(queries)
        self.feature_names = feature_names
        self.centroids = {}
        label_set = sorted(set(labels))
        for label in label_set:
            mask = np.asarray([item == label for item in labels], dtype=bool)
            self.centroids[label] = x[mask].mean(axis=0)
        return self

    def predict(self, queries: list[dict]) -> list[str]:
        x, _ = features_matrix(queries)
        preds = []
        for row in x:
            best_label = min(
                self.centroids.items(),
                key=lambda item: np.linalg.norm(row - item[1]),
            )[0]
            preds.append(best_label)
        return preds

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump({"centroids": self.centroids, "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str | Path) -> "QueryRouter":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.centroids = payload["centroids"]
        obj.feature_names = payload["feature_names"]
        return obj
