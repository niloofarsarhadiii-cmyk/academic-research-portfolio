"""
Contextual classification of French discourse markers
=====================================================

Exploratory NLP / introductory machine-learning project.

Compares three models on a small, hand-constructed dataset:

1. Majority-class baseline
2. Marker-only model          (input: the marker string, e.g. "en revanche")
3. Context-aware model        (input: the sentence with the marker replaced
                               by [MARQUEUR])

Features: TF-IDF. Classifier: logistic regression.

Three evaluations are run (see the README for why):

A. One stratified train/test split          -> easy to read, but unstable
B. Repeated stratified cross-validation     -> more stable, but the SAME markers
                                               appear in train and test
C. Leave-one-marker-out cross-validation    -> every test marker is UNSEEN
                                               in training

IMPORTANT: the data are constructed teaching examples. The numbers produced
here are a coding demonstration, not research findings.

Usage:
    python classifier.py
    python classifier.py --data data/french_discourse_markers.csv --seed 7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

LABELS = ["consequence", "contrast", "reformulation"]
PLACEHOLDER = "[MARQUEUR]"
DEFAULT_DATA = Path("data/french_discourse_markers.csv")
RESULTS_DIR = Path("results")

# Keeps one-letter tokens such as the "l" and "d" in l'équipe / d'autres.
TOKEN_PATTERN = r"(?u)\b\w+\b"

DISCLAIMER = (
    "NOTE: constructed teaching examples, tiny sample, provisional labels.\n"
    "These metrics are a coding demonstration, NOT reliable research findings."
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def mask_marker(sentence: str, marker: str) -> str:
    """Replace the first occurrence of `marker` in `sentence` with [MARQUEUR]."""
    pattern = re.compile(
        r"(?<!\w)" + re.escape(marker) + r"(?!\w)", flags=re.IGNORECASE
    )
    masked, n_replacements = pattern.subn(PLACEHOLDER, sentence, count=1)
    if n_replacements == 0:
        raise ValueError(f"Marker {marker!r} not found in sentence: {sentence!r}")
    return masked


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the CSV, then add the masked-context column."""
    df = pd.read_csv(path, encoding="utf-8")

    missing = {"sentence", "marker", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    for col in ("sentence", "marker", "label"):
        df[col] = df[col].astype(str).str.strip()

    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in dataset: {unknown}")

    if df.duplicated(subset=["sentence"]).any():
        raise ValueError("Duplicate sentences found; remove them to avoid leakage.")

    df["marker_key"] = df["marker"].str.lower()
    df["context"] = [mask_marker(s, m) for s, m in zip(df["sentence"], df["marker"])]
    return df


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models(seed: int) -> dict[str, tuple[str, Pipeline]]:
    """Return {model name: (input column, sklearn pipeline)}."""

    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=TOKEN_PATTERN,
            sublinear_tf=True,
        )

    def logreg() -> LogisticRegression:
        return LogisticRegression(max_iter=1000, random_state=seed)

    return {
        "Majority baseline": (
            "context",
            Pipeline([("tfidf", tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        ),
        "Marker only": (
            "marker_key",
            Pipeline([("tfidf", tfidf()), ("clf", logreg())]),
        ),
        "Context (marker masked)": (
            "context",
            Pipeline([("tfidf", tfidf()), ("clf", logreg())]),
        ),
    }


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def confusion_table(y_true, y_pred) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    table = pd.DataFrame(
        matrix,
        index=[f"true:{label}" for label in LABELS],
        columns=[f"pred:{label}" for label in LABELS],
    )
    return table.to_string()


def section(title: str, char: str = "=") -> list[str]:
    return ["", char * 72, title, char * 72]


def evaluate_single_split(df, models, seed, test_size) -> list[str]:
    lines = section(f"A. Single stratified split (test_size={test_size}, seed={seed})")
    lines.append("Unstable on small data: change --seed and watch the numbers move.")

    idx_train, idx_test = train_test_split(
        df.index, test_size=test_size, stratify=df["label"], random_state=seed
    )
    y_train, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]
    lines.append(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

    for name, (column, pipe) in models.items():
        pipe.fit(df.loc[idx_train, column], y_train)
        pred = pipe.predict(df.loc[idx_test, column])
        lines += section(name, "-")
        lines.append(
            classification_report(y_test, pred, labels=LABELS, zero_division=0, digits=3)
        )
        lines.append("Confusion matrix:")
        lines.append(confusion_table(y_test, pred))
    return lines


def evaluate_repeated_cv(df, models, seed) -> list[str]:
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=seed)
    lines = section("B. Repeated stratified CV (4 folds x 10 repeats), macro-F1")
    lines.append(
        "Caution: the same markers appear in training and test folds, so the\n"
        "marker-only model can simply memorise marker -> label."
    )
    for name, (column, pipe) in models.items():
        scores = cross_val_score(pipe, df[column], df["label"], cv=cv, scoring="f1_macro")
        lines.append(f"{name:<26} mean={scores.mean():.3f}  sd={scores.std():.3f}")
    return lines


def evaluate_leave_marker_out(df, models) -> tuple[list[str], dict[str, np.ndarray]]:
    lines = section("C. Leave-one-marker-out CV (test markers are UNSEEN in training)")
    lines.append(
        "Harder and more informative: can the model label a marker it has never\n"
        "seen, using only what it learned from other markers?"
    )
    cv = LeaveOneGroupOut()
    predictions: dict[str, np.ndarray] = {}

    for name, (column, pipe) in models.items():
        pred = cross_val_predict(
            pipe, df[column], df["label"], groups=df["marker_key"], cv=cv
        )
        predictions[name] = pred
        macro_f1 = f1_score(df["label"], pred, labels=LABELS, average="macro", zero_division=0)
        lines += section(f"{name}  (pooled macro-F1 = {macro_f1:.3f})", "-")
        lines.append(
            classification_report(df["label"], pred, labels=LABELS, zero_division=0, digits=3)
        )
        lines.append("Confusion matrix:")
        lines.append(confusion_table(df["label"], pred))
    return lines, predictions


def error_analysis(df, predictions, model_name="Context (marker masked)") -> list[str]:
    lines = section(f"Error analysis: {model_name} (leave-one-marker-out)")
    wrong = df[predictions[model_name] != df["label"]].copy()
    wrong["predicted"] = predictions[model_name][wrong.index]
    if wrong.empty:
        lines.append("No errors.")
        return lines

    lines.append(f"{len(wrong)} of {len(df)} examples misclassified.\n")
    by_marker = wrong.groupby("marker_key").size().sort_values(ascending=False)
    lines.append("Errors per marker:")
    lines.append(by_marker.to_string())
    lines.append("\nMisclassified sentences:")
    for _, row in wrong.iterrows():
        lines.append(
            f"- [{row['marker']}] gold={row['label']} pred={row['predicted']}\n"
            f"    {row['sentence']}"
        )
    return lines


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    args = parser.parse_args()

    df = load_dataset(args.data)
    models = build_models(args.seed)

    report: list[str] = ["CONTEXTUAL CLASSIFICATION OF FRENCH DISCOURSE MARKERS", DISCLAIMER]
    report += section("Dataset summary")
    report.append(f"Examples: {len(df)} | Distinct markers: {df['marker_key'].nunique()}")
    report.append("Examples per label:")
    report.append(df["label"].value_counts().reindex(LABELS).to_string())
    report.append("Markers per label:")
    report.append(df.groupby("label")["marker_key"].nunique().reindex(LABELS).to_string())
    report.append(f"\nExample masked input: {df['context'].iloc[0]}")

    report += evaluate_single_split(df, models, args.seed, args.test_size)
    report += evaluate_repeated_cv(df, models, args.seed)
    lmo_lines, predictions = evaluate_leave_marker_out(df, models)
    report += lmo_lines
    report += error_analysis(df, predictions)
    report += ["", DISCLAIMER]

    text = "\n".join(report)
    print(text)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "report.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
