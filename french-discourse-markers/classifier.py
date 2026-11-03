"""
Exploratory classifier for French discourse-marker functions.

The bundled data are constructed examples for learning and code testing,
not a validated research corpus. Do not interpret the metrics as research findings.
"""
from pathlib import Path
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "discourse_markers.csv"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def make_context(sentence: str, marker: str) -> str:
    """Replace the first occurrence of the target marker to retain its context."""
    lower_sentence = sentence.lower()
    lower_marker = marker.lower()
    position = lower_sentence.find(lower_marker)
    if position == -1:
        raise ValueError(f"Marker {marker!r} not found in sentence: {sentence!r}")
    return sentence[:position] + " [MARQUEUR] " + sentence[position + len(marker):]


def make_marker_text(marker: str) -> str:
    """Return a marker-only input for the comparison baseline."""
    return f"marqueur {marker}"


def evaluate(name, model, x_train, x_test, y_train, y_test, labels):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Accuracy: {accuracy:.3f}")
    print(report)
    print("Confusion matrix (rows = true labels, columns = predicted labels):")
    print(matrix)
    return report, matrix, accuracy


def main():
    data = pd.read_csv(DATA_PATH)
    required = {"sentence", "marker", "function"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data = data.dropna(subset=["sentence", "marker", "function"]).copy()
    data["context"] = [
        make_context(sentence, marker)
        for sentence, marker in zip(data["sentence"], data["marker"])
    ]
    data["marker_input"] = data["marker"].apply(make_marker_text)

    print(f"Loaded {len(data)} examples.")
    print("\nLabel counts:")
    print(data["function"].value_counts())

    labels = sorted(data["function"].unique())
    x_context = data["context"]
    x_marker = data["marker_input"]
    y = data["function"]

    # Stratification preserves class proportions. With a tiny dataset this
    # is still only a demonstration; a larger source-based split is preferable.
    x_context_train, x_context_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            x_context, y, data.index,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )
    x_marker_train = x_marker.loc[idx_train]
    x_marker_test = x_marker.loc[idx_test]

    dummy = DummyClassifier(strategy="most_frequent")
    marker_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    context_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    RESULTS_DIR.mkdir(exist_ok=True)

    dummy.fit(x_context_train, y_train)
    dummy_predictions = dummy.predict(x_context_test)
    dummy_report = classification_report(
        y_test, dummy_predictions, labels=labels, zero_division=0
    )
    marker_report, marker_matrix, marker_accuracy = evaluate(
        "Marker-only model", marker_model,
        x_marker_train, x_marker_test, y_train, y_test, labels
    )
    context_report, context_matrix, context_accuracy = evaluate(
        "Context-aware model (marker masked)", context_model,
        x_context_train, x_context_test, y_train, y_test, labels
    )

    output = [
        "EXPLORATORY RESULTS — CONSTRUCTED STARTER DATASET",
        f"Examples: {len(data)}",
        f"Random state: {RANDOM_STATE}",
        "",
        "WARNING: This tiny constructed dataset is for code practice only.",
        "Do not present these metrics as evidence of general model performance.",
        "",
        "Majority-class baseline:",
        dummy_report,
        "",
        "Marker-only model:",
        f"Accuracy: {marker_accuracy:.3f}",
        marker_report,
        "",
        "Context-aware model (marker masked):",
        f"Accuracy: {context_accuracy:.3f}",
        context_report,
        "",
        "Interpretation: compare the models cautiously. The data are small and",
        "constructed; a larger authentic corpus and source-aware evaluation are needed."
    ]
    report_path = RESULTS_DIR / "evaluation_report.txt"
    report_path.write_text("\n".join(output), encoding="utf-8")
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
