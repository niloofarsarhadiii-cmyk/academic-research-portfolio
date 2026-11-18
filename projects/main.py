
# ==========================================
# FRENCH DISCOURSE MARKER CLASSIFICATION
# Preliminary Experiment
# ==========================================

# 1. Import libraries

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 2. Read the dataset

data = pd.read_csv("data/discourse_markers.csv")


# 3. Separate sentences and labels

X = data["sentence"]
y = data["label"]


# 4. Divide the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
    stratify=y
)


# 5. Convert text into numerical features using TF-IDF

vectorizer = TfidfVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# 6. Create and train the Logistic Regression model

model = LogisticRegression()

model.fit(X_train_vectors, y_train)


# 7. Predict the test examples

predictions = model.predict(X_test_vectors)


# 8. Display actual and predicted labels

print("\nACTUAL TEST LABELS:")
print(y_test.tolist())

print("\nPREDICTED LABELS:")
print(predictions)


# 9. Calculate model accuracy

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY:")
print(accuracy)


# 10. Examine classification errors

print("\nERROR ANALYSIS:")

for sentence, actual, predicted in zip(
    X_test, y_test, predictions
):
    if actual != predicted:
        print("\nSentence:", sentence)
        print("Actual function:", actual)
        print("Predicted function:", predicted)


# 11. Display training and testing sentences

print("\nTRAINING SENTENCES:")
print(X_train.tolist())

print("\nTESTING SENTENCES:")
print(X_test.tolist())


# 12. Create the results folder

results_folder = Path("results")

results_folder.mkdir(exist_ok=True)


# 13. Save test predictions in a CSV file

results = pd.DataFrame({
    "sentence": X_test.tolist(),
    "actual_label": y_test.tolist(),
    "predicted_label": predictions,
    "correct": y_test.to_numpy() == predictions
})

results.to_csv(
    results_folder / "test_predictions.csv",
    index=False
)


# 14. Create and save the confusion matrix

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues")

plt.title("French Discourse Marker Classification")
plt.tight_layout()

plt.savefig(
    results_folder / "confusion_matrix.png",
    dpi=300
)

plt.show()


# 15. Create the evaluation summary

summary = f"""
FRENCH DISCOURSE MARKER CLASSIFICATION
Preliminary Experiment

DATASET
Dataset size: {len(data)}
Training examples: {len(X_train)}
Testing examples: {len(X_test)}

METHODOLOGY
Model: Logistic Regression
Text representation: TF-IDF
Train/test ratio: 50/50
Random state: 42

RESULTS
Actual test labels:
{y_test.tolist()}

Predicted test labels:
{predictions.tolist()}

Accuracy: {accuracy:.4f}

ERROR ANALYSIS
"""

for sentence, actual, predicted in zip(
    X_test, y_test, predictions
):
    if actual != predicted:
        summary += f"""
Sentence: {sentence}
Actual function: {actual}
Predicted function: {predicted}
"""


summary += """

LIMITATIONS

This is a small exploratory experiment using six
constructed French examples.

The test set contains only three sentences.
Therefore, the accuracy score should not be
interpreted as evidence of general model performance.

Further research would require a larger dataset
and carefully annotated examples.
"""


# 16. Save the evaluation summary

with open(
    results_folder / "evaluation_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)


# 17. Confirm that results have been saved

print("\nRESULTS SAVED IN THE RESULTS FOLDER.")

print("\nFILES CREATED:")

for file in results_folder.iterdir():
    print(file.name)
