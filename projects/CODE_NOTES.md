# French Discourse Marker Classification
## Code Archive and Learning Notes

This file documents the Python code used in the French Discourse Marker Classification project. It includes the final experiment script, explanations of the Python libraries and methods, and notes about the development of the experiment.

The purpose of this archive is to preserve the code used in the project and explain how each part works. It is also a learning reference for understanding the connection between Python programming, machine learning, and computational linguistics.

---

## 1. Project Setup

The project was developed using Python in PyCharm.

The project uses a small dataset of six constructed French sentences. The goal is to explore whether a basic machine-learning model can classify sentences according to the communicative functions assigned to selected French discourse markers.

### Project structure

```text
french-discourse-markers/
│
├── data/
│   └── discourse_markers.csv
│
├── results/
│   ├── test_predictions.csv
│   ├── confusion_matrix.png
│   └── evaluation_summary.txt
│
├── main.py
├── classifier.py
├── requirements.txt
├── README.md
├── research_paper.md
└── CODE_NOTES.md
```

### Python environment

The experiment was developed in a Python virtual environment.

A virtual environment keeps the libraries used by a project separate from other Python projects on the computer.

### Installing the required libraries

The project uses a `requirements.txt` file containing:

```text
pandas
scikit-learn
matplotlib
```

These libraries can be installed with:

```bash
pip install -r requirements.txt
```

---

## 2. Python Libraries

### 2.1 Importing libraries

```python
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
```

### What does this code do?

A library is a collection of code that programmers can use in their own programs.

This project uses the following libraries and tools:

- `Path`: helps work with file and folder paths.
- `pandas`: loads and organizes data in tables.
- `matplotlib`: creates graphs and visualizations.
- `TfidfVectorizer`: converts text into numerical features.
- `LogisticRegression`: creates a machine-learning classification model.
- `train_test_split`: divides a dataset into training and testing groups.
- `accuracy_score`: calculates the proportion of correct predictions.
- `confusion_matrix`: compares actual and predicted categories.
- `ConfusionMatrixDisplay`: displays the confusion matrix as a graph.

### What does `as pd` mean?

```python
import pandas as pd
```

The expression `as pd` gives pandas a shorter name.

Instead of writing `pandas`, we can write `pd`.

### What does `from ... import ...` mean?

For example:

```python
from sklearn.linear_model import LogisticRegression
```

This imports a specific tool from a larger library.

Importing a library makes its tools available. It does not automatically run the experiment.

---

## 3. Loading the Dataset

### 3.1 Reading the CSV file

```python
data = pd.read_csv("data/discourse_markers.csv")
```

### Explanation

- `pd.read_csv()` reads a CSV file.
- `"data/discourse_markers.csv"` tells Python where the dataset is located.
- `data` is the variable that stores the resulting table.

CSV stands for Comma-Separated Values. It is a common format for storing structured data.

### 3.2 Dataset contents

The dataset contains six constructed French sentences.

| Sentence | Marker | Label |
|---|---|---|
| Il pleut, donc je reste chez moi. | donc | consequence |
| Je voulais sortir, mais il pleut. | mais | contrast |
| En fait, je préfère rester ici. | en fait | reformulation |
| Elle étudie beaucoup, donc elle réussit. | donc | consequence |
| Il est gentil, mais parfois impatient. | mais | contrast |
| En fait, nous avons déjà terminé. | en fait | reformulation |

The dataset has three columns:

- `sentence`: the complete French sentence.
- `marker`: the discourse marker appearing in the sentence.
- `label`: the category assigned to the sentence.

**Important:** These are constructed examples, not sentences collected from an authentic French-language corpus. Each marker is associated with only one category in this experiment.

---

## 4. Selecting the Input and Output Data

### 4.1 Selecting the sentences and labels

```python
X = data["sentence"]
y = data["label"]
```

### Explanation

This code separates the information used by the model.

- `X` contains the French sentences. It is the model's input.
- `y` contains the labels. These are the categories the model is expected to predict.

In machine learning, the input is often called the *features*, while the correct answers are called the *labels* or *target*.

### What is a variable?

A variable is a name used to store information.

For example:

```python
X = data["sentence"]
```

The variable `X` stores the sentence column from the dataset.

### What is a DataFrame?

A DataFrame is a table of data provided by pandas.

The variable `data` contains the dataset as a DataFrame. It has rows and columns, similar to a spreadsheet.

**Important:** Although the CSV contains a `marker` column, this experiment does not use that column as a separate model input. The model receives the complete sentence through `X`.

---

## 5. Splitting the Dataset into Training and Testing Sets

### 5.1 Dividing the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
    stratify=y
)
```

### Explanation

The dataset is divided into two groups:

- Training data: examples used to teach the model.
- Testing data: examples used to evaluate its predictions.

The parameters mean:

- `X`: the input sentences.
- `y`: the correct labels.
- `test_size=0.5`: 50% of the examples are used for testing.
- `random_state=42`: makes the random split reproducible.
- `stratify=y`: helps preserve the proportion of each category in both groups.

Because the dataset contains six sentences, the experiment uses:

- 3 training examples.
- 3 testing examples.

Each category has one training example and one testing example.

### What does reproducible mean?

A reproducible result is one that can be obtained again under the same conditions.

Setting `random_state=42` allows the same train-test split to be generated when the code is run again with the same data and settings.

### Why separate training and testing data?

If a model is evaluated only on the examples it has already seen, we cannot properly examine how it performs on unseen examples.

The testing group provides a small initial check of its predictions on sentences that were not used for training.

---

## 6. Converting Text into Numerical Features with TF-IDF

### 6.1 Creating the TF-IDF vectorizer

```python
vectorizer = TfidfVectorizer()
```

### 6.2 Transforming the sentences

```python
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
```

### Explanation

Machine-learning algorithms generally need numerical input rather than ordinary sentences.

TF-IDF stands for *Term Frequency–Inverse Document Frequency*.

It converts text into numerical values that represent the importance of words in the documents being analyzed.

In this experiment:

- `TfidfVectorizer()` creates the tool that converts text into numerical features.
- `fit_transform(X_train)` learns the vocabulary and calculates the TF-IDF features for the training sentences.
- `transform(X_test)` uses the same learned vocabulary to convert the testing sentences.

### What is the difference between `fit_transform()` and `transform()`?

`fit_transform()` learns from the training data and transforms it.

`transform()` applies what was learned to new data without learning a new vocabulary.

This is important because the testing data should not be used to fit the vectorizer.

### What is a feature?

A feature is a measurable piece of information that a machine-learning model uses to make a prediction.

Here, the TF-IDF values of words in the sentences are the numerical features.

---

## 7. Training the Logistic Regression Model

### 7.1 Creating the model

```python
model = LogisticRegression()
```

### 7.2 Training the model

```python
model.fit(X_train_vectors, y_train)
```

### Explanation

Logistic Regression is a supervised machine-learning algorithm that can be used for classification.

In this project:

- `model` stores the Logistic Regression classifier.
- `X_train_vectors` contains the numerical representations of the training sentences.
- `y_train` contains the correct categories.

The `.fit()` method trains the model using these examples.

The model learns patterns in the numerical features that are associated with the supplied labels.

### What is supervised learning?

Supervised learning is a type of machine learning in which a model learns from examples that already have correct answers.

For this experiment, the model receives French sentences together with their assigned categories.

It then uses the training examples to make predictions about the testing sentences.

---

## 8. Making Predictions

### 8.1 Predicting the testing examples

```python
predictions = model.predict(X_test_vectors)
```

### Explanation

The `.predict()` method asks the trained model to assign a category to each testing sentence.

The resulting predictions are stored in the variable `predictions`.

### 8.2 Displaying actual and predicted labels

```python
print("\nACTUAL TEST LABELS:")
print(y_test.tolist())

print("\nPREDICTED LABELS:")
print(predictions)
```

### Explanation

- `print()` displays information in the Python output.
- `\n` creates a new line.
- `.tolist()` converts the pandas labels into a regular Python list.

This section makes it possible to compare the actual categories with the model's predictions.

---

## 9. Evaluating Model Accuracy

### 9.1 Calculating accuracy

```python
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY:")
print(accuracy)
```

### Explanation

Accuracy measures the proportion of predictions that are correct.

The formula is:

\[
\text{Accuracy} =
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}
\]

In this experiment:

- Total testing sentences: 3
- Correct predictions: 2
- Incorrect predictions: 1

Therefore:

\[
\text{Accuracy} = \frac{2}{3} = 0.6667
\]

The accuracy is approximately **66.7%**.

### Important limitation

The testing set contains only three sentences. One incorrect prediction changes the accuracy by approximately 33.3 percentage points.

Therefore, this result describes the outcome of this small experiment. It should not be interpreted as evidence of reliable performance on general French texts.

---

## 10. Creating the Confusion Matrix

### 10.1 Examining classification errors

```python
print("\nERROR ANALYSIS:")

for sentence, actual, predicted in zip(
    X_test, y_test, predictions
):
    if actual != predicted:
        print("\nSentence:", sentence)
        print("Actual function:", actual)
        print("Predicted function:", predicted)
```

### Explanation

This code examines the predictions one by one.

- `for` repeats an action for each item.
- `zip()` combines corresponding items from the testing sentences, actual labels, and predictions.
- `if actual != predicted` checks whether the actual label differs from the predicted label.

The program prints only the sentences that were classified incorrectly.

### 10.2 Understanding the training and testing sentences

```python
print("\nTRAINING SENTENCES:")
print(X_train.tolist())

print("\nTESTING SENTENCES:")
print(X_test.tolist())
```

This code displays the sentences in each group so that the experiment can be inspected.

### 10.3 Creating the confusion matrix

```python
cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)
```

### Explanation

A confusion matrix is a table that compares the actual categories with the categories predicted by the model.

It helps identify which categories were classified correctly and which were confused with other categories.

### 10.4 Displaying the confusion matrix

```python
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
```

### Explanation

- `ConfusionMatrixDisplay()` prepares the confusion matrix for visualization.
- `.plot()` creates the graph.
- `plt.title()` adds a title.
- `plt.tight_layout()` adjusts the layout.
- `plt.savefig()` saves the graph as an image.
- `plt.show()` displays the graph.

The image is saved in the `results` folder as `confusion_matrix.png`.

### What does DPI mean?

DPI stands for *dots per inch*. It is a measure of image resolution.

The value `dpi=300` creates a high-resolution image suitable for including in a research paper.

---

## 11. Saving the Results

### 11.1 Creating the results folder

```python
results_folder = Path("results")

results_folder.mkdir(exist_ok=True)
```

### Explanation

- `Path("results")` defines the location of the results folder.
- `.mkdir()` creates the folder.
- `exist_ok=True` prevents an error if the folder already exists.

### 11.2 Saving predictions as a CSV file

```python
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
```

### Explanation

This code creates a new table containing the testing results.

It includes:

- The original sentence.
- Its actual category.
- The model's predicted category.
- Whether the prediction was correct.

`pd.DataFrame()` creates a table.

`.to_csv()` saves the table as a CSV file.

`index=False` prevents pandas from adding an extra row-number column.

### 11.3 Creating the evaluation summary

```python
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
```

### Explanation

This section creates a text summary of the experiment.

- `f""" ... """` creates a multiline formatted string.
- `{len(data)}` inserts the number of dataset rows.
- `{accuracy:.4f}` displays accuracy to four decimal places.
- `summary +=` adds more text to the existing summary.

The summary includes information about the dataset, methodology, results, errors, and limitations.

### 11.4 Saving the evaluation summary

```python
with open(
    results_folder / "evaluation_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)
```

### Explanation

- `open()` opens or creates a file.
- `"w"` means write mode.
- `encoding="utf-8"` supports accented French characters.
- `file.write(summary)` writes the summary into the file.
- `with` ensures that the file is properly closed after writing.

### 11.5 Confirming that the results were saved

```python
print("\nRESULTS SAVED IN THE RESULTS FOLDER.")

print("\nFILES CREATED:")

for file in results_folder.iterdir():
    print(file.name)
```

This code displays the names of the files inside the results folder.

`iterdir()` allows Python to go through the items in a folder.

---

## 12. Complete Final Python Script

The following is the complete version of the experiment's `main.py` script.

It brings together the code documented in the previous sections.

```python
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


# 12