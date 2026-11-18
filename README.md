# French Discourse Marker Classification

### A Preliminary Computational Approach Using TF-IDF and Logistic Regression
## Project Overview

I started this project to explore how computational methods can be used to study language.

My background is in French literature, linguistics, and language teaching. I became interested in how words and expressions can communicate different relationships depending on their context.

For this project, I focused on French discourse markers such as *mais*, *donc*, and *en fait*. These expressions can help connect ideas, show contrast, express consequences, or introduce reformulations.

I wanted to take a first step toward combining my background in language studies with programming and computational linguistics.
## Research Question

How can a machine learning model use the words in a French sentence to classify the function of a discourse marker?

More specifically, this project explores whether a basic text-classification model can distinguish among three functions:

- **Contrast:** expressing opposition between ideas.
- **Consequence:** expressing a result or consequence.
- **Reformulation:** presenting or clarifying an idea in another way.

The goal was to build a simple working classification model and understand its possibilities and limitations.
## Dataset

The experiment used a small dataset of six French sentences created for this preliminary study.

Each sentence included:

- The complete French sentence.
- The discourse marker being examined.
- The function assigned to that marker.

The dataset contained three discourse markers:

- *mais* — contrast
- *donc* — consequence
- *en fait* — reformulation

Each function was represented by two sentences.

The examples were constructed for practice and experimentation. They were not collected from an authentic French-language corpus.

The dataset was stored in a CSV file and processed using Python and Pandas.

## Tools and Methodology

The project was developed in Python using PyCharm.

The following libraries and methods were used:

- **Pandas:** to load and examine the dataset.
- **Matplotlib:** to visualize the distribution of discourse markers and classification results.
- **Scikit-learn:** to build and evaluate the machine learning model.

The classification process involved several steps:

1. Loading the French sentences from a CSV file.
2. Separating the sentences from their assigned labels.
3. Converting the text into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF).
4. Dividing the dataset into training and testing sets.
5. Training a Logistic Regression classifier.
6. Using the trained model to predict the functions of unseen test sentences.
7. Comparing the predictions with the actual labels to evaluate the results.

This process allowed me to explore a basic text-classification workflow and gain practical experience with Python and machine learning.
## Results

The dataset contained six French sentences. Three were used for training and three for testing.

The Logistic Regression model correctly classified two of the three test sentences.

The results were:

- **Accuracy:** 66.7%
- **Correct predictions:** 2
- **Incorrect predictions:** 1

The model correctly identified the contrast and reformulation examples. However, it incorrectly classified one sentence expressing consequence as contrast.

The misclassified sentence was:

*Il pleut, donc je reste chez moi.*

- Actual label: consequence
- Predicted label: contrast

This result illustrates some of the challenges of classifying discourse markers using a very small dataset.

Because the test set contained only three sentences, the accuracy score should be interpreted as an observation from a preliminary experiment, not as evidence of the model's general performance.

## Limitations

This project was an initial experiment designed to explore the use of machine learning for French discourse marker classification.

Several limitations should be considered:

- **Small dataset:** The experiment used only six constructed sentences.
- **Limited training data:** The model was trained on only three examples.
- **Limited test data:** The evaluation was based on only three sentences. A single incorrect prediction had a substantial effect on the accuracy score.
- **Constructed examples:** The sentences were created for this experiment rather than collected from an authentic French-language corpus.
- **Limited linguistic coverage:** Only three discourse markers and three functional categories were examined.

These limitations mean that the results cannot establish how well the model would perform on authentic French texts or more complex linguistic contexts.

A larger dataset with carefully annotated examples would be necessary for a more reliable evaluation.
## Conclusion and Future Work

This project gave me an opportunity to apply basic machine-learning techniques to a linguistic question.

By building a French discourse marker classifier, I gained practical experience with Python, data processing, text representation, model training, and evaluation.

The experiment showed that a simple Logistic Regression model could classify some of the examples correctly. However, the small dataset and the incorrect prediction also highlighted the limitations of this approach.

In future work, I would like to:

- Expand the dataset using authentic French texts.
- Include more discourse markers and functional categories.
- Examine how the same discourse marker can express different functions in different contexts.
- Compare models using different types of linguistic information.
- Investigate whether incorporating contextual information can improve classification.

Ultimately, I hope to develop this preliminary project into a more substantial study connecting linguistic analysis with computational methods.

## How to Run the Project

### Requirements

The project uses:

- Python
- Pandas
- Scikit-learn
- Matplotlib

### Installation

Install the required libraries using:

```bash
pip install -r requirements.txt



- `test_predictions.csv`: Contains the test sentences, their actual labels, the model's predicted labels, and whether each prediction was correct.
- `confusion_matrix.png`: Visualizes the classification results and shows which categories were correctly classified or confused.

The evaluation produced two correct predictions and one incorrect prediction, resulting in an accuracy of approximately 66.7%.
