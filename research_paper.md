# A Preliminary Computational Approach to the Classification of French Discourse Markers

### An Exploratory Study Using TF-IDF and Logistic Regression

## Abstract

This study explores the use of basic machine-learning techniques to classify French sentences containing selected discourse markers according to predefined communicative functions. The project examines three French expressions: *mais*, *donc*, and *en fait*, which were assigned the categories contrast, consequence, and reformulation, respectively, in the experimental dataset.

A small dataset of six constructed French sentences was developed and processed using Python. The sentences were represented numerically using Term Frequency-Inverse Document Frequency (TF-IDF), and a Logistic Regression classifier was trained using three examples and evaluated on three held-out examples. The model correctly classified two of the three test sentences, achieving an accuracy of 66.7%.

The results are preliminary and cannot establish the general effectiveness of this classification approach. The small dataset, constructed examples, and limited representation of discourse-marker functions restrict the conclusions that can be drawn. In particular, each marker was associated with only one functional category, meaning that the experiment did not test how the same marker might function differently across contexts. Nevertheless, the project provides an initial demonstration of a computational text-classification workflow and highlights some of the challenges involved in representing linguistic functions through word-based textual features. The study concludes by identifying directions for future research, particularly the use of larger authentic datasets and greater attention to linguistic context.

**Keywords:** computational linguistics, French discourse markers, text classification, TF-IDF, Logistic Regression, machine learning

## 1. Introduction

Language is not simply a collection of words and grammatical structures. Speakers use linguistic expressions to establish relationships between ideas, organize information, express attitudes, and guide the interpretation of discourse. Some expressions help connect different parts of an utterance and indicate how one proposition relates to another. These expressions are commonly studied as discourse markers.

Discourse markers have been examined from different linguistic perspectives, including discourse analysis, pragmatics, and sociolinguistics. In her foundational study, Schiffrin (1987) examines how discourse markers contribute to the organization and interpretation of conversational interaction. Her approach emphasizes that the function of a marker cannot always be understood by examining the expression in isolation; its role must also be considered in relation to the surrounding discourse. Although Schiffrin's study focuses primarily on English, her approach provides a useful theoretical starting point for examining discourse markers in other languages. It establishes an important distinction between identifying a linguistic expression and understanding the role it performs within an interaction.

Research on spoken French offers a more direct connection to the present study. Crible and Degand (2021) investigate the co-occurrence and ordering of discourse markers in spoken French, examining how syntactic, prosodic, and pragmatic factors influence the ways markers appear together and are organized. Their research is relevant because it demonstrates that the analysis of discourse markers involves relationships between linguistic elements, rather than simply identifying individual expressions. Similarly, Šišić (2024), using examples from the French spoken-language corpus *Les Vocaux*, examines eleven discourse markers, including *mais*, *en fait*, and *donc*. His analysis considers their roles in the organization of spoken communication and their contribution to the development of illocutionary force. Together, these studies provide a linguistic foundation for examining the three expressions selected for the present experiment.

These findings raise an important question for computational linguistics: can the function of a discourse marker be identified from the words in a sentence, or does classification also require a closer understanding of its linguistic context? The expression *donc*, for example, can indicate an inference or consequence, but its use is not limited to this function. Vlemings (2003) examines *donc* in French imperative sentences and demonstrates that it can function as a discourse marker without necessarily linking an antecedent to a logical consequence. This example illustrates a central challenge for computational classification: a model that associates each discourse marker with only one predefined function may fail to account for other uses of the same expression in different linguistic contexts. It also suggests that the relationship between a linguistic form and its communicative function deserves closer attention when designing computational classification tasks.

My interest in this question comes from my academic background in French literature, linguistics, and language teaching. Studying language from literary and pedagogical perspectives led me to become interested in how meaning is constructed and interpreted across different contexts. As I began exploring programming and computational linguistics, I became interested in how computational methods could contribute to the study of linguistic patterns and whether they could help investigate questions traditionally approached through linguistic analysis.

This project represents an initial attempt to connect these interests through a small text-classification experiment. Using Python and machine-learning techniques, I developed a preliminary classifier to examine whether French sentences containing selected discourse markers could be assigned to predefined functional categories. The project approaches a linguistic question through a computational method: it examines whether numerical representations of sentences can provide information that allows a machine-learning model to distinguish among selected categories.

The experiment focuses on three functional categories: contrast, consequence, and reformulation. It uses Term Frequency-Inverse Document Frequency (TF-IDF) to represent sentences numerically and Logistic Regression to classify them. The functional categories are predefined labels assigned to the constructed examples for this experiment; they are not intended to represent the complete range of possible functions of the three expressions.

The central question is whether a basic machine-learning model can use the textual features of French sentences to distinguish among these predefined categories. By examining its predictions and errors, the study explores the relationship between word-based textual representations and linguistic classification. Rather than attempting to develop a comprehensive model of French discourse, this project serves as an exploratory investigation of a computational approach and its limitations. It also considers how future research could incorporate larger authentic datasets and more detailed representations of linguistic context.

## 2. Research Question and Objectives

### 2.1 Research Question

The central research question guiding this experiment is:

**How can a machine-learning model use the textual features of French sentences to classify the functions of selected discourse markers?**

The study examines whether a basic text-classification model can distinguish among three predefined functional categories: contrast, consequence, and reformulation.

The question is exploratory. It does not assume that the model can identify every possible function of a discourse marker or interpret linguistic context in the same way as a human researcher. Instead, it examines how a basic machine-learning model processes textual features and predicts categories assigned to a small set of French sentences.

### 2.2 Research Objectives

The experiment has four main objectives:

1. To create a small dataset of French sentences containing selected discourse markers.
2. To represent the sentences numerically using TF-IDF.
3. To train a Logistic Regression classifier to predict the functional category assigned to each sentence.
4. To evaluate the model's predictions and examine its classification errors.

### 2.3 Scope of the Study

This study focuses on a limited set of French examples and three functional categories. It does not attempt to provide a comprehensive account of discourse-marker functions or develop a general-purpose French language classifier.

Each selected marker was assigned one functional category in the constructed dataset. Therefore, the experiment does not examine whether the same marker can express different functions depending on its context.

Instead, the study investigates a basic computational approach and considers what its results reveal about the possibilities and limitations of applying machine-learning methods to linguistic data.

## 3. Background

### 3.1 French Discourse Markers

Discourse markers are linguistic expressions that contribute to the organization and interpretation of communication. They can signal relationships between propositions, indicate how information is connected, and help listeners or readers follow the development of an argument or conversation.

Schiffrin (1987) provides a foundational account of discourse markers as elements that contribute to conversational organization and interpretation. Her work emphasizes the importance of examining how linguistic expressions operate within discourse rather than treating them as isolated vocabulary items.

Research on spoken French provides further insight into these relationships. Crible and Degand (2021) examine the co-occurrence and ordering of discourse markers, showing the relevance of syntactic, prosodic, and pragmatic factors in the analysis of spoken discourse. Šišić (2024) also examines the use of French discourse markers in voice messages, including *mais*, *donc*, and *en fait*. These studies demonstrate the importance of examining discourse markers in relation to their linguistic and communicative environments.

In this experiment, the three selected expressions are associated with the following functional categories:

- **Contrast:** *mais* is assigned the function of introducing an opposition or contrast between ideas.
- **Consequence:** *donc* is assigned the function of indicating a result or consequence.
- **Reformulation:** *en fait* is assigned the function of introducing a clarification or reformulation in the constructed examples.

These categories provide the linguistic basis for the classification task. However, they represent the labels used in this particular dataset rather than fixed definitions of the expressions.

The distinction is important because the relationship between a discourse marker and its function is not necessarily fixed. Vlemings (2003), for example, examines the use of *donc* in French imperative sentences and illustrates that its discourse function cannot always be reduced to a simple relationship between an antecedent and a consequence. The interpretation of an expression may vary according to its context and use.

This observation provides an important motivation for the present experiment: computational classification should be examined not only in terms of whether a model recognizes textual patterns, but also in terms of how those patterns relate to the linguistic functions assigned to them.

### 3.2 Computational Approaches to Text Classification

Text classification is a natural language processing task in which a computer model assigns predefined categories to textual data. It can be used in different applications, including sentiment analysis, topic identification, and document categorization.

For a machine-learning model to process written language, text must first be represented in a numerical format. One common approach is Term Frequency-Inverse Document Frequency (TF-IDF), which represents documents according to the importance of words within a collection of texts.

The term-frequency component reflects how often a word occurs in a document, while the inverse-document-frequency component gives less weight to terms that appear frequently across the collection. This produces numerical features that can be used by machine-learning algorithms. Term-weighting methods such as TF-IDF have a long history in information retrieval and automatic text processing (Salton & Buckley, 1988).

In this project, TF-IDF provides a way to convert French sentences into numerical representations that can be processed by a classification algorithm.

### 3.3 Logistic Regression for Text Classification

Logistic Regression is a statistical method that can be used for classification. In a text-classification task, it uses numerical features extracted from text to estimate the probability that an example belongs to a particular category.

In this experiment, Logistic Regression was selected as a basic supervised-learning algorithm. It was trained on the TF-IDF representations of the training sentences and their assigned labels.

The purpose of using this method was to establish a straightforward classification workflow that could be implemented and examined with a small dataset. The experiment was not designed to compare multiple algorithms or determine which model performs best for French discourse-marker classification.

## 4. Methodology

### 4.1 Research Design

This project used an exploratory quantitative approach to examine the classification of French discourse markers through machine learning.

A small dataset of six constructed French sentences was developed. Each sentence was assigned a functional label: contrast, consequence, or reformulation.

The experiment followed a basic supervised machine-learning workflow. The dataset was divided into training and testing subsets. A Logistic Regression classifier was trained using numerical text representations generated through TF-IDF. Its predictions were then compared with the assigned labels in the test subset.

The experiment was implemented in Python using Pandas for data handling and scikit-learn for text representation, classification, and evaluation. Matplotlib was used to produce a visual representation of the classification results.

### 4.2 Dataset Construction

The dataset consisted of six French sentences created specifically for this experiment. It included three discourse markers:

- *mais*
- *donc*
- *en fait*

Each marker appeared in two sentences and was associated with one predefined functional category.

The dataset was organized in a CSV file named `discourse_markers.csv`, containing three columns:

| Column | Description |
|---|---|
| sentence | The complete French sentence |
| marker | The discourse marker examined |
| label | The assigned functional category |

The six constructed examples were:

| Discourse marker | Example sentence | Assigned label |
|---|---|---|
| *mais* | Je voulais sortir, mais il pleut. | Contrast |
| *mais* | Il est gentil, mais parfois impatient. | Contrast |
| *donc* | Il pleut, donc je reste chez moi. | Consequence |
| *donc* | Elle étudie beaucoup, donc elle réussit. | Consequence |
| *en fait* | En fait, je préfère rester ici. | Reformulation |
| *en fait* | En fait, nous avons déjà terminé. | Reformulation |

Because these sentences were constructed for the experiment rather than collected from an authentic corpus, the dataset represents a controlled set of examples and does not capture the full variation of naturally occurring French discourse.

The dataset was divided into three examples for training and three for testing, using a 50/50 split. Stratification was applied to maintain the representation of all three categories in both subsets. A random state of 42 was specified to make the split reproducible.

### 4.3 Text Representation: TF-IDF

Before classification, the French sentences were converted into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF).

TF-IDF assigns numerical weights to words based on their frequency within a document and their distribution across the collection of documents. This allows a machine-learning algorithm to process textual information as numerical features.

In this experiment, the TF-IDF vectorizer was fitted on the training sentences. The same fitted vectorizer was then used to transform the test sentences. This ensured that the vocabulary and weighting information learned from the training data were also used when representing the test examples.

The model received the complete sentence as its textual input. Although the CSV file also contained a column identifying the discourse marker, the classification features were generated from the sentence column rather than from the marker column.

This distinction is important because the experiment examined whether the textual features of the complete sentences could be used to predict their assigned categories.

### 4.4 Classification Algorithm: Logistic Regression

Logistic Regression was selected as the classification algorithm.

The model was trained using the TF-IDF representations of the training sentences and their corresponding functional labels.

After training, the classifier generated predictions for the three test sentences. Each prediction represented one of the three predefined categories: contrast, consequence, or reformulation.

Because the dataset contained only one training example for each category, the model had very limited information from which to learn the associations between textual features and functional labels.

### 4.5 Evaluation

The model was evaluated by comparing its predicted labels with the actual labels assigned to the test sentences.

Accuracy was used as the primary evaluation metric. It measures the proportion of correct predictions among all predictions.

A confusion matrix was also generated to examine the relationship between actual and predicted categories and identify classification errors.

In addition, an error analysis was conducted to identify sentences for which the predicted category differed from the assigned label.

Given the small size of the test set, the evaluation was treated as a preliminary description of the model's performance on these three examples rather than a reliable estimate of its performance on unseen French texts.

## 5. Results

### 5.1 Classification Accuracy

The Logistic Regression classifier was evaluated on three test sentences. Of these, two were correctly classified and one was incorrectly classified.

The model achieved an accuracy of 66.7%, corresponding to two correct predictions out of three.

This result describes the performance of the classifier on the selected test examples. Because the test set contained only three sentences, the accuracy score should be interpreted cautiously.

### 5.2 Individual Predictions

Table 1 presents the actual and predicted labels for each test sentence.

**Table 1. Classification results**

| Test sentence | Actual label | Predicted label | Result |
|---|---|---|---|
| Il est gentil, mais parfois impatient. | Contrast | Contrast | Correct |
| En fait, je préfère rester ici. | Reformulation | Reformulation | Correct |
| Il pleut, donc je reste chez moi. | Consequence | Contrast | Incorrect |

The model correctly classified the sentences containing *mais* and *en fait*. However, it incorrectly assigned the consequence category to the contrast category for the sentence containing *donc*.

### 5.3 Confusion Matrix

The confusion matrix provides a more detailed view of the model's predictions.

**Table 2. Confusion matrix**

| Actual / Predicted | Consequence | Contrast | Reformulation |
|---|---:|---:|---:|
| Consequence | 0 | 1 | 0 |
| Contrast | 0 | 1 | 0 |
| Reformulation | 0 | 0 | 1 |

The diagonal entries represent correct predictions, while the non-diagonal entries represent classification errors.

The matrix shows that the model correctly classified one contrast example and one reformulation example. However, it incorrectly classified the consequence example as contrast.

Because the test set contained only three sentences, this matrix provides a description of the predictions in this experiment rather than a reliable measure of the model's general classification ability.

### 5.4 Error Analysis

The classification error occurred in the sentence:

*Il pleut, donc je reste chez moi.*

The actual label assigned to this sentence was consequence, but the model predicted contrast.

One possible explanation is the limited amount of training information available to the classifier. The model had access to only one training sentence containing *donc*. Therefore, it had very little information from which to learn the textual patterns associated with the consequence category.

However, the experiment does not provide enough evidence to determine the precise reason for this error. The incorrect prediction could be related to the limited training data, the numerical representation of the sentences, or the characteristics of the classification algorithm.

The error should therefore be understood as an observation from this preliminary experiment rather than evidence of a specific weakness in the model.

## 6. Discussion

The purpose of this experiment was to explore how a basic machine-learning model could be used to classify French sentences containing selected discourse markers according to predefined functional categories.

The results show that the Logistic Regression classifier correctly classified two of the three test sentences. Although the accuracy of 66.7% provides a numerical summary of the experiment, it should be interpreted cautiously because of the extremely small dataset. A single incorrect prediction represents one-third of the test set and therefore has a substantial effect on the reported accuracy.

One observation concerns the relationship between individual discourse markers and their assigned categories. In the constructed dataset, each marker was consistently associated with one functional label. As a result, the model was not required to distinguish between different functions of the same marker across different contexts.

This is an important limitation from a linguistic perspective. Previous research has emphasized that discourse markers should be examined in relation to their surrounding discourse and communicative environments (Schiffrin, 1987; Crible & Degand, 2021). Research on French also illustrates that the functions of expressions such as *donc* cannot always be reduced to a single relationship between propositions (Vlemings, 2003). The present experiment therefore addresses only a simplified version of a more complex linguistic problem.

The classification error involving *donc* also illustrates the difficulty of drawing conclusions from limited training examples. The model had access to only one training sentence containing this marker. Its incorrect prediction suggests that the available textual features and training information were insufficient to classify this particular example correctly. However, the experiment cannot determine the precise reason for the error.

Another important consideration is the use of TF-IDF. This method provides a practical way to represent textual data numerically, making it suitable for a first classification experiment. However, TF-IDF represents text through weighted word features and does not explicitly encode the full semantic and discourse relationships between parts of a sentence (Salton & Buckley, 1988).

The experiment therefore highlights a distinction between identifying patterns in textual data and analyzing linguistic meaning. While machine-learning models can learn associations between textual features and predefined labels, these associations do not necessarily demonstrate an understanding of the linguistic functions being classified.

It is also important to distinguish between what the experiment tested and what it did not test. The classifier received complete sentences rather than isolated discourse markers. However, no comparison was conducted between a marker-only model and a full-sentence model. Consequently, the experiment cannot establish whether the surrounding context improved classification performance.

Overall, this project provided an initial opportunity to connect linguistic questions with computational methods. It also helped identify methodological questions that could guide a more substantial investigation of French discourse markers in future work.

## 7. Limitations

Several limitations should be considered when interpreting the findings of this experiment.

First, the dataset consisted of only six constructed French sentences. This limited the amount of linguistic variation available to the model and restricted its ability to learn meaningful patterns. Because the sentences were created specifically for the experiment, they do not represent the full range of naturally occurring French discourse.

Second, the dataset contained only three discourse markers: *mais*, *donc*, and *en fait*. Each marker was associated with only one functional category. Consequently, the experiment could not investigate whether the same discourse marker might perform different functions in different contexts. This is particularly important because previous linguistic research has demonstrated the importance of examining discourse markers in relation to their use and surrounding discourse (Schiffrin, 1987; Vlemings, 2003).

Third, the training and testing subsets each contained only three sentences. Each category was represented by one training example and one test example. With such a small test set, one incorrect prediction substantially affected the overall accuracy score. The reported accuracy of 66.7% therefore cannot be considered a reliable estimate of performance on unseen French texts.

Fourth, the experiment used TF-IDF, which represents textual information through word-based numerical features. Although useful for a preliminary classification task, this representation does not explicitly capture deeper semantic relationships or the broader discourse context.

Finally, the experiment used only one classification algorithm and did not compare alternative models or text-representation methods. It also did not test whether using the complete sentence provided an advantage over using the discourse marker alone.

These limitations mean that the findings should be interpreted as observations from an initial programming and research experiment rather than evidence of a reliable French discourse-marker classification system.

## 8. Future Research

Future research could extend this experiment in several directions.

First, a larger dataset could be developed using authentic French texts from carefully selected sources. Spoken-language corpora would be particularly relevant because discourse markers frequently occur in conversational interaction. A larger dataset would provide greater linguistic variation and more examples for model training and evaluation.

Second, the annotation scheme could be expanded to include multiple functions for individual discourse markers. For example, *donc* could be examined in sentences where it expresses a consequence, an inference, or another discourse-related function. This would make it possible to investigate how the interpretation of an expression changes according to its linguistic context.

Third, different text-representation methods could be compared. A future study could examine whether contextual language representations capture linguistic distinctions that a basic TF-IDF approach may overlook. Such a comparison could help investigate the extent to which different computational representations are suitable for studying discourse-related meaning.

Another possible direction would be to compare a model trained on the complete sentence with one trained only on the discourse marker itself. Such an experiment could help investigate whether surrounding linguistic information contributes to classification performance. This comparison would directly address one of the questions raised by the present study.

Finally, future work could explore more advanced classification methods while maintaining careful attention to linguistic interpretation and annotation quality. The use of more sophisticated models would need to be accompanied by a sufficiently large and carefully annotated dataset to allow meaningful evaluation.

These directions would help develop the present exploratory project into a more systematic investigation of the relationship between discourse, context, and computational language analysis.

## 9. Conclusion

This study explored the use of machine-learning techniques to classify selected French sentences containing discourse markers according to predefined functional categories. Using a small dataset of six constructed sentences, TF-IDF text representation, and a Logistic Regression classifier, the experiment provided an initial demonstration of computational text classification applied to a linguistic question.

The model correctly classified two of the three test sentences, achieving an accuracy of 66.7%. Although this result is limited by the size and design of the dataset, the experiment made it possible to examine the classification process, identify an error, and consider some of the challenges involved in computationally representing linguistic functions.

One of the main observations from this project is that identifying a discourse marker is not necessarily equivalent to identifying its function. Previous linguistic research has emphasized the importance of discourse context in understanding how markers operate (Schiffrin, 1987; Crible & Degand, 2021). The present experiment approaches this issue from a computational perspective by examining whether word-based textual features can be used to predict predefined functional categories.

The results do not demonstrate that the classifier understands discourse functions or that contextual information improves its predictions. Instead, they illustrate the possibilities and limitations of a basic classification workflow applied to a small set of constructed examples. The experiment also identifies questions that could be investigated through larger datasets, more varied functional categories, and comparisons between different text-representation methods.

Beyond the classification results, this project provided an opportunity to connect my background in French literature, linguistics, and language studies with programming and machine learning. It represents an initial step in exploring how computational methods can support the study of language and discourse.

Ultimately, the experiment demonstrates the value of beginning with a small, clearly defined research question, testing a computational approach, examining its limitations, and using those observations to develop more focused questions for future research.

## References

Crible, L., & Degand, L. (2021). Co-occurrence and ordering of discourse markers in sequences: A multifactorial study in spoken French. *Journal of Pragmatics, 177*, 18–28. https://doi.org/10.1016/j.pragma.2021.02.006

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

Schiffrin, D. (1987). *Discourse markers*. Cambridge University Press. https://doi.org/10.1017/CBO9780511611841

Šišić, E. (2024). The use of discourse markers in voice messages based on selected examples from the corpus *Les Vocaux*. *Govor, 41*(2), 179–202. https://doi.org/10.22210/govor.2024.41.09

Vlemings, J. (2003). The discourse use of French *donc* in imperative sentences. *Journal of Pragmatics, 35*(7), 1095–1112. https://doi.org/10.1016/S0378-2166(03)00024-9

scikit-learn developers. (n.d.). *TfidfVectorizer*. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html