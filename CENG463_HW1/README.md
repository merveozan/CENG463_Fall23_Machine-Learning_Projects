
# Meta-Analysis Classification Task

## CENG463 - HOMEWORK 1

### Introduction

In this assignment, the first step is to identify three different topics through Pubmed and create a dataset based on articles. Subsequently, we are required to run this dataset through four different classification algorithms using various parameters. After obtaining the accuracy scores, the task involves preparing a report based on the results.

Firstly, we retrieved the titles and abstracts of 500 articles for each topic (kidney disease, type 2 diabetes, brain injury) from Google Scholar using web scraping techniques with Beautiful Soup. Subsequently, we used these features to create a dataset. Finally, we aimed to achieve the highest test-train accuracy scores by running this dataset through four different classification methods (LogisticRegression, SVC, MultinomialNB, KNeighborsClassifier) with various parameters such as C, alpha, k, gamma. We demonstrated this process using functions.

### Creating the Dataset

In the code, we retrieved titles and abstracts of 500 articles for each topic (kidney disease, type 2 diabetes, brain injury) from PubMed using `Beatiful Soup` for web scraping. When we run the code, a file named `dataset.csv` will be created, representing the dataset we will use for the classifications. Depending on internet and computer speed, this process may take approximately 15-45 minutes. If you do not want to download this file, we will provide the appearance of the functions below since we have created the dataset.

### Creating of the Test and Training Sets

The purpose of the provided code is to perform text classification on a dataset. The code uses the scikit-learn library and other relevant modules to preprocess textual data and apply the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert the text into numerical features. The dataset is loaded from a CSV file (`dataset.csv`), and the `Abstract` column in the dataset is preprocessed by removing special characters, single characters, and multiple spaces.

### Classification Algorithms

We employed four different classification algorithms to evaluate the dataset:
1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Multinomial Naive Bayes
4. K-Nearest Neighbors (KNeighborsClassifier)

Each algorithm was tested with various parameters to determine the best performing model.

### Results

After running the classification algorithms, we obtained accuracy scores for each model. These results were used to prepare a report that compares the performance of each algorithm and discusses the implications of the findings.

### Usage

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/merveozan/CENG463_Fall23_Machine-Learning_Projects/meta-analysis_classification.git
    ```
2. Install the required dependencies:

3. Run the notebook:
    ```bash
    jupyter notebook meta-analysis_classification.ipynb
    ```

### Files

- `meta-analysis_classification.ipynb`: Jupyter notebook containing the code and explanations.
- `dataset.csv`: The dataset created from the scraped articles.

### Dependencies

- Beautiful Soup
- Requests
- Scikit-learn
- Pandas
- Numpy


