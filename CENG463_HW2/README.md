
# Unsupervised Learning on Animal Images

## CENG463 - HOMEWORK 2

### Introduction

In this assignment, we explore unsupervised learning clustering methods using animal images (cat, frog, horse). Our objective is to visualize scores (Silhouette and Ground Truth - ARI) by experimenting with various strategies such as feature extraction and dimensionality reduction, including PCA, t-SNE, and NMF, before applying three different clustering methods (K-Means, Agglomerative, DBSCAN).

### Creating the Dataset

This Python code performs various tasks using the CIFAR-10 dataset. Initially, the CIFAR-10 dataset is loaded using the `cifar10.load_data()` function, and specific classes are selected. We prepared 500 randomly chosen examples from each selected class by converting them to grayscale, normalizing, and flattening the images. The dataset is standardized using `StandardScaler`, centering the features around the mean and scaling by the standard deviation.

### Comparative Analysis of Clustering Algorithms on PCA

We perform dimensionality reduction using PCA (Principal Component Analysis) with two components and subsequently apply different clustering algorithms, including KMeans, Agglomerative Clustering, and DBSCAN. Metrics evaluating clustering performance, such as Silhouette and Adjusted Rand Index (ARI), are calculated for each algorithm. Special procedures are applied to exclude noise when using DBSCAN.

### Results

We visualize and compare the clustering performance using Silhouette scores and ARI. The analysis helps understand how different dimensionality reduction techniques and clustering algorithms affect the clustering results.

### Usage

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/merveozan/CENG463_Fall23_Machine-Learning_Projects/unsupervised_learning_animal_images.git
    ```
2. Install the required dependencies:

3. Run the notebook:
    ```bash
    jupyter notebook unsupervised_learning_with_animal_images.ipynb
    ```

### Files

- `unsupervised_learning_with_animal_images.ipynb`: Jupyter notebook containing the code and explanations.


### Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
