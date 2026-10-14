# Iris Flower Classification

Classifies iris flowers into three species (Setosa, Versicolor, Virginica) using a K-Nearest Neighbors (KNN) classifier on the Iris dataset from scikit-learn.

## What the script does
1. Loads the Iris dataset and prints the first rows, shape, missing values and summary statistics
2. Visualizes the data with a pairplot and feature histograms
3. Splits the data into 80% train and 20% test (stratified by species)
4. Scales features with StandardScaler
5. Trains a KNN classifier with 5 neighbors
6. Evaluates it with accuracy, precision, a classification report and a confusion matrix

## Files
- task1_iris_classification.py: main script
- iris_pairplot.png: pairwise feature plots by species
- iris_histograms.png: feature distributions
- iris_confusion_matrix.png: confusion matrix of the predictions

## Libraries
pandas, seaborn, matplotlib, scikit-learn

## How to run
pip install pandas seaborn matplotlib scikit-learn
python task1_iris_classification.py
