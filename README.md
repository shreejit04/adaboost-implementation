# adaboost-implementation

**Introduction**

This code implements an AdaBoost algorithm from scratch and compares its performance with scikit-learn's AdaBoost implementation. The AdaBoost algorithm is an ensemble learning method that combines weak learners to create a strong classifier. The code specifically uses decision trees as weak learners.

**Dependencies**

1.  numpy: Numerical operations library for efficient array operations.
2.  scikit-learn: Machine learning library providing tools for classification and regression.
3.  matplotlib: Plotting library for visualizing the results.
4.  csv: Module for reading and writing CSV files.
5.  medmnist: Library for working with medical image datasets.

**Functions and Classes**

1. get_error_rate(pred, Y): Calculates the error rate between predicted and actual labels.

Input Parameters:

  pred: Predicted labels.
  Y: Actual labels.

Output:

  Error rate as a float.

2. my_adaboost(Y_train, X_train, Y_test, X_test, M, clf): Implements the AdaBoost algorithm with decision trees as weak learners.

Input Parameters:

    Y_train: Training labels.
    X_train: Training features.
    Y_test: Test labels.
    X_test: Test features.
    M: Number of weak learners.
    clf: Weak learner (DecisionTreeClassifier or RandomForestClassifier).

Output: Tuple containing the error rates for training and testing.

main(): Description: The main section of the script that loads the digits dataset, performs data preprocessing, trains scikit-learn's AdaBoost, and compares it with the custom AdaBoost implementation on both synthetic and Kaggle datasets.

**Usage**
1.  Import necessary libraries and modules.
2.  Define the get_error_rate and my_adaboost functions.
3.  Load a dataset (digits or Kaggle) and preprocess it.
4.  Create weak learners (DecisionTreeClassifier and RandomForestClassifier).
5.  Train scikit-learn's AdaBoost and custom AdaBoost on the dataset.
6.  Evaluate and compare their performances using error rates.
Plot the training and testing error curves for both AdaBoost implementations.
