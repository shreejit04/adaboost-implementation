import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
from medmnist import OrganMNIST3D


def get_error_rate(pred, Y):
    return np.sum(pred != Y) / float(len(Y))


def my_adaboost(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    w = np.ones(n_train) / n_train  # Initialize weights
    pred_train, pred_test = np.zeros(n_train), np.zeros(n_test)

    for i in range(M):
        if clf == clf_rf:
            clf.fit(X_train, Y_train)
        else:
            clf.fit(X_train, Y_train, sample_weight=w)

        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        err_m = np.sum(w * (pred_train_i != Y_train)) / np.sum(w)

        if err_m == 0:
            alpha_m = 0.01
        else:
            alpha_m = 0.5 * np.log((1 - err_m) / err_m)

        w *= np.exp(-alpha_m * Y_train * pred_train_i)
        w /= np.sum(w)

        pred_train += alpha_m * pred_train_i
        pred_test += alpha_m * pred_test_i

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    return get_error_rate(pred_train, Y_train), get_error_rate(pred_test, Y_test)


if __name__ == '__main__':
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target

    y_binary = np.where(y % 2 == 0, 1, -1)  # Converting to even versus odd; two-class dataset

    x_train, x_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.3)

    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)  # max_depth = 1 would be weak
    clf_rf = RandomForestClassifier(n_estimators=1, random_state=1)

    # Sci-kit learn Adabost
    sklearn_adaboost = AdaBoostClassifier(clf_tree, n_estimators=200)
    sklearn_adaboost.fit(x_train, y_train)
    pred_adaboost = sklearn_adaboost.predict(x_test)
    accuracy_sklearn_adaboost = accuracy_score(y_test, pred_adaboost)
    print("scikit-learn AdaBoost Accuracy: ", accuracy_sklearn_adaboost)

    er_train_ada, er_test_ada = [], []
    for i in range(10, 200, 10):
        er_i = my_adaboost(y_train, x_train, y_test, x_test, i, clf_tree)
        er_train_ada.append(er_i[0])
        er_test_ada.append(er_i[1])

    plt.plot(range(10, 200, 10), er_train_ada, label='train_error')
    plt.plot(range(10, 200, 10), er_test_ada, label='test_error')
    plt.legend()
    plt.title("my_adaboost with DecisionTreeClassifier")
    plt.show()

    er_train_ada, er_test_ada = [], []
    for i in range(10, 200, 10):
        er_i = my_adaboost(y_train, x_train, y_test, x_test, i, clf_rf)
        er_train_ada.append(er_i[0])
        er_test_ada.append(er_i[1])

    plt.plot(range(10, 200, 10), er_train_ada, label='train_error')
    plt.plot(range(10, 200, 10), er_test_ada, label='test_error')
    plt.legend()
    plt.title("my_adaboost with RandomForest classifier")
    plt.show()

    ############## BONUS #####################
    # We used a dataset from kaggle
    x_kaggle = []
    y_kaggle = []

    # Read the CSV file
    with open('sample.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            x_kaggle.append(int(row['PassengerId']))
            y_kaggle.append(int(row['Survived']))

    x_kaggle = np.array(x_kaggle)
    y_kaggle = np.array(y_kaggle)

    x_train, x_test, y_train, y_test = train_test_split(x_kaggle, y_kaggle, test_size=0.3)

    x_train_reshaped = x_train.reshape(-1, 1)
    x_test_reshaped = x_test.reshape(-1, 1)

    er_train_ada, er_test_ada = [], []
    for i in range(10, 200, 10):
        er_i = my_adaboost(y_train, x_train_reshaped, y_test, x_test_reshaped, i, clf_tree)
        er_train_ada.append(er_i[0])
        er_test_ada.append(er_i[1])

    plt.plot(range(10, 200, 10), er_train_ada, label='train_error')
    plt.plot(range(10, 200, 10), er_test_ada, label='test_error')
    plt.legend()
    plt.title("my_adaboost with DecisionTreeClassifier: Kaggle dataset")
    plt.show()

    er_train_ada, er_test_ada = [], []
    for i in range(10, 200, 10):
        er_i = my_adaboost(y_train, x_train_reshaped, y_test, x_test_reshaped, i, clf_rf)
        er_train_ada.append(er_i[0])
        er_test_ada.append(er_i[1])

    plt.plot(range(10, 200, 10), er_train_ada, label='train_error')
    plt.plot(range(10, 200, 10), er_test_ada, label='test_error')
    plt.legend()
    plt.title("my_adaboost with RandomForest classifier: Kaggle dataset")
    plt.show()
