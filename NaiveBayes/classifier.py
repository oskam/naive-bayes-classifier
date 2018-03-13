import argparse
import itertools

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as color

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score, \
    recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from mdlp.discretization import MDLP

warnings.filterwarnings('ignore')

# possible data types:
# iris
# wine
# glass
# diabetes

# discretization types:
# equal_width_intervals
# equal_frequency_intervals
# MDLP

DISC_EQUAL_FREQ = 0
DISC_EQUAL_WIDTH = 1
DISC_MDLP = 2

DATA_TYPE = "iris"

# Importing the datasets
datasets_info = {
    'iris': {
        'path': "NaiveBayes/iris.txt",
        'drop_columns': [],
        'class_column': 4
    },
    'wine': {
        'path': "NaiveBayes/wine.txt",
        'drop_columns': [],
        'class_column': 0
    },
    'glass': {
        'path': "NaiveBayes/glass.txt",
        'drop_columns': [0],
        'class_column': 9
    },
    'diabetes': {
        'path': "NaiveBayes/diabetes.txt",
        'drop_columns': [],
        'class_column': 8
    }
}


def classifier(args):
    dataset_info = datasets_info[args.data_type]

    df = pd.read_csv(dataset_info['path'])
    for drop_col in dataset_info['drop_columns']:
        df = df.drop(columns=df.columns[drop_col])
    y = df[df.columns[dataset_info['class_column']]]
    X = df.drop(columns=df.columns[dataset_info['class_column']])

    if args.plot:
        sns.pairplot(df, hue=df.columns[dataset_info['class_column']])
        plt.show()

    # Discretize values before training
    if args.discretization_bins > 0:
        if args.discretization_mode == DISC_MDLP:
            transformer = MDLP()
            X = transformer.fit_transform(X, y)
        else:
            for column in X:
                bins = discretization(args.discretization_mode, X[column], args.discretization_bins)
                X[column] = bins

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Create a new figure and set the figsize argument so we get square-ish plots of the 4 features.
    if args.plot:
        plt.figure(figsize=(10, 3))

    # Iterate over the features, creating a subplot with a histogram for each one.
    if args.plot:
        for feature in range(X_train.shape[1]):
            plt.subplot(1, len(X_train.columns), feature + 1)
            sns.distplot(X_train.values[:, feature])
        plt.show()

    # Fitting Naive Bayes Classification to the Training set
    # classifier = GaussianNB()
    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X_train, y_train)

    cross_validation(classifier, X, y)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(y_pred)

    evaluation(y_test, y_pred, args)


def discretization(mode, x, k):
    if mode == DISC_EQUAL_WIDTH:
        return pd.cut(x, k, labels=False)
    elif mode == DISC_EQUAL_FREQ:
        return pd.qcut(x, k, labels=False, duplicates='drop')


def evaluation(y_test, y_pred, args):
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    if args.plot:
        plt.figure()
        plot_confusion_matrix(cm, classes=[str(i) for i in range(0,2)], args=args)

    a = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, warn_for=())
    print("Accuracy: " + str(a))
    print("Precision: " + str(p))
    print("Precision Weighted: " + str(precision_score(y_test, y_pred, average="weighted")))
    print("Precision Macro: " + str(precision_score(y_test, y_pred, average="macro")))
    print("Recall:  " + str(r))
    print("Recall Weighted: " + str(recall_score(y_test, y_pred, average="weighted")))
    print("Recall Macro: " + str(recall_score(y_test, y_pred, average="macro")))
    print("Fscore: " + str(f))
    print("Fscore Weighted: " + str(f1_score(y_test, y_pred, average="weighted")))
    print("Fscore Micro: " + str(f1_score(y_test, y_pred, average="micro")))
    print("Fscore Macro: " + str(f1_score(y_test, y_pred, average="macro")))


def cross_validation(model, x, y):
    # Perform N-fold cross validation
    cv_N = 10
    kf = KFold(n_splits=cv_N)
    scores = cross_val_score(model, x, y, cv=kf)
    print("Cross-validated scores: " + str(scores))

    # Perform N-fold cross stratified validation
    skf = StratifiedKFold(n_splits=cv_N)
    scores = cross_val_score(model, x, y, cv=skf)
    print("Stratified Cross-validated scores: " + str(scores))


def plot_confusion_matrix(cm, classes, args,
                          title='Confusion matrix'):
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=color.Reds)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NaiveBayes classifier.')
    parser.add_argument('--data-type', '-d', required=True, action='store', choices=datasets_info.keys(),
                        help='data type')
    parser.add_argument('--plot', '-p', default=False, action='store_true',
                        help='draw the plots')
    parser.add_argument('--discretization-mode', '-m', default=DISC_EQUAL_FREQ, action='store', type=int,
                        choices=[DISC_EQUAL_FREQ, DISC_EQUAL_WIDTH, DISC_MDLP],
                        help='discretize using equal widths (0(default): equal frequency, 1: equal width, 2: mdlp)')
    parser.add_argument('--discretization-bins', '-b', default=0, action='store', type=int, choices=range(0, 11),
                        help='discretize the values to n bins (default: 0, do not discretize)')

    args = parser.parse_args()

    classifier(args)
