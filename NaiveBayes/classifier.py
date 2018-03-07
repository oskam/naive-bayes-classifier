import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm as colormap

# possible data types:
# iris
# wine
# glass
# diabetes

# discretization types:
# equal_width_intervals
# equal_frequency_intervals
# ???

data_type = "iris"

# Importing the datasets
iris_path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/iris.txt"
wine_path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/wine.txt"
glass_path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/glass.txt"
diabetes_path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/diabetes.txt"


def classifier(data_type):

    X =[]
    y = []

    if data_type == "iris":
        path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/iris.txt"
        df = pd.read_csv(path)
        X = df.iloc[:,:4].values
        y = df['species'].values
        # sns.pairplot(df, hue='species')
        # plt.show()

    elif data_type == "wine":
        path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/wine.txt"
        df = pd.read_csv(path)
        class_column = 0
        y = df[df.columns[class_column]].values
        X = df.drop(columns=df.columns[class_column]).values
        # sns.pairplot(df, hue='class')
        # plt.show()

    elif data_type == "glass":
        path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/glass.txt"
        df = pd.read_csv(path)
        df = df.drop(columns=df.columns[0])
        class_column = 9
        y = df[df.columns[class_column]].values
        X = df.drop(columns=df.columns[class_column]).values

    elif data_type == "diabetes":
        path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/diabetes.txt"
        df = pd.read_csv(path)
        class_column = 8
        y = df[df.columns[class_column]].values
        X = df.drop(columns=df.columns[class_column]).values
        # sns.pairplot(df, hue='class')
        # plt.show()
    test = [2,2,3,4,6,6]
    print(test)
    print(discretization("equal_width_intervals", test, 3))
    print(discretization("equal_frequency_intervals", test, 3))

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    print(X_train)

    # Create a new figure and set the figsize argument so we get square-ish plots of the 4 features.
    # plt.figure(figsize=(15, 3))

    # Iterate over the features, creating a subplot with a histogram for each one.
    # for feature in range(X_train.shape[1]):
    #     plt.subplot(1, 8, feature + 1)
    #     plt.hist(X_train[:, feature], 20)
    # plt.show()

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    # nvclassifier = GaussianNB()
    nvclassifier = MultinomialNB(alpha=1.0)
    nvclassifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvclassifier.predict(X_test)

    #---------------------------------

    #lets see the actual and predicted value side by side
    y_compare = np.vstack((y_test,y_pred)).T

    #actual value on the left side and predicted value on the right hand side
    #printing the top 5 values
    print(y_compare[:5,:])

    evaluation(y_test, y_pred, y)

def discretization(type, X, k):
    if type == "equal_width_intervals":
        return pd.cut(X,k)
    elif type == "equal_frequency_intervals":
        return pd.qcut(X,k)


def evaluation(y_test, y_pred, y):
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # plt.figure()
    # plot_confusion_matrix(cm, classes=[str(i) for  i in range(2)],
    #                       title='Confusion matrix, without normalization')

    #finding accuracy from the confusion matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred +=cm[row,c]
            else:
                falsePred += cm[row,c]
    print('Correct predictions: ', corrPred)
    print('False predictions', falsePred)
    print ('Accuracy of the Naive Bayes Classification is: ', corrPred/(cm.sum()))
    print(accuracy_score(y_test, y_pred))

    print(precision_score(y_test, y_pred, average=None))
    print(recall_score(y_test, y_pred, average=None))
    print(f1_score(y_test, y_pred, average=None))


def cross_validation():
    pass

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):

    print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


classifier(data_type)