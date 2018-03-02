import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



#Importing the dataset
path = "/Users/oskam/PycharmProjects/classification/NaiveBayes/iris.txt"
dataset = pd.read_csv(path)


#looking at the first 5 values of the dataset
# print(dataset.head())
X = dataset.iloc[:,:4].values
y = dataset['species'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# print(len(X_train))
# print(len(X_test))

# Feature Scaling to bring the variable in a single scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes Classification to the Training set with linear kernel
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

print(nvclassifier)

# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
print(y_pred)

#lets see the actual and predicted value side by side
y_compare = np.vstack((y_test,y_pred)).T
#actual value on the left side and predicted value on the right hand side
#printing the top 5 values
print(y_compare[:5,:])