import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #plotting
from sklearn.model_selection import train_test_split
import sys
!{sys.executable} -m pip install mglearn
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine

wine_dataset = load_wine()

print("Keys of wine_dataset:\n", wine_dataset.keys())
print(wine_dataset['DESCR'][:193] + "\n...")

#print target_names
print("Target names:", wine_dataset['target_names'])
#print feature_names
print("Feature names:\n", wine_dataset['feature_names'])
#print type of data
print("Type of data:", type(wine_dataset['data']))
#print shape of data
print("Shape of data:", wine_dataset['data'].shape)
#First five rows
print("First five rows of data:\n", wine_dataset['data'][:5])
#print type of target
print("Type of target:", type(wine_dataset['target']))
#print shape of target
print("Shape of target:", wine_dataset['target'].shape)
#print target values
print("Target:\n", wine_dataset['target'])

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    wine_dataset['data'], wine_dataset['target'], random_state=0)
#shape of the X_train and y_train
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
#shape of the X_test and y_test
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# create dataframe from data in X_train
# label the columns using the strings in wine_dataset.feature_names
wine_dataframe = pd.DataFrame(X_train, columns=wine_dataset.feature_names)
#info
wine_dataframe.info()
#head
wine_dataframe.head()
#describe
wine_dataframe.describe()

# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(wine_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
X_new = np.array([[11.96,	1.09,	2.30,	21.0,	101.0,	3.38,	2.14,	0.13,	1.65,	3.21,	0.99,	3.13,	886.0]])
print("X_new.shape:", X_new.shape)
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",
       wine_dataset['target_names'][prediction])

#make prediction
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
#print the test score
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#checking k 1 to 25 for the best choice
neighbors = np.arange(1, 25)
train_accuracy, test_accuracy = list(), list()

for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.figure(figsize=[13, 8])
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("k Value VS Accuracy")
plt.xlabel("k Value (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))