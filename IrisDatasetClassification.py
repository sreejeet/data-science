# Using decision tree Classifier to classify plants

import sklearn.datasets as datasets
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

model = DecisionTreeClassifier()

# We are spliting the dataset to test and train with different values
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model.fit(X_train, y_train)
predications = model.predict(X_test)
mae = mean_absolute_error(y_test, predications)

# Printing the first lines of our test
print("For features %s\nWe predicted value %s and the actual value is %s"
    % (X_test[0], predications[0], y_test[0]))

print("Mean absolute error is:", mae)
