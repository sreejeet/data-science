# This is a decision tree classifier using scikit-learn
# Lets say we have a bunch of black pens and green tennis balls
# We can take numeric values for each feature and label:
# shape 'pointed' as 0
# shape 'blunt' as 1
# colour 'green' as 2
# colour 'black' as 3
# colour 'ball' as 4
# colour 'pen' as 5

from sklearn import tree

dict = {
    #shape
    0:'pointed',
    1:'blunt',

    #colour
    2:'green',
    3:'black',

    #object
    4:'ball',
    5:'pen',
}

features = [
    [0, 3],
    [1, 2],
    [0, 3],
    [1, 2],
    [1, 2],
    [0, 3],
]

labels = [
    5,
    4,
    5,
    4,
    4,
    5,
]

# This is our classifier or model
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)
pred = classifier.predict( features )

# Predict the Type of object with all combination of features
# e.g. [['blunt','green']] -> expected output is 'ball'
for x in range(0,2):
    for y in range(2,4):
        print(dict.get(x), dict.get(y))
        print(dict.get(classifier.predict( [[x,y]] )[0]))

# We can find the mean absolute error of our prediction
# by importing the MAE function from sklearn
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(labels, pred)

# We get a perfect value of 0.0 because the
# features are 100% accurate and
# training and testing set are the same.
print("The mean absolute error is", mae)
