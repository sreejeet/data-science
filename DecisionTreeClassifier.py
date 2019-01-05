# This is a decision tree classifier using scikit-learn
from sklearn import tree

# Lets say we have a bunch of black pens and green tennis balls
# We can take numeric values for each feature:
# shape 'pointed' as 0
# shape 'blunt' as 1
# colour 'green' as 2
# colour 'black' as 3

dict = {
    0:'pointed',
    1:'blunt',
    2:'green',
    3:'black',
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
    'pen',
    'ball',
    'pen',
    'ball',
    'ball',
    'pen',
]

# This is our classifier or model
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)

# Predict the Type of object with given features
# e.g. [['blunt','green']] -> expected output is 'ball'
for x in range(0,2):
    for y in range(2,4):
        print(dict.get(x), dict.get(y))
        print(classifier.predict( [[x,y]] ))
