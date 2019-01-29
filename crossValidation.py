'''
Using crossvalidation to score the prediction model and seeing the diffference
between using train_test_split and cross_val_score for model scoring.
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# Setup
data = pd.read_csv('./data/melb_data.csv')
cols = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols]
y = data.Price
pipe = make_pipeline(SimpleImputer(), RandomForestRegressor())


# Getting cross validation score
scores = cross_val_score(pipe, X, y, scoring='neg_mean_absolute_error')
print('Cross validation scores')
print('Negetive mean absolute error:')
print(scores)
print('Mean of the above:', -1 * scores.mean())


# Getting score using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
mae = mean_absolute_error(preds, y_test)

print('Mean absolute error', mae)
