# Using xgboost for machine learning
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


# Function for testing mae for XGBRegressor model
def model_test(X_train, X_test, y_train, y_test, est=None, esr=None, lr=None):
    if est and lr:
        model = XGBRegressor(n_estimators=est, learning_rate=lr)
    elif est:
        model = XGBRegressor(n_estimators=est)
    else:
        model = XGBRegressor()

    if esr:
        model.fit(X_train, y_train, early_stopping_rounds=esr,
            eval_set=[(X_test,y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    return mean_absolute_error(y_test, pred)

# Setting up train and test data
data = pd.read_csv('../data/train.csv')
data.dropna(inplace=True, subset=['SalePrice'], axis=0)

X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
y = data.SalePrice
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values)

imputer = SimpleImputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)


# Simple model test
print('Simple model test:')
mae = model_test(X_train, X_test, y_train, y_test)
print('Mean absolute error is %d' % (mae))


# Cycling through different values for n_estimators
print('\n\nModel test using different values for n_estimators:')
track = {}
for x in range(100,1000,100):
    mae = model_test(X_train, X_test, y_train, y_test, x)
    track[mae] = x
    print('Mean absolute error for %d n_estimators is %d' % (x, mae))

print('Lowest mean absolute error is %d for %d estimators'
    % (min(track), track.get(min(track))))


# Using early_stopping_rounds to stop unnecessary iterations
# by automatically stopping when mae stops improving
print('\n\nModel test using early_stopping_rounds for automatic stop:')
est = 1000
esr = 10
mae = model_test(X_train, X_test, y_train, y_test, est, esr)
print('Mean absolute error is %d' % (mae))


# Using learning rate to increase estimations without major overfitting
print('\n\nModel test using learning_rate for increased \
estimations and low overfitting:')
est = 100000
esr = 1000
lr = 0.02
mae = model_test(X_train, X_test, y_train, y_test, est, esr, lr)
print('Mean absolute error is %d' % (mae))


# Sample output:
# Simple model test:
# Mean absolute error is 18718
#
#
# Model test using different values for n_estimators:
# Mean absolute error for 100 n_estimators is 18718
# Mean absolute error for 200 n_estimators is 18366
# Mean absolute error for 300 n_estimators is 18183
# Mean absolute error for 400 n_estimators is 18129
# Mean absolute error for 500 n_estimators is 18162
# Mean absolute error for 600 n_estimators is 18245
# Mean absolute error for 700 n_estimators is 18265
# Mean absolute error for 800 n_estimators is 18235
# Mean absolute error for 900 n_estimators is 18236
# Lowest mean absolute error is 18129 for 400 estimators
#
#
# Model test using early_stopping_rounds for automatic stop:
# Mean absolute error is 18705
#
#
# Model test using learning_rate for increased estimations and low overfitting:
# Mean absolute error is 18081
