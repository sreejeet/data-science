# This example shows 3 ways to handle missing values in datasets
# We will use RandomForestRegressor to predict housing prices

# A little code to suppress FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


melb_data = pd.read_csv('./data/melb_data.csv')
y = melb_data.Price # Our target column to be predicted
X = melb_data.drop(['Price'], axis=1) # Our prediction features

# Removing all non-nuimeric values to keep this example simple
X = X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=0)

columns_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Get the mean absolute error of each method using this function
def dataset_score(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_absolute_error(y_test, pred)



# 1. Removing entire columns with missing vlaues
reduced_X_train = X_train.drop(columns_with_missing, axis=1)
reduced_X_test = X_test.drop(columns_with_missing, axis=1)
print('Mean Absolute error after dropping columns with missing values:')
print(dataset_score(reduced_X_train, reduced_X_test, y_train, y_test))



# 2. Using imputation to fill in missing data
imputer = SimpleImputer()
imputed_X_train = imputer.fit_transform(X_train)
imputed_X_test = imputer.fit_transform(X_test)
print('Mean Absolute error after imputing missing values:')
print(dataset_score(imputed_X_train, imputed_X_test, y_train, y_test))



# 3. Imputation with extra columns showing imputed values
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
for col in columns_with_missing:
    imputed_X_train_plus[col + '_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_missing'] = imputed_X_test_plus[col].isnull()
imputed_X_train_plus = imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = imputer.fit_transform(imputed_X_test_plus)
print('Mean Absolute error after imputing missing values and tracking them:')
print(dataset_score(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))



# Test Execution output
# Mean Absolute error after dropping columns with missing values:
# 188159.51050097012
# Mean Absolute error after imputing missing values:
# 186287.46044135865
# Mean Absolute error after imputing missing values and tracking them:
# 184788.7433620871
