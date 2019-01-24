# Using gradient boost regressor to plot partial dependence plot
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.impute import SimpleImputer


# The PDP will show us the relationship between the target and its features
features = ['Distance', 'Landsize', 'BuildingArea']
data = pd.read_csv('./data/melb_data.csv')
y = data.Price
X = data[features]
imputer = SimpleImputer()
X = imputer.fit_transform(X)

model = GradientBoostingRegressor()
model.fit(X, y)
fig, plots = plot_partial_dependence(
    model,
    features=[0,1,2],
    X=X,
    feature_names=features,
    grid_resolution=40)

fig.show()
input('Press enter to continue...')
