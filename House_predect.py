import pandas as pd
import numpy as np
data = pd.read_csv("data.csv")

from sklearn.model_selection import StratifiedShuffleSplit

data_Split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_set, test_set in data_Split.split(data, data["CHAS"]):
    Strain_set = data.loc[train_set]
    Stest_set = data.loc[test_set]
data = Strain_set.drop("MEDV", axis=1)
data_labels = Strain_set["MEDV"].copy()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_slcr", StandardScaler())
])

data_tr = my_pipeline.fit_transform(data)

"""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)
"""
"""
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(data_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)"""


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(data_tr, data_labels)
some_data = data.iloc[:5]
some_labels = data_labels[:5]
prepared_data = my_pipeline.transform(some_data)


from sklearn.metrics import mean_squared_error
data_predection = model.predict(data_tr)
lin_mse = mean_squared_error(data_labels, data_predection)
lin_rmse = np.sqrt(lin_mse)

#Cross Validation
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, data_tr, data_labels, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-score)




def print_Score(scorese):
    print("Score: ", scorese)
    print("mean: ", scorese.mean())
    print("std: ", scorese.std())

print(print_Score(rmse_scores))

X_test = Stest_set.drop("MEDV", axis = 1)
Y_test = Stest_set["MEDV"].copy()

X_test_prepared = my_pipeline.transform(X_test)
final_Prepared = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_Prepared)
final_rmse = np.sqrt(final_mse)

print(final_rmse)

