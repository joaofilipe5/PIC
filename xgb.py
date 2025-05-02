import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from data_prep import data_prep
import joblib
from sklearn.model_selection import train_test_split

data, target_specific, target_all = data_prep()
data = data.drop(columns=['Year'])
train_X, test_X, train_y, test_y = train_test_split(data.drop(columns=target_specific+[target_all]), data[target_specific], test_size=0.2, random_state=42)

xgb = XGBRegressor(
    random_state=42
    )

xgb.fit(train_X, train_y)
predictions = xgb.predict(test_X)
predictions.round(0)

print("Mean Squared Error: ", mean_squared_error(test_y, predictions))
print("Mean Absolute Error: ", mean_absolute_error(test_y, predictions))
print("R2 Score: ", r2_score(test_y, predictions))

joblib.dump(xgb, 'xgboost.joblib')
