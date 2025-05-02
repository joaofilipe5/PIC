import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB
import joblib


from data_prep import data_prep

xgb = joblib.load('xgboost.joblib')

data, target_specific, target_all = data_prep()

def CC_all_pred(model, data):
    # Predict the target variable using the model
    predictions = model.predict(data.drop(columns=target_specific+[target_all]+['Year']))
    predictions = predictions.round(0)
    return np.sum(predictions)

data_2023 = data[data['Year'] == 2023]

print("Predicted CC_all for 2023")
print(CC_all_pred(xgb, data_2023))
print('Real CC_all for 2023')
print(data_2023[target_all].sum())

m = gp.Model()

for feature in data_2023.drop(columns=target_specific+[target_all]+['Year']).columns:
    m.addVar(vtype=GRB.INTEGER, name=feature)

m.setObjective()
# incompleto