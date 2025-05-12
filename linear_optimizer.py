import pandas as pd
import numpy as np
from gurobipy import GRB, Model
import joblib
import tkinter as tk


from src.data_prep import data_prep

model_filename = 'models/ridge.joblib' # ATTENTION: Must be linear (Linear Regression, Ridge, Lasso, ElasticNet, etc.)
model = joblib.load(model_filename)

year = 2016 # Pick the year you want to optimize for. Goal is to give the model future maximum staffing limits to predict completed cases.
print('Optimizing for year:', year)

data, target_specific = data_prep('src/data.xlsx')
data_year = data[data['Year'] == year]
data_year = data_year.drop(columns=['Year'])
data_year = data_year.drop(columns=target_specific)

coef_df = pd.DataFrame(
    model.coef_,
    columns=data_year.columns,
    index=target_specific
)

intercepts = pd.Series(model.intercept_, index=target_specific)

m = Model('Objective')

court_vars = [feature for feature in coef_df.columns if 'Court' in feature]
mun_vars = [feature for feature in coef_df.columns if 'Municipality' in feature]
bench_vars = [feature for feature in coef_df.columns if 'Bench' in feature]

court_mun_bench_tuples = []
for court in court_vars:
    court_data = data_year[data_year[court] == 1]
    for mun in mun_vars:
        mun_data = court_data[court_data[mun] == 1]
        for bench in bench_vars:
            if bench in mun_data.columns and mun_data[bench].any():
                court_mun_bench_tuples.append((court, mun, bench))
print(f"Number of court-municipality-bench combinations: {len(court_mun_bench_tuples)}")
print('Number of records in the year:', data_year.shape[0])

pc_vars = [feature for feature in coef_df.columns if 'PC' in feature]

staff_vars = ['Judges',
'Justice Secretary',
'Law Clerck',
'Auxiliar Clerck',
'Administrative/Technical People',
'Operational/Auxiliar People']

x_vars = {(staff, court, mun, bench): m.addVar(vtype=GRB.INTEGER, name=f"{staff}, {court}, {mun}, {bench}") for staff in staff_vars
          for court, mun, bench in court_mun_bench_tuples}

staff_max = {}
for court in court_vars:
    court_rows = data_year[data_year[court] == 1]
    for staff in staff_vars:
        total = court_rows[staff].sum()
        staff_max[(court, staff)] = total


outputs = []

for i in range(data_year.shape[0]):
    row = data_year.iloc[i]
    # Extract the court, municipality, and bench from the row
    try:
        court = row[court_vars].idxmax()
        mun = row[mun_vars].idxmax()
        bench = row[bench_vars].idxmax()
        #print(f"Processing: {court}, {mun}, {bench}")
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        # print(f"{bench} does not exist in {mun} of {court} in the year of {year}.")
        continue


    for target in target_specific:
        expr = intercepts[target]
        for feature in coef_df.columns:
            if feature in staff_vars:
                expr += coef_df.loc[target, feature] * x_vars[(feature, court, mun, bench)]
            else:
                expr += coef_df.loc[target, feature] * row[feature]
        outputs.append(expr)

print('Setting objective function...')
m.setObjective(sum(outputs), GRB.MAXIMIZE)
print('Objective function set.')


print('Adding constraints...')
judge_var_keys = [k for k in x_vars.keys() if k[0] == 'Judges']
for key, var in x_vars.items():
    m.addConstr(var >= 0, name="non_negativity")

    if key in judge_var_keys:
        m.addConstr(var >= 1, name="Min_Judges")

for court in court_vars:
    for staff in staff_vars:
        staff_vars_ = [var for key, var in x_vars.items() if court == key[1] and staff == key[0]]
        if staff_vars_:
            m.addConstr(sum(staff_vars_) <= staff_max[(court, staff)], name=f"Max_{staff}_{court}")
        else:
            continue


print('Optimizing...')
m.optimize()

print('Optimization complete.')
# Check the optimization status
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    print(f"Objective value: {m.objVal}")
    print(f"Rounded objective value: {round(m.objVal,0)}")
elif m.status == GRB.INFEASIBLE:
    print("Model is infeasible!")
elif m.status == GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print(f"Solver ended with status {m.status}")


print(f"ML model prediction (default allocations): {model.predict(data_year).sum().sum()}")
print(f"Real value: {data.loc[data['Year']==year, target_specific].sum().sum()}")
'''print("Variable values:")
for var in m.getVars():
    if 'Judges' in var.VarName:
        print(f"{var.VarName}: {var.X}")
'''