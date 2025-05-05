import pandas as pd

def data_prep(filename):
    # Load your dataset here
    # For example, using pandas to read a CSV file
    # Assuming the dataset is in a file named 'data.csv'
    # Adjust the file path and column indices as necessary
    data = pd.read_excel(filename)

    data = data.sort_values(by='Year')
    
    target_specific = [column for column in data.columns if 'CC' in column and column != 'CC_all']
    target_all = 'CC_all'
    for pc_column in [col for col in data.columns if 'PC' in col]:
        data[pc_column + '_prev_year'] = data.groupby(['Court', 'Municipality', 'Bench'])[pc_column].shift(1)
    
    data = data.drop(columns=[col for col in data.columns if 'PC_all' in col or 'IC' in col or ('PC' in col and '_prev_year' not in col)])

    categorical_columns = ['Court', 'Municipality', 'Bench']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    data = data.drop(columns=['Informatic People'])
    data = data.drop(columns=['Justice Officials'])
    data = data.dropna(axis=0, how='any')

    return data, target_specific, target_all