from gurobipy import GRB, Model, Var
import pandas as pd

from src.data_prep import data_prep

data, target_specific = data_prep('src/data.xlsx')

class Optimization:
    def __init__(self, debug_callback=None):
        self.staff_vars = ['Judges', 'Justice Secretary', 'Law Clerck', 'Auxiliar Clerck',
                          'Administrative/Technical People', 'Operational/Auxiliar People']
        self.debug_callback = debug_callback

    def log_debug(self, message):
        """Log debug message using the callback if provided"""
        if self.debug_callback:
            self.debug_callback(message)

    def create_decision_variables(self, lp_model: Model, year_data, granularity, selected_court=None, selected_municipality=None) -> dict[tuple, Var]:
        """Create decision variables for the optimization model"""
        
        staff_vars = self.staff_vars
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]
        bench_vars = [col for col in year_data.columns if col.startswith('Bench_')]

        if granularity == "Country":
            court_mun_bench_tuples = []
            for court in court_vars:
                court_data = year_data[year_data[court] == 1]
                for mun in mun_vars:
                    mun_data = court_data[court_data[mun] == 1]
                    for bench in bench_vars:
                        if bench in mun_data.columns and mun_data[bench].any():
                            court_mun_bench_tuples.append((court, mun, bench))

            x_vars = {(staff, court, mun, bench): lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {court}, {mun}, {bench}") for staff in staff_vars
            for court, mun, bench in court_mun_bench_tuples}
        
        elif granularity == "Court":
            if not selected_court:
                raise ValueError("Selected court is required for Court granularity")
            tuples = []
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            for mun in mun_vars:
                mun_data = court_data[court_data[mun] == 1]
                for bench in bench_vars:
                    if bench in mun_data.columns and mun_data[bench].any():
                        tuples.append((mun, bench))
            
            x_vars = {(staff, mun, bench): lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {mun}, {bench}") for staff in staff_vars
                      for mun, bench in tuples}
        
        elif granularity == "Municipality":
            if not selected_court or not selected_municipality:
                raise ValueError("Selected court and municipality are required for Municipality granularity")
            benches = []
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]
            for bench in bench_vars:
                if bench in mun_data.columns and mun_data[bench].any():
                    benches.append(bench)
            
            x_vars = {(staff, bench): lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {bench}") for staff in staff_vars
                      for bench in benches}
            
        else:
            raise ValueError("Invalid granularity selected")
        

        return x_vars

    def objective_function(self, ml_model, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        """Define the objective function for the optimization model"""

        staff_vars = self.staff_vars
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]
        bench_vars = [col for col in year_data.columns if col.startswith('Bench_')]

        coef_df = pd.DataFrame(
        ml_model.coef_,
        columns=year_data.columns,
        index=target_specific
        )

        intercepts = pd.Series(ml_model.intercept_, index=target_specific)

        outputs = []

        if granularity == "Country":
            for i in range(year_data.shape[0]):
                row = year_data.iloc[i]
                # Extract the court, municipality, and bench from the row
                try:
                    court = row[court_vars].idxmax()
                    mun = row[mun_vars].idxmax()
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    self.log_debug(f"Error extracting data for row {i}: {str(e)}")
                    continue
                
                for target in target_specific:
                    expr = intercepts[target]
                    for feature in coef_df.columns:
                        if feature in staff_vars:
                            expr += coef_df.loc[target, feature] * decision_vars[(feature, court, mun, bench)]
                        else:
                            expr += coef_df.loc[target, feature] * row[feature]
                    outputs.append(expr)
        
        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            for i in range(court_data.shape[0]):
                row = court_data.iloc[i]
                # Extract the municipality and bench from the row
                try:
                    mun = row[mun_vars].idxmax()
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    self.log_debug(f"Error extracting data for row {i}: {str(e)}")
                    continue
            
                for target in target_specific:
                    expr = intercepts[target]
                    for feature in coef_df.columns:
                        if feature in staff_vars:
                            expr += coef_df.loc[target, feature] * decision_vars[(feature, mun, bench)]
                        else:
                            expr += coef_df.loc[target, feature] * row[feature]
                    outputs.append(expr)
        
        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            for i in range(mun_data.shape[0]):
                row = mun_data.iloc[i]
                # Extract the bench from the row
                try:
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    self.log_debug(f"Error extracting data for row {i}: {str(e)}")
                    continue
            
                for target in target_specific:
                    expr = intercepts[target]
                    for feature in coef_df.columns:
                        if feature in staff_vars:
                            expr += coef_df.loc[target, feature] * decision_vars[(feature, bench)]
                        else:
                            expr += coef_df.loc[target, feature] * row[feature]
                    outputs.append(expr)

        return outputs

    def add_constraints(self, lp_model: Model, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        """Add constraints to the optimization model"""
        staff_vars = self.staff_vars
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]
        staff_max = self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)

        # Define island groups for Açores and Madeira
        island_groups = {
            'Court_Acores': {
                'Sao_Miguel': ['Municipality_Ponta Delgada', 'Municipality_Vila Franca do Campo', 'Municipality_Ribeira Grande'],
                'Terceira': ['Municipality_Angra do Heroismo', 'Municipality_Vila Praia da Vitoria'],
                'Pico': ['Municipality_Sao Roque do Pico'],
                'Faial': ['Municipality_Horta'],
                'Flores': ['Municipality_Santa Cruz das Flores'],
                'Graciosa': ['Municipality_Santa Cruz Graciosa (R.A.A)'],
                'Sao_Jorge': ['Municipality_Velas (R.A.A.)']
            },
            'Court_Madeira': {
                'Madeira': ['Municipality_Funchal', 'Municipality_Ponta do Sol', 'Municipality_Santa Cruz'],
                'Porto_Santo': ['Municipality_Porto Santo']
            }
        }

        # Add non-negativity constraints for all variables
        for var in decision_vars.values():
            lp_model.addConstr(var >= 0, name="non_negativity")

        # Add minimum judge constraints
        judge_var_keys = [k for k in decision_vars.keys() if k[0] == 'Judges']
        for key in judge_var_keys:
            lp_model.addConstr(decision_vars[key] >= 1, name="Min_Judges")

        if granularity == "Country":
            # For each staff type, sum across all courts/municipalities/benches
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if key[0] == staff]
                total_current_staff = year_data[staff].sum()
                if self.debug_callback:
                    self.debug_callback(f"[DEBUG] Country: {staff} total vars: {len(total_staff_vars)}, max: {total_current_staff}")
                lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}")

            # For each court, enforce max staff per court
            for court in court_vars:
                court_data = year_data[year_data[court] == 1]
                for staff in staff_vars:
                    staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court]
                    current_staff = court_data[staff].sum()
                    if self.debug_callback:
                        self.debug_callback(f"[DEBUG] Country: {staff} in {court} vars: {len(staff_vars_)}, max: {current_staff}")
                    lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}_{court}")

            # Special constraints for Açores and Madeira
            for court in ['Court_Acores', 'Court_Madeira']:
                if court in court_vars:
                    court_data = year_data[year_data[court] == 1]
                    for island, municipalities in island_groups[court].items():
                        for staff in staff_vars:
                            staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] in municipalities]
                            total_staff = sum(court_data[court_data[mun] == 1][staff].sum() for mun in municipalities)
                            if self.debug_callback:
                                self.debug_callback(f"[DEBUG] Country: {staff} in {court} {island} vars: {len(staff_vars_)}, max: {total_staff}")
                            lp_model.addConstr(sum(staff_vars_) <= total_staff, name=f"Max_{staff}_{court}_{island}")

        elif granularity == "Court":
            if not selected_court:
                raise ValueError("Selected court is required for Court granularity")
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if key[0] == staff]
                total_current_staff = court_data[staff].sum()
                if self.debug_callback:
                    self.debug_callback(f"[DEBUG] Court: {staff} total vars: {len(total_staff_vars)}, max: {total_current_staff}")
                lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}_{court_col}")

            if selected_court in ['Acores', 'Madeira']:
                for island, municipalities in island_groups[court_col].items():
                    for staff in staff_vars:
                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] in municipalities]
                        total_staff = sum(court_data[court_data[mun] == 1][staff].sum() for mun in municipalities)
                        if self.debug_callback:
                            self.debug_callback(f"[DEBUG] Court: {staff} in {court_col} {island} vars: {len(staff_vars_)}, max: {total_staff}")
                        lp_model.addConstr(sum(staff_vars_) <= total_staff, name=f"Max_{staff}_{court_col}_{island}")
            else:
                for mun in mun_vars:
                    mun_data = court_data[court_data[mun] == 1]
                    for staff in staff_vars:
                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] == mun]
                        current_staff = mun_data[staff].sum()
                        if self.debug_callback:
                            self.debug_callback(f"[DEBUG] Court: {staff} in {mun} vars: {len(staff_vars_)}, max: {current_staff}")
                        lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}_{mun}")

        elif granularity == "Municipality":
            if not selected_court or not selected_municipality:
                raise ValueError("Selected court and municipality are required for Municipality granularity")
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]
            for staff in staff_vars:
                staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 2 and key[0] == staff]
                current_staff = mun_data[staff].sum()
                if self.debug_callback:
                    self.debug_callback(f"[DEBUG] Municipality: {staff} vars: {len(staff_vars_)}, max: {current_staff}")
                lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}")

    def calculate_staff_max(self, year_data, granularity, selected_court=None, selected_municipality=None):
        """Calculate maximum staffing possibilities based on granularity level"""
        staff_vars = self.staff_vars
        staff_max = {}

        # Get all court columns
        court_columns = [col for col in year_data.columns if col.startswith('Court_')]


        # Get all municipality columns
        mun_columns = [col for col in year_data.columns if col.startswith('Municipality_')]


        if granularity == "Country":
            # Calculate maximums for each court
            for court in court_columns:
                if court not in ["Court_Acores", "Court_Madeira"]:
                    court_data = year_data[year_data[court] == 1]
                    for staff in staff_vars:
                        staff_max[(court, staff)] = court_data[staff].sum()

                        
                

        elif granularity == "Court":
            if not selected_court:
                return {}
            # Only calculate for selected court
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            for mun in mun_columns:
                mun_data = court_data[court_data[mun] == 1]
                for staff in staff_vars:
                    staff_max[(mun,staff)] = mun_data[staff].sum()

        elif granularity == "Municipality":
            if not selected_court or not selected_municipality:
                return {}
            # Filter for selected court and municipality
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]
            # Sum across all benches
            for staff in staff_vars:
                staff_max[staff] = mun_data[staff].sum()

        return staff_max

