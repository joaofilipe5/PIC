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

            x_vars = {}
            for staff in staff_vars:
                for court, mun, bench in court_mun_bench_tuples:
                    x_vars[(staff, court, mun, bench)] = lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {court}, {mun}, {bench}")
        
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
            
            x_vars = {}
            for staff in staff_vars:
                for mun, bench in tuples:
                    x_vars[(staff, mun, bench)] = lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {mun}, {bench}")
        
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
            
            x_vars = {}
            for staff in staff_vars:
                for bench in benches:
                    x_vars[(staff, bench)] = lp_model.addVar(vtype=GRB.INTEGER, name=f"{staff}, {bench}")
            
        else:
            raise ValueError("Invalid granularity selected")
        
        if self.debug_callback:
            self.debug_callback(f"[DEBUG] Created {len(x_vars)} decision variables")
            self.debug_callback(f"[DEBUG] First few variable keys: {list(x_vars.keys())[:5]}")

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
        bench_vars = [col for col in year_data.columns if col.startswith('Bench_')]
        staff_max = self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)

        # DEBUG: Print all decision variable keys
        if self.debug_callback:
            self.debug_callback(f"[DEBUG] Decision variable keys: {list(decision_vars.keys())}")
            self.debug_callback(f"[DEBUG] Granularity: {granularity}, Selected court: {selected_court}, Selected municipality: {selected_municipality}")

        # Add non-negativity constraints for all variables
        for var in decision_vars.values():
            lp_model.addConstr(var >= 0, name="non_negativity")

        # Ensure at least one judge in every bench (applied to all granularities)
        for court in court_vars:
            court_data = year_data[year_data[court] == 1]
            for mun in mun_vars:
                if court_data[mun].any():
                    mun_data = court_data[court_data[mun] == 1]
                    for bench in bench_vars:
                        # Apply constraint only if the bench currently has at least one judge
                        if mun_data[bench].any() and mun_data['Judges'].sum() > 0:
                            if granularity == "Country":
                                judge_vars = [var for key, var in decision_vars.items()
                                             if key[0] == 'Judges' and key[1] == court and key[2] == mun and key[3] == bench]
                            elif granularity == "Court":
                                judge_vars = [var for key, var in decision_vars.items()
                                             if key[0] == 'Judges' and key[1] == mun and key[2] == bench]
                            else:  # Municipality
                                judge_vars = [var for key, var in decision_vars.items()
                                             if key[0] == 'Judges' and key[1] == bench]
                            if judge_vars:
                                lp_model.addConstr(sum(judge_vars) >= 1, name=f"Min_Judges_{court}_{mun}_{bench}")

        # Define island groups for Açores and Madeira
        island_groups = {
            'Court_Acores': {
                'Sao_Miguel': ['Municipality_Ponta Delgada', 'Municipality_Vila Franca do Campo', 'Municipality_Ribeira Grande'],
                'Terceira': ['Municipality_Angra do Heroismo', 'Municipality_Vila Praia da Vitoria'],
                'Pico': ['Municipality_Sao Roque do Pico'],
                'Faial': ['Municipality_Horta'],
                'Flores': ['Municipality_Santa Cruz das Flores'],
                'Graciosa': ['Municipality_Santa Cruz  Graciosa (R.A.A)'],
                'Sao_Jorge': ['Municipality_Velas (R.A.A.)'],
                'Santa_Maria': ['Municipality_Vila do Porto']
            },
            'Court_Madeira': {
                'Madeira': ['Municipality_Funchal', 'Municipality_Ponta do Sol', 'Municipality_Santa Cruz'],
                'Porto_Santo': ['Municipality_Porto Santo']
            }
        }

        # Define bench groups for similar benches
        bench_groups = {
            'Civil': ['Bench_Central Civil', 'Bench_Local Civil', 'Bench_Central Civil and Criminal', 'Bench_Generic'],
            'Criminal': ['Bench_Central Criminal', 'Bench_Local Criminal', 'Bench_Central Civil and Criminal', 'Bench_Criminal Instruction', 'Bench_Small Criminal', 'Bench_Generic'],
            'Family': ['Bench_Family and Minors', 'Bench_Family/Minors and Labor', 'Bench_Generic'],
            'Labor': ['Bench_Labor', 'Bench_Family/Minors and Labor', 'Bench_Generic'],
            'Commerce': ['Bench_Commerce', 'Bench_Generic'],
            'Execution': ['Bench_Execution', 'Bench_Generic'],
            'Generic': ['Bench_Central Civil', 'Bench_Local Civil', 'Bench_Central Civil and Criminal', 
                       'Bench_Central Criminal', 'Bench_Local Criminal', 'Bench_Criminal Instruction', 
                       'Bench_Small Criminal', 'Bench_Family and Minors', 'Bench_Family/Minors and Labor',
                       'Bench_Labor', 'Bench_Commerce', 'Bench_Execution', 'Bench_Generic']
        }

        if granularity == "Country":
            # Global staff conservation constraint (Country level)
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff]
                total_current_staff = year_data[staff].sum()
                if total_staff_vars:
                    lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}_Country")
                    self.log_debug(f"[DEBUG] Added total staff constraint for Country: {staff} <= {total_current_staff}")

            # For each staff type, total staff in each court/island group should not exceed current staff
            for court in court_vars:
                if court not in ['Court_Acores', 'Court_Madeira']:
                    court_data = year_data[year_data[court] == 1]
                    for staff in staff_vars:
                        # Total staff constraint for the court
                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court]
                        current_staff = court_data[staff].sum()
                        if staff_vars_:
                            lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_Total_{staff}_{court}")
                            self.log_debug(f"[DEBUG] Added total staff constraint for {court}: {staff} <= {current_staff}")

                        # Add bench movement constraints within court's municipalities
                        for mun in mun_vars:
                            if court_data[mun].any():
                                mun_data = court_data[court_data[mun] == 1]
                                # Total staff constraint for the municipality
                                mun_staff_vars = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun]
                                current_mun_staff = mun_data[staff].sum()
                                if mun_staff_vars:
                                    lp_model.addConstr(sum(mun_staff_vars) <= current_mun_staff, name=f"Max_Total_{staff}_{court}_{mun}")
                                    self.log_debug(f"[DEBUG] Added total staff constraint for {court} - {mun}: {staff} <= {current_mun_staff}")

                                for bench_group, allowed_benches in bench_groups.items():
                                    self.log_debug(f"[DEBUG] Country (Non-Island) Bench Movement: Court={court}, Municipality={mun}, Staff={staff}, Bench Group={bench_group}")
                                    existing_benches = [b for b in allowed_benches if b in bench_vars and mun_data[b].any()]
                                    self.log_debug(f"[DEBUG]   Existing Benches in Municipality: {existing_benches}")
                                    if len(existing_benches) > 1:
                                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun and key[3] in existing_benches]
                                        self.log_debug(f"[DEBUG]   Variables in constraint: {[str(v) for v in staff_vars_]}")
                                        if staff_vars_:
                                            lp_model.addConstr(sum(staff_vars_) <= sum(mun_data[staff].sum() for b in existing_benches), name=f"Bench_Movement_Country_{court}_{mun}_{staff}_{bench_group}")
                                            self.log_debug(f"[DEBUG]   Added constraint: {f'Bench_Movement_Country_{court}_{mun}_{staff}_{bench_group}'}")

            # For Açores and Madeira
            for court in ['Court_Acores', 'Court_Madeira']:
                if court in court_vars:
                    court_data = year_data[year_data[court] == 1]
                    for island, municipalities in island_groups[court].items():
                        # Filter data for this island's municipalities
                        island_data = pd.DataFrame()
                        for mun in municipalities:
                            if court_data[mun].any():
                                island_data = pd.concat([island_data, court_data[court_data[mun] == 1]])
                        if not island_data.empty:
                            for staff in staff_vars:
                                # Total staff constraint for the island
                                island_staff_vars = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] in municipalities]
                                current_island_staff = island_data[staff].sum()
                                if island_staff_vars:
                                    lp_model.addConstr(sum(island_staff_vars) <= current_island_staff, name=f"Max_Total_{staff}_{court}_{island}")
                                    self.log_debug(f"[DEBUG] Added total staff constraint for {court} - {island}: {staff} <= {current_island_staff}")

                                # Add bench movement constraints within island municipalities
                                for mun in municipalities:
                                    if court_data[mun].any():
                                        mun_data = court_data[court_data[mun] == 1]
                                        # Total staff constraint for the municipality
                                        mun_staff_vars = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun]
                                        current_mun_staff = mun_data[staff].sum()
                                        if mun_staff_vars:
                                            lp_model.addConstr(sum(mun_staff_vars) <= current_mun_staff, name=f"Max_Total_{staff}_{court}_{mun}")
                                            self.log_debug(f"[DEBUG] Added total staff constraint for {court} - {mun}: {staff} <= {current_mun_staff}")

                                        for bench_group, allowed_benches in bench_groups.items():
                                            self.log_debug(f"[DEBUG] Country (Island) Bench Movement: Court={court}, Island={island}, Municipality={mun}, Staff={staff}, Bench Group={bench_group}")
                                            existing_benches = [b for b in allowed_benches if b in bench_vars and mun_data[b].any()]
                                            self.log_debug(f"[DEBUG]   Existing Benches in Municipality: {existing_benches}")
                                            if len(existing_benches) > 1:
                                                staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun and key[3] in existing_benches]
                                                self.log_debug(f"[DEBUG]   Variables in constraint: {[str(v) for v in staff_vars_]}")
                                                if staff_vars_:
                                                    lp_model.addConstr(sum(staff_vars_) <= sum(mun_data[staff].sum() for b in existing_benches), name=f"Bench_Movement_Country_{court}_{island}_{mun}_{staff}_{bench_group}")
                                                    self.log_debug(f"[DEBUG]   Added constraint: {f'Bench_Movement_Country_{court}_{island}_{mun}_{staff}_{bench_group}'}")

        elif granularity == "Court":
            if not selected_court:
                raise ValueError("Selected court is required for Court granularity")
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]

            # Global staff conservation constraint (Court level)
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff]
                total_current_staff = court_data[staff].sum()
                if total_staff_vars:
                    lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}_{selected_court}")
                    self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court}: {staff} <= {total_current_staff}")

            if selected_court in ['Acores', 'Madeira']:
                # Bench movement constraints within island municipalities
                for island, municipalities in island_groups[court_col].items():
                    # Filter data for this island's municipalities
                    island_data = pd.DataFrame()
                    for mun in municipalities:
                        if court_data[mun].any():
                            island_data = pd.concat([island_data, court_data[court_data[mun] == 1]])
                    if not island_data.empty:
                        for staff in staff_vars:
                            # Total staff constraint for the island
                            island_staff_vars = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] in municipalities]
                            current_island_staff = island_data[staff].sum()
                            if island_staff_vars:
                                lp_model.addConstr(sum(island_staff_vars) <= current_island_staff, name=f"Max_Total_{staff}_{selected_court}_{island}")
                                self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {island}: {staff} <= {current_island_staff}")

                            # Add bench movement constraints within island municipalities
                            for mun in municipalities:
                                if court_data[mun].any():
                                    mun_data = court_data[court_data[mun] == 1]
                                    # Total staff constraint for the municipality
                                    mun_staff_vars = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] == mun]
                                    current_mun_staff = mun_data[staff].sum()
                                    if mun_staff_vars:
                                        lp_model.addConstr(sum(mun_staff_vars) <= current_mun_staff, name=f"Max_Total_{staff}_{selected_court}_{mun}")
                                        self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {mun}: {staff} <= {current_mun_staff}")

                                    for bench_group, allowed_benches in bench_groups.items():
                                        self.log_debug(f"[DEBUG] Court (Island) Bench Movement: Court={selected_court}, Island={island}, Municipality={mun}, Staff={staff}, Bench Group={bench_group}")
                                        existing_benches = [b for b in allowed_benches if b in bench_vars and mun_data[b].any()]
                                        self.log_debug(f"[DEBUG]   Existing Benches in Municipality: {existing_benches}")
                                        if len(existing_benches) > 1:
                                            staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] == mun and key[2] in existing_benches]
                                            self.log_debug(f"[DEBUG]   Variables in constraint: {[str(v) for v in staff_vars_]}")
                                            if staff_vars_:
                                                lp_model.addConstr(sum(staff_vars_) <= sum(mun_data[staff].sum() for b in existing_benches), name=f"Bench_Movement_Court_{selected_court}_{island}_{mun}_{staff}_{bench_group}")
                                                self.log_debug(f"[DEBUG]   Added constraint: {f'Bench_Movement_Court_{selected_court}_{island}_{mun}_{staff}_{bench_group}'}")

            else:
                # Bench movement constraints within municipality
                for mun in mun_vars:
                    if court_data[mun].any():
                        mun_data = court_data[court_data[mun] == 1]
                        for staff in staff_vars:
                            # Total staff constraint for the municipality
                            mun_staff_vars = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] == mun]
                            current_mun_staff = mun_data[staff].sum()
                            if mun_staff_vars:
                                lp_model.addConstr(sum(mun_staff_vars) <= current_mun_staff, name=f"Max_Total_{staff}_{selected_court}_{mun}")
                                self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {mun}: {staff} <= {current_mun_staff}")

                            for bench_group, allowed_benches in bench_groups.items():
                                self.log_debug(f"[DEBUG] Court (Non-Island) Bench Movement: Court={selected_court}, Municipality={mun}, Staff={staff}, Bench Group={bench_group}")
                                existing_benches = [b for b in allowed_benches if b in bench_vars and mun_data[b].any()]
                                self.log_debug(f"[DEBUG]   Existing Benches in Municipality: {existing_benches}")
                                if len(existing_benches) > 1:
                                    staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff and key[1] == mun and key[2] in existing_benches]
                                    self.log_debug(f"[DEBUG]   Variables in constraint: {[str(v) for v in staff_vars_]}")
                                    if staff_vars_:
                                        lp_model.addConstr(sum(staff_vars_) <= sum(mun_data[staff].sum() for b in existing_benches), name=f"Bench_Movement_Court_{selected_court}_{mun}_{staff}_{bench_group}")
                                        self.log_debug(f"[DEBUG]   Added constraint: {f'Bench_Movement_Court_{selected_court}_{mun}_{staff}_{bench_group}'}")

        elif granularity == "Municipality":
            if not selected_court or not selected_municipality:
                raise ValueError("Selected court and municipality are required for Municipality granularity")
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]

            # Global staff conservation constraint (Municipality level)
            for staff in staff_vars:
                staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 2 and key[0] == staff]
                current_staff = mun_data[staff].sum()
                if staff_vars_:
                    lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_Total_{staff}_{selected_court}_{selected_municipality}")
                    self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {selected_municipality}: {staff} <= {current_staff}")

            # Add bench movement constraints within municipality
            for staff in staff_vars:
                for bench_group, allowed_benches in bench_groups.items():
                    self.log_debug(f"[DEBUG] Municipality Bench Movement: Court={selected_court}, Municipality={selected_municipality}, Staff={staff}, Bench Group={bench_group}")
                    existing_benches = [b for b in allowed_benches if b in bench_vars and mun_data[b].any()]
                    self.log_debug(f"[DEBUG]   Existing Benches in Municipality: {existing_benches}")
                    if len(existing_benches) > 1:
                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 2 and key[0] == staff and key[1] in existing_benches]
                        self.log_debug(f"[DEBUG]   Variables in constraint: {[str(v) for v in staff_vars_]}")
                        if staff_vars_:
                            current_total = sum(mun_data[staff].sum() for b in existing_benches)
                            lp_model.addConstr(sum(staff_vars_) <= current_total, name=f"Bench_Movement_{selected_court}_{selected_municipality}_{staff}_{bench_group}")
                            self.log_debug(f"[DEBUG]   Added constraint: {f'Bench_Movement_{selected_court}_{selected_municipality}_{staff}_{bench_group}'}")

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

