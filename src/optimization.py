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
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]
        bench_vars = [col for col in year_data.columns if col.startswith('Bench_')]

        # Create coefficient matrix with target variables as rows
        coef_df = pd.DataFrame(
            ml_model.coef_,
            columns=year_data.columns,
            index=target_specific
        )

        intercepts = pd.Series(ml_model.intercept_,
                               index = target_specific)

        # Calculate ML model prediction with original allocations for comparison
        original_prediction = 0
        if granularity == "Country":
            original_prediction = ml_model.predict(year_data).sum()
        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            original_prediction = ml_model.predict(court_data).sum()
        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            original_prediction = ml_model.predict(mun_data).sum()

        # Initialize expressions for each target variable
        target_expressions = {target: [] for target in target_specific}

        if granularity == "Country":
            for i in range(year_data.shape[0]):
                row = year_data.iloc[i]
                try:
                    court = row[court_vars].idxmax()
                    mun = row[mun_vars].idxmax()
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    continue
                
                for target in target_specific:
                    expr = intercepts[target]
                    
                    # Add staff variable terms
                    for staff in self.staff_vars:
                        expr += coef_df.loc[target, staff] * decision_vars[(staff, court, mun, bench)]
                    
                    # Add non-staff variable terms
                    for feature in year_data.columns:
                        if feature not in self.staff_vars:
                            expr += coef_df.loc[target, feature] * row[feature]
                    
                    target_expressions[target].append(expr)

        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            
            for i in range(court_data.shape[0]):
                row = court_data.iloc[i]
                try:
                    mun = row[mun_vars].idxmax()
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    continue
                
                for target in target_specific:
                    expr = intercepts[target]
                    
                    # Add staff variable terms
                    for staff in self.staff_vars:
                        expr += coef_df.loc[target, staff] * decision_vars[(staff, mun, bench)]
                    
                    # Add non-staff variable terms
                    for feature in year_data.columns:
                        if feature not in self.staff_vars:
                            expr += coef_df.loc[target, feature] * row[feature]
                    
                    
                    target_expressions[target].append(expr)

        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            
            for i in range(mun_data.shape[0]):
                row = mun_data.iloc[i]
                try:
                    bench = row[bench_vars].idxmax()
                except Exception as e:
                    continue
                
                for target in target_specific:
                    expr = intercepts[target]
                    
                    # Add staff variable terms
                    for staff in self.staff_vars:
                        expr += coef_df.loc[target, staff] * decision_vars[(staff, bench)]
                    
                    # Add non-staff variable terms
                    for feature in year_data.columns:
                        if feature not in self.staff_vars:
                            expr += coef_df.loc[target, feature] * row[feature]
                    
                    target_expressions[target].append(expr)

        # Combine all target expressions
        outputs = []
        for target in target_specific:
            outputs.extend(target_expressions[target])

        return outputs

    def calculate_staff_max(self, year_data, granularity, selected_court=None, selected_municipality=None):
        # Implementation of calculate_staff_max method
        pass

    def add_constraints(self, lp_model: Model, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        """Add constraints to the optimization model"""
        staff_vars = self.staff_vars
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]
        bench_vars = [col for col in year_data.columns if col.startswith('Bench_')]

        # DEBUG: Print all decision variable keys
        if self.debug_callback:
            self.debug_callback(f"[DEBUG] Decision variable keys: {list(decision_vars.keys())}")
            self.debug_callback(f"[DEBUG] Granularity: {granularity}, Selected court: {selected_court}, Selected municipality: {selected_municipality}")

        # Add non-negativity constraints for all variables
        for var in decision_vars.values():
            try:
                lp_model.addConstr(var >= 0, name="non_negativity")
            except Exception as e:
                self.log_debug(f"[DEBUG] Error adding non-negativity constraint: {str(e)}")

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

        if granularity == "Country":
            # Global staff conservation constraint (Country level)
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff]
                total_current_staff = year_data[staff].sum()
                if total_staff_vars:
                    try:
                        lp_model.addConstr(sum(total_staff_vars) == total_current_staff, name=f"Max_Total_{staff}_Country")
                        self.log_debug(f"[DEBUG] Added total staff constraint for Country: {staff} <= {total_current_staff}")
                    except Exception as e:
                        self.log_debug(f"[DEBUG] Error adding total staff constraint for Country - {staff}: {str(e)}")

            # For each staff type, total staff in each court/island group should not exceed current staff
            for court in court_vars:
                if court not in ['Court_Acores', 'Court_Madeira']:
                    court_data = year_data[year_data[court] == 1]
                    for staff in staff_vars:
                        # Total staff constraint for the court
                        staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 4 and key[0] == staff and key[1] == court]
                        current_staff = court_data[staff].sum()
                        if staff_vars_:
                            try:
                                lp_model.addConstr(sum(staff_vars_) == current_staff, name=f"Max_Total_{staff}_{court}")
                                self.log_debug(f"[DEBUG] Added total staff constraint for {court}: {staff} <= {current_staff}")
                            except Exception as e:
                                self.log_debug(f"[DEBUG] Error adding total staff constraint for {court} - {staff}: {str(e)}")

                        # Add max staff per bench constraint for each municipality
                        for mun in mun_vars:
                            if court_data[mun].any():
                                mun_data = court_data[court_data[mun] == 1]
                                # Count number of benches in this municipality
                                num_benches = sum(1 for bench in bench_vars if bench in mun_data.columns and mun_data[bench].any())
                                if num_benches > 0:
                                    max_staff_per_bench = 2 * (current_staff // num_benches + (1 if current_staff % num_benches else 0))
                                    for bench in bench_vars:
                                        if bench in mun_data.columns and mun_data[bench].any():
                                            bench_var = [var for key, var in decision_vars.items() 
                                                       if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun and key[3] == bench]
                                            if bench_var:
                                                try:
                                                    lp_model.addConstr(bench_var[0] <= max_staff_per_bench, 
                                                                    name=f"Max_Staff_Per_Bench_{court}_{mun}_{bench}_{staff}")
                                                    self.log_debug(f"[DEBUG] Added max staff per bench constraint for {court} - {mun} - {bench}: {staff} <= {max_staff_per_bench}")
                                                except Exception as e:
                                                    self.log_debug(f"[DEBUG] Error adding max staff per bench constraint for {court} - {mun} - {bench} - {staff}: {str(e)}")

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
                                    try:
                                        lp_model.addConstr(sum(island_staff_vars) == current_island_staff, name=f"Max_Total_{staff}_{court}_{island}")
                                        self.log_debug(f"[DEBUG] Added total staff constraint for {court} - {island}: {staff} <= {current_island_staff}")
                                    except Exception as e:
                                        self.log_debug(f"[DEBUG] Error adding total staff constraint for {court} - {island} - {staff}: {str(e)}")

                                # Add max staff per bench constraint for each municipality in the island
                                for mun in municipalities:
                                    if court_data[mun].any():
                                        mun_data = court_data[court_data[mun] == 1]
                                        # Count number of benches in this municipality
                                        num_benches = sum(1 for bench in bench_vars if bench in mun_data.columns and mun_data[bench].any())
                                        if num_benches > 0:
                                            max_staff_per_bench = 2 * (current_island_staff // num_benches + (1 if current_island_staff % num_benches else 0))
                                            for bench in bench_vars:
                                                if bench in mun_data.columns and mun_data[bench].any():
                                                    bench_var = [var for key, var in decision_vars.items() 
                                                               if len(key) == 4 and key[0] == staff and key[1] == court and key[2] == mun and key[3] == bench]
                                                    if bench_var:
                                                        try:
                                                            lp_model.addConstr(bench_var[0] <= max_staff_per_bench, 
                                                                            name=f"Max_Staff_Per_Bench_{court}_{mun}_{bench}_{staff}")
                                                            self.log_debug(f"[DEBUG] Added max staff per bench constraint for {court} - {mun} - {bench}: {staff} <= {max_staff_per_bench}")
                                                        except Exception as e:
                                                            self.log_debug(f"[DEBUG] Error adding max staff per bench constraint for {court} - {mun} - {bench} - {staff}: {str(e)}")

            # Add minimum judges constraint for all courts
            for court in court_vars:
                court_data = year_data[year_data[court] == 1]
                for mun in mun_vars:
                    if court_data[mun].any():
                        mun_data = court_data[court_data[mun] == 1]
                        for bench in bench_vars:
                            if bench in mun_data.columns and mun_data[bench].any():
                                # Check if this bench originally had judges
                                if mun_data.loc[mun_data[bench] == 1, 'Judges'].sum() > 0:
                                    bench_judge_vars = [var for key, var in decision_vars.items() 
                                                      if len(key) == 4 and key[0] == 'Judges' and key[1] == court and key[2] == mun and key[3] == bench]
                                    if bench_judge_vars:
                                        try:
                                            lp_model.addConstr(sum(bench_judge_vars) >= 1, 
                                                            name=f"Min_Judges_{court}_{mun}_{bench}")
                                            self.log_debug(f"[DEBUG] Added minimum judges constraint for {court} - {mun} - {bench}")
                                        except Exception as e:
                                            self.log_debug(f"[DEBUG] Error adding minimum judges constraint for {court} - {mun} - {bench}: {str(e)}")

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
                    try:
                        lp_model.addConstr(sum(total_staff_vars) == total_current_staff, name=f"Max_Total_{staff}_{selected_court}")
                        self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court}: {staff} <= {total_current_staff}")
                    except Exception as e:
                        self.log_debug(f"[DEBUG] Error adding total staff constraint for {selected_court} - {staff}: {str(e)}")

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
                                try:
                                    lp_model.addConstr(sum(island_staff_vars) == current_island_staff, name=f"Max_Total_{staff}_{selected_court}_{island}")
                                    self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {island}: {staff} <= {current_island_staff}")
                                except Exception as e:
                                    self.log_debug(f"[DEBUG] Error adding total staff constraint for {selected_court} - {island} - {staff}: {str(e)}")

                            # Add max staff per bench constraint for each municipality in the island
                            for mun in municipalities:
                                if court_data[mun].any():
                                    mun_data = court_data[court_data[mun] == 1]
                                    # Count number of benches in this municipality
                                    num_benches = sum(1 for bench in bench_vars if bench in mun_data.columns and mun_data[bench].any())
                                    if num_benches > 0:
                                        max_staff_per_bench = 2 * (current_island_staff // num_benches + (1 if current_island_staff % num_benches else 0))
                                        for bench in bench_vars:
                                            if bench in mun_data.columns and mun_data[bench].any():
                                                bench_var = [var for key, var in decision_vars.items() 
                                                           if len(key) == 3 and key[0] == staff and key[1] == mun and key[2] == bench]
                                                if bench_var:
                                                    try:
                                                        lp_model.addConstr(bench_var[0] <= max_staff_per_bench, 
                                                                        name=f"Max_Staff_Per_Bench_{selected_court}_{mun}_{bench}_{staff}")
                                                        self.log_debug(f"[DEBUG] Added max staff per bench constraint for {selected_court} - {mun} - {bench}: {staff} <= {max_staff_per_bench}")
                                                    except Exception as e:
                                                        self.log_debug(f"[DEBUG] Error adding max staff per bench constraint for {selected_court} - {mun} - {bench} - {staff}: {str(e)}")

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
                                try:
                                    lp_model.addConstr(sum(mun_staff_vars) == current_mun_staff, name=f"Max_Total_{staff}_{selected_court}_{mun}")
                                    self.log_debug(f"[DEBUG] Added total staff constraint for {selected_court} - {mun}: {staff} <= {current_mun_staff}")
                                except Exception as e:
                                    self.log_debug(f"[DEBUG] Error adding total staff constraint for {selected_court} - {mun} - {staff}: {str(e)}")

                            # Add max staff per bench constraint
                            num_benches = sum(1 for bench in bench_vars if bench in mun_data.columns and mun_data[bench].any())
                            if num_benches > 0:
                                max_staff_per_bench = 2 * (current_mun_staff // num_benches + (1 if current_mun_staff % num_benches else 0))
                                for bench in bench_vars:
                                    if bench in mun_data.columns and mun_data[bench].any():
                                        bench_var = [var for key, var in decision_vars.items() 
                                                   if len(key) == 3 and key[0] == staff and key[1] == mun and key[2] == bench]
                                        if bench_var:
                                            try:
                                                lp_model.addConstr(bench_var[0] <= max_staff_per_bench, 
                                                                name=f"Max_Staff_Per_Bench_{selected_court}_{mun}_{bench}_{staff}")
                                                self.log_debug(f"[DEBUG] Added max staff per bench constraint for {selected_court} - {mun} - {bench}: {staff} <= {max_staff_per_bench}")
                                            except Exception as e:
                                                self.log_debug(f"[DEBUG] Error adding max staff per bench constraint for {selected_court} - {mun} - {bench} - {staff}: {str(e)}")

            # Add minimum judges constraint for all municipalities in the court
            for mun in mun_vars:
                if court_data[mun].any():
                    mun_data = court_data[court_data[mun] == 1]
                    for bench in bench_vars:
                        if bench in mun_data.columns and mun_data[bench].any():
                            # Check if this bench originally had judges
                            if mun_data.loc[mun_data[bench] == 1, 'Judges'].sum() > 0:
                                bench_judge_vars = [var for key, var in decision_vars.items() 
                                                  if len(key) == 3 and key[0] == 'Judges' and key[1] == mun and key[2] == bench]
                                if bench_judge_vars:
                                    try:
                                        lp_model.addConstr(sum(bench_judge_vars) >= 1, 
                                                        name=f"Min_Judges_{selected_court}_{mun}_{bench}")
                                        self.log_debug(f"[DEBUG] Added minimum judges constraint for {selected_court} - {mun} - {bench}")
                                    except Exception as e:
                                        self.log_debug(f"[DEBUG] Error adding minimum judges constraint for {selected_court} - {mun} - {bench}: {str(e)}")

        elif granularity == "Municipality":
            if not selected_court or not selected_municipality:
                raise ValueError("Selected court and municipality are required for Municipality granularity")
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]

            # Get all benches that exist in this municipality
            existing_benches = [bench for bench in bench_vars if mun_data[bench].any()]
            self.log_debug(f"[DEBUG] Municipality {selected_municipality} has {len(existing_benches)} benches: {existing_benches}")

            # Global staff conservation constraint (Municipality level)
            for staff in staff_vars:
                staff_vars_ = [var for key, var in decision_vars.items() if len(key) == 2 and key[0] == staff]
                current_staff = mun_data[staff].sum()
                self.log_debug(f"[DEBUG] Municipality {selected_municipality} has {current_staff} {staff}")
                self.log_debug(f"[DEBUG] Number of staff variables: {len(staff_vars_)}")
                
                if staff_vars_:
                    try:
                        # Ensure total staff is maintained exactly
                        lp_model.addConstr(sum(staff_vars_) == current_staff, 
                                        name=f"Total_Staff_{selected_court}_{selected_municipality}_{staff}")
                        self.log_debug(f"[DEBUG] Added total staff constraint for {selected_municipality} - {staff}: <= {current_staff}")
                    except Exception as e:
                        self.log_debug(f"[DEBUG] Error adding total staff constraint for {selected_municipality} - {staff}: {str(e)}")

                    # Add max staff per bench constraint
                    num_benches = len(existing_benches)
                    if num_benches > 0:
                        max_staff_per_bench = 2 * (current_staff // num_benches + (1 if current_staff % num_benches else 0))
                        for bench in existing_benches:
                            bench_var = [var for key, var in decision_vars.items() 
                                       if len(key) == 2 and key[0] == staff and key[1] == bench]
                            if bench_var:
                                try:
                                    lp_model.addConstr(bench_var[0] <= max_staff_per_bench, 
                                                    name=f"Max_Staff_Per_Bench_{selected_court}_{selected_municipality}_{bench}_{staff}")
                                    self.log_debug(f"[DEBUG] Added max staff per bench constraint for {selected_municipality} - {bench}: {staff} <= {max_staff_per_bench}")
                                except Exception as e:
                                    self.log_debug(f"[DEBUG] Error adding max staff per bench constraint for {selected_municipality} - {bench} - {staff}: {str(e)}")

            # Add minimum judges constraint for all benches in the municipality
            for bench in existing_benches:
                # Check if this bench originally had judges
                if mun_data.loc[mun_data[bench] == 1, 'Judges'].sum() > 0:
                    bench_judge_vars = [var for key, var in decision_vars.items() 
                                      if len(key) == 2 and key[0] == 'Judges' and key[1] == bench]
                    if bench_judge_vars:
                        try:
                            lp_model.addConstr(sum(bench_judge_vars) >= 1, 
                                            name=f"Min_Judges_{selected_court}_{selected_municipality}_{bench}")
                            self.log_debug(f"[DEBUG] Added minimum judges constraint for {selected_municipality} - {bench}")
                        except Exception as e:
                            self.log_debug(f"[DEBUG] Error adding minimum judges constraint for {selected_municipality} - {bench}: {str(e)}")

