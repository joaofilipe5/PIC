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
        staff_max = self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)

        # DEBUG: Print all decision variable keys
        if self.debug_callback:
            self.debug_callback(f"[DEBUG] Decision variable keys: {list(decision_vars.keys())}")
            self.debug_callback(f"[DEBUG] Granularity: {granularity}, Selected court: {selected_court}, Selected municipality: {selected_municipality}")

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

        # Add non-negativity constraints for all variables
        for var in decision_vars.values():
            lp_model.addConstr(var >= 0, name="non_negativity")

        if granularity == "Country":
            # For each staff type, sum across all courts/municipalities/benches
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if key[0] == staff]
                total_current_staff = year_data[staff].sum()
                if self.debug_callback:
                    self.debug_callback(f"[DEBUG] Country: {staff} total vars: {len(total_staff_vars)}, max: {total_current_staff}")
                if total_staff_vars:
                    lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}")

            # For each court, enforce max staff per court
            for court in court_vars:
                court_data = year_data[year_data[court] == 1]
                for staff in staff_vars:
                    staff_vars_ = [var for key, var in decision_vars.items() if key[0] == staff and key[1] == court]
                    current_staff = court_data[staff].sum()
                    if self.debug_callback:
                        self.debug_callback(f"[DEBUG] Country: {staff} in {court} vars: {len(staff_vars_)}, max: {current_staff}")
                    if staff_vars_:
                        lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}_{court}")

            # Special constraints for Açores and Madeira
            for court in ['Court_Acores', 'Court_Madeira']:
                if court in court_vars:
                    court_data = year_data[year_data[court] == 1]
                    for island, municipalities in island_groups[court].items():
                        if len(municipalities) == 1:
                            # Only one municipality in this island: staff must stay in that municipality
                            mun = municipalities[0]
                            mun_data = court_data[court_data[mun] == 1]
                            for staff in staff_vars:
                                # Filter variables for this staff type and the single municipality in this island
                                staff_vars_mun = [var for key, var in decision_vars.items() 
                                                  if key[0] == staff and key[1] == court and key[2] == mun]
                                current_staff = mun_data[staff].sum()
                                if self.debug_callback:
                                    self.debug_callback(f"[DEBUG] Country (Single Mun Island): {staff} in {court} {island} ({mun}) - Vars: {len(staff_vars_mun)}, Max: {current_staff}, Var Keys: {[k for k in decision_vars.keys() if k[0] == staff and k[1] == court and k[2] == mun]}")
                                if staff_vars_mun:
                                    lp_model.addConstr(sum(staff_vars_mun) <= current_staff, name=f"Max_{staff}_{court}_{island}_{mun}_single")
                        else:
                            # Multiple municipalities: allow movement within the island
                            island_staff = {}
                            for staff in staff_vars:
                                island_staff[staff] = sum(court_data[court_data[mun] == 1][staff].sum() for mun in municipalities)
                                if self.debug_callback:
                                    self.log_debug(f"[DEBUG] Country (Multi Mun Island): {staff} in {court} {island} - Current Total: {island_staff[staff]}, Municipalities: {municipalities}")
                            for staff in staff_vars:
                                # Filter variables for this staff type, court, and all municipalities in this island group
                                staff_vars_ = [var for key, var in decision_vars.items() 
                                               if key[0] == staff and key[1] == court and key[2] in municipalities]
                                if self.debug_callback:
                                    var_keys = [k for k in decision_vars.keys() if len(k) == 4 and k[0] == staff and k[1] == court and k[2] in municipalities]
                                    self.debug_callback(f"[DEBUG] Country (Multi Mun Island): {staff} in {court} {island} - Vars: {len(staff_vars_)}, Max: {island_staff[staff]}, Var Keys: {var_keys}")
                                if staff_vars_:
                                    lp_model.addConstr(sum(staff_vars_) <= island_staff[staff], name=f"Max_{staff}_{court}_{island}")

            # Constraints for regular courts (not Açores or Madeira) - Allow staff movement within the court
            for court in court_vars:
                if court not in ['Court_Acores', 'Court_Madeira']:
                    court_data = year_data[year_data[court] == 1]
                    for staff in staff_vars:
                        # Filter variables for this staff type and court (across all municipalities/benches within this court)
                        staff_vars_ = [var for key, var in decision_vars.items() 
                                       if key[0] == staff and key[1] == court]
                        current_staff = court_data[staff].sum() # Sum staff across all municipalities in this court
                        if self.debug_callback:
                             var_keys = [k for k in decision_vars.keys() if k[0] == staff and k[1] == court]
                             self.debug_callback(f"[DEBUG] Country (Regular Court): {staff} in {court} - Vars: {len(staff_vars_)}, Max: {current_staff}, Var Keys: {var_keys}")
                        if staff_vars_:
                            lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}_{court}")

            # Minimum allocation constraint for Judges: Ensure municipalities with judge variables get appropriate judge allocation
            self.log_debug("[DEBUG] Adding refined minimum judge allocation constraints")
            for staff in staff_vars:
                if staff == 'Judges':
                    judge_vars_by_mun = {}
                    judge_vars = [(key, var) for key, var in decision_vars.items() if key[0] == 'Judges']
                    for key, var in judge_vars:
                        court = key[1]
                        mun = key[2]
                        if (court, mun) not in judge_vars_by_mun:
                            judge_vars_by_mun[(court, mun)] = []
                        judge_vars_by_mun[(court, mun)].append(var)

                    # Add constraint for each municipality with judge variables
                    for (court, mun), vars_list in judge_vars_by_mun.items():
                        if vars_list:
                            # Get the data for the specific municipality to check current judges
                            mun_data = year_data[(year_data[court] == 1) & (year_data[mun] == 1)]
                            current_judges_in_mun = mun_data['Judges'].sum()
                            num_judge_vars_in_mun = len(vars_list)

                            if num_judge_vars_in_mun > 1:
                                if current_judges_in_mun > 0:
                                    # If municipality has multiple benches and current judges, encourage distribution
                                    min_allocation = min(current_judges_in_mun + 1, num_judge_vars_in_mun)
                                    lp_model.addConstr(sum(vars_list) >= min_allocation, name=f"Min_Judges_Distr_{court}_{mun}")
                                    if self.debug_callback:
                                        self.log_debug(f"[DEBUG]   Min Judges Distribution in {court} {mun}: Sum of {num_judge_vars_in_mun} vars >= {min_allocation} (Current judges: {current_judges_in_mun})")
                                else:
                                    # If municipality has multiple benches but no current judges, ensure at least 1
                                    lp_model.addConstr(sum(vars_list) >= 1, name=f"Min_Judges_in_Municipality_{court}_{mun}")
                                    if self.debug_callback:
                                        self.log_debug(f"[DEBUG]   Min Judges in Municipality {court} {mun}: Sum of {num_judge_vars_in_mun} vars >= 1 (Current judges: {current_judges_in_mun})")
                            elif num_judge_vars_in_mun > 0:
                                # If municipality has only one bench with judge variables, ensure at least 1
                                lp_model.addConstr(sum(vars_list) >= 1, name=f"Min_Judges_in_Municipality_{court}_{mun}")
                                if self.debug_callback:
                                    self.log_debug(f"[DEBUG]   Min Judges in Municipality {court} {mun}: Sum of {num_judge_vars_in_mun} vars >= 1 (Current judges: {current_judges_in_mun})")

            # Constraint: Keep staff inside similar benches in multi-bench municipalities
            self.log_debug("[DEBUG] Adding constraint: Keep staff inside similar benches")
            judicial_staff = ['Judges', 'Justice Secretary', 'Law Clerck']
            admin_staff = ['Auxiliar Clerck', 'Administrative/Technical People', 'Operational/Auxiliar People']
            bench_categories = {
                'Civil': ['Central Civil', 'Local Civil'],
                'Criminal': ['Central Criminal', 'Local Criminal'],
                'Labor': ['Labor'],
                'Family': ['Family and Minors'],
                'Trade': ['Trade']
            }

            for court in court_vars:
                court_data = year_data[year_data[court] == 1]
                mun_vars_in_court = [col for col in court_data.columns if col.startswith('Municipality_')]

                for mun in mun_vars_in_court:
                    mun_data = court_data[court_data[mun] == 1]
                    bench_vars_in_mun = [col for col in mun_data.columns if col.startswith('Bench_') and mun_data[col].any()]

                    # Identify municipalities with multiple benches excluding 'Generic'
                    specialized_benches_in_mun = [b for b in bench_vars_in_mun if b != 'Bench_Generic']

                    if len(specialized_benches_in_mun) > 0:
                         # Constraint for staff allocated to Generic bench: cannot exceed total current staff in the municipality
                         generic_bench_vars = [var for key, var in decision_vars.items() 
                                              if key[0] in staff_vars and key[1] == court and key[2] == mun and key[3] == 'Bench_Generic']
                         if generic_bench_vars:
                             total_current_staff_in_mun = mun_data[staff_vars].sum().sum()
                             lp_model.addConstr(sum(generic_bench_vars) <= total_current_staff_in_mun, name=f"Max_Generic_Bench_{court}_{mun}")
                             if self.debug_callback:
                                self.log_debug(f"[DEBUG] Max Generic Bench {court} {mun}: Sum of {len(generic_bench_vars)} vars <= {total_current_staff_in_mun}")
                                generic_var_keys = [k for k in decision_vars.keys() if k[0] in staff_vars and k[1] == court and k[2] == mun and k[3] == 'Bench_Generic']
                                self.log_debug(f"[DEBUG]   Generic Bench Var Keys: {generic_var_keys}")
                                
                         # Constraints for staff allocated to specialized benches
                         for staff in staff_vars:
                              allocated_specialized_vars = []
                              current_specialized_staff_total = 0
                              for category, benches in bench_categories.items():
                                   for bench_name in benches:
                                        full_bench_name = f'Bench_{bench_name}'
                                        if full_bench_name in specialized_benches_in_mun:
                                             # Sum allocated staff for this type in this specific specialized bench
                                             allocated_specialized_vars.extend([var for key, var in decision_vars.items()
                                                                                 if key[0] == staff and key[1] == court and key[2] == mun and key[3] == full_bench_name])
                                             # Sum current staff for this type in this specific specialized bench
                                             current_specialized_staff_total += mun_data[staff][mun_data[full_bench_name] == 1].sum() # Should be only one row per mun/bench
                              
                              if allocated_specialized_vars:
                                  lp_model.addConstr(sum(allocated_specialized_vars) <= current_specialized_staff_total, name=f"Max_Specialized_Benches_{staff}_{court}_{mun}")
                                  if self.debug_callback:
                                       self.log_debug(f"[DEBUG] Max Specialized Benches {staff} in {court} {mun}: Sum of {len(allocated_specialized_vars)} vars <= {current_specialized_staff_total}")
                                       specialized_var_keys = [k for k in decision_vars.keys() if k[0] == staff and k[1] == court and k[2] == mun and k[3] in specialized_benches_in_mun]
                                       self.log_debug(f"[DEBUG]   Specialized Bench Var Keys: {specialized_var_keys}")

        elif granularity == "Court":
            if not selected_court:
                raise ValueError("Selected court is required for Court granularity")
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            
            # Total staff constraints for the court
            for staff in staff_vars:
                total_staff_vars = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff]
                total_current_staff = court_data[staff].sum()
                if self.debug_callback:
                    self.debug_callback(f"[DEBUG] Court: {staff} total vars: {len(total_staff_vars)}, max: {total_current_staff}")
                if total_staff_vars:
                    lp_model.addConstr(sum(total_staff_vars) <= total_current_staff, name=f"Max_Total_{staff}_{court_col}")

            if selected_court in ['Acores', 'Madeira']:
                # Island group constraints for Açores and Madeira
                for island, municipalities in island_groups[court_col].items():
                    if len(municipalities) == 1:
                        # Only one municipality in this island: staff must stay in that municipality
                        mun = municipalities[0]
                        mun_data = court_data[court_data[mun] == 1]
                        for staff in staff_vars:
                            # Filter variables for this staff type and the single municipality in this island
                            staff_vars_mun = [var for key, var in decision_vars.items() 
                                              if key[0] == staff and key[1] == mun]
                            current_staff = mun_data[staff].sum()
                            if self.debug_callback:
                                self.log_debug(f"[DEBUG] Court (Single Mun Island): {staff} in {court_col} {island} ({mun}) - Vars: {len(staff_vars_mun)}, Max: {current_staff}, Var Keys: {[k for k in decision_vars.keys() if k[0] == staff and k[1] == mun]}")
                            if staff_vars_mun:
                                lp_model.addConstr(sum(staff_vars_mun) <= current_staff, name=f"Max_{staff}_{court_col}_{island}_{mun}_single")
                    else:
                        # Multiple municipalities: allow movement within the island
                        island_staff = {}
                        for staff in staff_vars:
                            island_staff[staff] = sum(court_data[court_data[mun] == 1][staff].sum() for mun in municipalities)
                            if self.debug_callback:
                                self.log_debug(f"[DEBUG] Court (Multi Mun Island): {staff} in {court_col} {island} - Current Total: {island_staff[staff]}, Municipalities: {municipalities}")
                        for staff in staff_vars:
                            # Filter variables for this staff type and all municipalities in this island group
                            staff_vars_ = [var for key, var in decision_vars.items() 
                                           if key[0] == staff and key[1] in municipalities]
                            if self.debug_callback:
                                var_keys = [k for k in decision_vars.keys() if len(k) == 3 and k[0] == staff and k[1] in municipalities]
                                self.log_debug(f"[DEBUG] Court (Multi Mun Island): {staff} in {court_col} {island} - Vars: {len(staff_vars_)}, Max: {island_staff[staff]}, Var Keys: {var_keys}")
                            if staff_vars_:
                                lp_model.addConstr(sum(staff_vars_) <= island_staff[staff], name=f"Max_{staff}_{court_col}_{island}")

                # Minimum allocation constraint for Court granularity: If staff exist in the court, each variable for that staff type within the court must be >= 1
                self.log_debug(f"[DEBUG] Adding constraint: Minimum allocation if staff exist in {selected_court}")
                for staff in staff_vars:
                    total_current_staff_in_court = court_data[staff].sum()
                    if total_current_staff_in_court > 0:
                         staff_vars_court = [var for key, var in decision_vars.items() if len(key) == 3 and key[0] == staff]
                         if staff_vars_court:
                              # Ensure the sum of variables for this staff type is at least 1 if staff exist in the court
                              lp_model.addConstr(sum(staff_vars_court) >= 1, name=f"Min_Total_Allocation_{staff}_{selected_court}")
                              if self.debug_callback:
                                   self.log_debug(f"[DEBUG] Court Min Allocation {staff} in {selected_court}: Sum of {len(staff_vars_court)} vars >= 1 (Current total: {total_current_staff_in_court})")

            else:
                # Municipality-level constraints for regular courts
                for mun in mun_vars:
                    if court_data[mun].any():  # Only add constraint if municipality exists in this court
                        mun_data = court_data[court_data[mun] == 1]
                        for staff in staff_vars:
                            # Filter variables for this staff type and municipality
                            staff_vars_ = [var for key, var in decision_vars.items() 
                                           if key[0] == staff and key[1] == mun]
                            current_staff = mun_data[staff].sum()
                            if self.debug_callback:
                                var_keys = [k for k in decision_vars.keys() if len(k) == 3 and k[0] == staff and k[1] == mun]
                                self.log_debug(f"[DEBUG] Court: {staff} in {mun} vars: {len(staff_vars_)}, max: {current_staff}, Var Keys: {var_keys}")
                            if staff_vars_:
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
                    self.log_debug(f"[DEBUG] Municipality: {staff} vars: {len(staff_vars_)}, max: {current_staff}, Var Keys: {[k for k in decision_vars.keys() if len(k) == 2 and k[0] == staff]}")
                lp_model.addConstr(sum(staff_vars_) <= current_staff, name=f"Max_{staff}")

                # Minimum allocation constraint for Municipality granularity: If staff exist in the municipality, each variable for that staff type within the municipality must be >= 1
                self.log_debug(f"[DEBUG] Adding constraint: Minimum allocation if staff exist in {selected_municipality}")
                for staff in staff_vars:
                    total_current_staff_in_mun = mun_data[staff].sum()
                    if total_current_staff_in_mun > 0:
                        staff_vars_mun = [var for key, var in decision_vars.items() if len(key) == 2 and key[0] == staff]
                        if staff_vars_mun:
                            # Ensure the sum of variables for this staff type is at least 1 if staff exist in the municipality
                            lp_model.addConstr(sum(staff_vars_mun) >= 1, name=f"Min_Total_Allocation_{staff}_{selected_municipality}")
                            if self.debug_callback:
                                self.log_debug(f"[DEBUG] Municipality Min Allocation {staff} in {selected_municipality}: Sum of {len(staff_vars_mun)} vars >= 1 (Current total: {total_current_staff_in_mun})")

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

