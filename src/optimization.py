import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from gurobipy import GRB, Model, Var

from src.data_prep import data_prep

data, target_specific, target_all = data_prep('src/data.xlsx')

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.staff_vars = ['Judges', 'Justice Secretary', 'Law Clerck', 'Auxiliar Clerck',
                          'Administrative/Technical People', 'Operational/Auxiliar People']
        self.setup_ui()
        
        # Initialize model as None (we'll load it when needed)
        self.model = None
        self.status_var.set("Welcome - Ready to optimize")

    def setup_ui(self):
        self.root.title("Court Staff Optimizer")
        self.root.geometry("500x400")  # Increased height to accommodate disclaimer
        self.root.resizable(False, False)
        
        # Main frame for better organization
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Year input (moved to top)
        ttk.Label(main_frame, text="Year:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.year_var = tk.StringVar()
        self.year_dropdown = ttk.Combobox(main_frame, textvariable=self.year_var)
        self.year_dropdown['values'] = [str(year) for year in data['Year'].unique()]
        self.year_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.year_dropdown.current(0)  # Set default selection to the first year
        self.year_dropdown.bind('<<ComboboxSelected>>', self.on_year_change)

        # Granularity input
        ttk.Label(main_frame, text="Granularity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.granularity_var = tk.StringVar(value="Country")
        self.granularity_dropdown = ttk.Combobox(main_frame, textvariable=self.granularity_var)
        self.granularity_dropdown['values'] = ["Country", "Court", "Municipality"]
        self.granularity_dropdown.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.granularity_dropdown.current(0)  # Set default selection to "Country"
        self.granularity_dropdown.bind('<<ComboboxSelected>>', self.on_granularity_change)

        # Region selection frame (initially hidden)
        self.region_frame = ttk.Frame(main_frame)
        self.region_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        
        # Court selection (for both Court and Municipality granularity)
        self.court_label = ttk.Label(self.region_frame, text="Court:")
        self.court_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        self.court_var = tk.StringVar()
        self.court_dropdown = ttk.Combobox(self.region_frame, textvariable=self.court_var)
        self.court_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.court_dropdown.bind('<<ComboboxSelected>>', self.on_court_change)
        
        # Municipality selection (only for Municipality granularity)
        self.municipality_label = ttk.Label(self.region_frame, text="Municipality:")
        self.municipality_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.municipality_var = tk.StringVar()
        self.municipality_dropdown = ttk.Combobox(self.region_frame, textvariable=self.municipality_var)
        self.municipality_dropdown.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        self.region_frame.grid_remove()  # Hide initially
        
        # Prediction Model selection
        ttk.Label(main_frame, text="Prediction Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="Ridge")
        self.model_dropdown = ttk.Combobox(main_frame, textvariable=self.model_var)
        self.model_dropdown['values'] = ["Ridge", "Linear Regression"]
        self.model_dropdown.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.model_dropdown.current(0)  # Set default selection to Ridge
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Run button
        self.run_button = ttk.Button(main_frame, text="Run Optimization", command=self.on_run)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=15)
        
        # Disclaimer label
        disclaimer_text = "Disclaimer: The optimization targets the machine learning model prediction, not the real value. The real value is only used for comparison. It might not be possible to reach the real value."
        disclaimer_label = ttk.Label(main_frame, text=disclaimer_text, wraplength=480, justify=tk.CENTER)
        disclaimer_label.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.region_frame.columnconfigure(1, weight=1)
        
        # Debug console (hidden by default)
        self.debug_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.debug_text.grid(row=7, column=0, columnspan=2, sticky=tk.EW)
        self.show_debug(False)  # Hide by default
        
        # Add debug toggle button (for troubleshooting)
        ttk.Button(main_frame, text="Debug", command=lambda: self.show_debug()).grid(row=8, column=0, columnspan=2)

    def show_debug(self, show=None):
        """Toggle debug console visibility"""
        if show is None:
            current_state = self.debug_text.winfo_ismapped()
            self.debug_text.grid_remove() if current_state else self.debug_text.grid()
        else:
            self.debug_text.grid() if show else self.debug_text.grid_remove()

    def log_debug(self, message):
        """Add a message to the debug console"""
        self.debug_text.config(state=tk.NORMAL)
        self.debug_text.insert(tk.END, message + "\n")
        self.debug_text.config(state=tk.DISABLED)
        self.debug_text.see(tk.END)

    def on_model_change(self, event=None):
        """Handle model selection change"""
        self.model = None  # Reset model to force reload
        self.load_model()  # Load the newly selected model

    def load_model(self):
        """Load the machine learning model with error handling"""
        try:
            model_name = self.model_var.get().lower().replace(" ", "_")
            model_path = f'models/{model_name}.joblib'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
            
            self.log_debug(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            self.log_debug(f"Model loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return False

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

    def on_run(self):
        """Handle the optimization request"""
        try:
            self.log_debug("\n=== Starting Optimization ===")
            
            # Validate inputs
            try:
                year = int(self.year_var.get())
                if year <= 0:
                    raise ValueError("Year must be positive")
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid positive year number")
                return
            
            granularity = self.granularity_var.get()
            selected_court = self.court_var.get() if granularity in ["Court", "Municipality"] else None
            selected_municipality = self.municipality_var.get() if granularity == "Municipality" else None
            
            # Load model if not already loaded
            if self.model is None and not self.load_model():
                return
            
            # Disable UI during processing
            self.run_button.config(state=tk.DISABLED)
            self.status_var.set("Optimizing...")
            self.root.update()
            
            # Get data for selected year
            year_data = data[data['Year'] == year]
            year_data = year_data.drop(columns=['Year'])
            #year_data = year_data.drop(columns=target_specific + [target_all])
            
            self.log_debug("Disclaimer: The optimization targets the machine learning model prediction, not the real value. The real value is only used for comparison. It might not be possible to reach the real value.")
            self.log_debug(f"Running optimization for year {year}, granularity {granularity}")
            if selected_court:
                self.log_debug(f"Selected court: {selected_court}")
            if selected_municipality:
                self.log_debug(f"Selected municipality: {selected_municipality}")

            if granularity == "Country":
                lp_model = Model(f"Staff_Optimization_{year}")
            elif granularity == "Court":
                lp_model = Model(f"Staff_Optimization_{year}_{selected_court}")
            elif granularity == "Municipality":
                lp_model = Model(f"Staff_Optimization_{year}_{selected_court}_{selected_municipality}")
            else:
                raise ValueError("Invalid granularity selected")
            
            decision_variables = self.create_decision_variables(lp_model, year_data.drop(columns=target_specific + [target_all]), granularity, selected_court, selected_municipality)
            objective = self.objective_function(self.model, year_data.drop(columns=target_specific + [target_all]), decision_variables, granularity, selected_court, selected_municipality)
            lp_model.setObjective(sum(objective), GRB.MAXIMIZE)
            self.add_constraints(lp_model, year_data.drop(columns=target_specific + [target_all]), decision_variables, granularity, selected_court, selected_municipality)
            
            # Run optimization (simplified for debugging)
            result, status, optimal_allocation = self.simulate_optimization(lp_model)
            ml_prediction = self.ml_model_prediction(year_data.drop(columns=target_specific + [target_all]), decision_variables, granularity, selected_court, selected_municipality)
            real_value = self.real_value(year_data, granularity, selected_court, selected_municipality)
            
            if result is None:
                messagebox.showerror("Optimization failed", f"Model status: {status}")
                raise ValueError("Optimization failed")
            else:   
                messagebox.showinfo("Result", f"Optimization complete!\nObjective value: {result:.2f}")
                self.log_debug('Optimization complete!')
                self.log_debug(f"Optimized objective value: {result}")
                self.log_debug(f"ML model prediction (with default allocations): {ml_prediction}")
                self.log_debug(f"Real value: {real_value}")
            
        except Exception as e:
            self.log_debug(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")

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

        judge_var_keys = [k for k in decision_vars.keys() if k[0] == 'Judges']

        for key, var in decision_vars.items():
                lp_model.addConstr(var >= 0, name="non_negativity")

                if key in judge_var_keys:
                    lp_model.addConstr(var >= 1, name="Min_Judges")

        if granularity == "Country":
            for court in court_vars:
                for staff in staff_vars:
                    staff_vars_ = [var for key, var in decision_vars.items() if court == key[1] and staff == key[0]]
                    if staff_vars_:
                        lp_model.addConstr(sum(staff_vars_) <= staff_max[(court, staff)], name=f"Max_{staff}_{court}")
                    else:
                        continue
        
        elif granularity == "Court":
            for mun in mun_vars:
                for staff in staff_vars:
                    staff_vars_ = [var for key, var in decision_vars.items() if staff == key[0] and mun == key[1]]
                    if staff_vars_:
                        lp_model.addConstr(sum(staff_vars_) <= staff_max[(mun, staff)], name=f"Max_{staff}_{selected_municipality}")
                    else:
                        continue
        
        elif granularity == "Municipality":
            for staff in staff_vars:
                staff_vars_ = [var for key, var in decision_vars.items() if staff == key[0]]
                if staff_vars_:
                    lp_model.addConstr(sum(staff_vars_) <= staff_max[staff], name=f"Max_{staff}")
                else:
                    continue

    def simulate_optimization(self, lp_model: Model):
        """Simulate optimization for debugging"""
        self.log_debug("Simulating optimization...")

        # Enable Gurobi console output
        lp_model.setParam('OutputFlag', 1)
        
        # Create a string buffer to capture Gurobi's output
        import io
        import sys
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            lp_model.optimize()
            
            # Restore stdout and get the captured output
            sys.stdout = old_stdout
            gurobi_output = captured_output.getvalue()
            
            # Log the Gurobi output
            self.log_debug("\nGurobi Output:")
            self.log_debug(gurobi_output)

            if lp_model.status == GRB.OPTIMAL:
                return lp_model.objVal, lp_model.status, lp_model.getVars()
            elif lp_model.status == GRB.INFEASIBLE:
                self.log_debug("Model is infeasible!")
                return None, lp_model.status, None
            elif lp_model.status == GRB.UNBOUNDED:
                self.log_debug("Model is unbounded!")
                return None, lp_model.status, None
            else:
                self.log_debug(f"Model status: {lp_model.status}")
                return None, lp_model.status, None
        finally:
            # Ensure stdout is restored even if an error occurs
            sys.stdout = old_stdout

    def ml_model_prediction(self, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        """Predict the target variable using the machine learning model with the default allocations"""

        staff_vars = self.staff_vars
        court_vars = [col for col in year_data.columns if col.startswith('Court_')]
        mun_vars = [col for col in year_data.columns if col.startswith('Municipality_')]

        if granularity == "Country":
            ml_prediction = self.model.predict(year_data).sum().sum()

        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            ml_prediction = self.model.predict(court_data).sum().sum()

        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            ml_prediction = self.model.predict(mun_data).sum().sum()

        return ml_prediction
    
    def real_value(self, year_data, granularity, selected_court=None, selected_municipality=None):
        """Calculate the real value of the target variable"""
        if granularity == "Country":
            real_value = year_data[target_specific].sum().sum()

        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            real_value = court_data[target_specific].sum().sum()

        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            real_value = mun_data[target_specific].sum().sum()

        return real_value

        
    def update_dropdowns(self):
        """Update dropdown options based on selected year"""
        selected_year = int(self.year_var.get())
        year_data = data[data['Year'] == selected_year]
        
        # Get court columns and extract court names
        court_columns = [col for col in year_data.columns if col.startswith('Court_')]
        courts = sorted([col.replace('Court_', '') for col in court_columns])
        self.court_dropdown['values'] = courts
        if courts:
            self.court_dropdown.current(0)
            self.court_var.set(courts[0])
        
        # Update municipality dropdown if needed
        if self.granularity_var.get() == "Municipality":
            self.update_municipality_dropdown()

    def update_municipality_dropdown(self):
        """Update municipality dropdown based on selected court and year"""
        selected_year = int(self.year_var.get())
        selected_court = self.court_var.get()
        
        if selected_court:
            year_data = data[data['Year'] == selected_year]
            # Filter for rows where the selected court has value 1
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            
            # Get municipality columns that have value 1 in the filtered data
            mun_columns = [col for col in court_data.columns if col.startswith('Municipality_')]
            valid_municipalities = []
            for mun_col in mun_columns:
                # Check if any row has value 1 for this municipality in the selected year
                if court_data[mun_col].any():
                    valid_municipalities.append(mun_col.replace('Municipality_', ''))
            
            municipalities = sorted(valid_municipalities)
            self.municipality_dropdown['values'] = municipalities
            if municipalities:
                self.municipality_dropdown.current(0)
                self.municipality_var.set(municipalities[0])
            else:
                self.municipality_var.set("")  # Clear if no valid municipalities

    def on_granularity_change(self, event=None):
        """Handle granularity selection change"""
        selected = self.granularity_var.get()
        
        if selected == "Country":
            self.region_frame.grid_remove()
            self.court_var.set("")
            self.municipality_var.set("")
        else:
            self.region_frame.grid()
            self.update_dropdowns()
            
            if selected == "Court":
                self.municipality_label.grid_remove()
                self.municipality_dropdown.grid_remove()
            else:  # Municipality
                self.municipality_label.grid()
                self.municipality_dropdown.grid()
                self.update_municipality_dropdown()

    def on_year_change(self, event=None):
        """Handle year selection change"""
        self.update_dropdowns()

    def on_court_change(self, event=None):
        """Handle court selection change"""
        if self.granularity_var.get() == "Municipality":
            self.update_municipality_dropdown()

