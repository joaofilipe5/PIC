import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from gurobipy import GRB, Model, Var

from src.data_prep import data_prep
from src.optimization import Optimization as OrigOptimization

data, target_specific = data_prep('src/data.xlsx')

class Optimization(OrigOptimization):
    def create_decision_variables(self, *args, **kwargs):
        x_vars = super().create_decision_variables(*args, **kwargs)
        if hasattr(self, 'log_debug') and callable(self.log_debug):
            self.log_debug(f"[DEBUG] Decision variable keys: {list(x_vars.keys())}")
        return x_vars

    def add_constraints(self, lp_model, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        if hasattr(self, 'log_debug') and callable(self.log_debug):
            self.log_debug(f"[DEBUG] Adding constraints for granularity: {granularity}")
            self.log_debug(f"[DEBUG] Staff max: {self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)}")
        return super().add_constraints(lp_model, year_data, decision_vars, granularity, selected_court, selected_municipality)

class AllocationMatrix(ttk.Frame):
    def __init__(self, parent, staff_vars, benches, allocations, current_court=None, current_mun=None):
        super().__init__(parent)
        self.staff_vars = staff_vars
        self.benches = benches
        self.allocations = allocations
        self.current_court = current_court
        self.current_mun = current_mun
        self.create_matrix()
        self.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def create_matrix(self):
        matrix_frame = ttk.Frame(self, padding="10")
        matrix_frame.grid(row=0, column=0, sticky='nsew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        for i in range(len(self.benches) + 1):
            matrix_frame.grid_rowconfigure(i, weight=1)
        for j in range(len(self.staff_vars) + 1):
            matrix_frame.grid_columnconfigure(j, weight=1)
            
        for j, staff in enumerate(self.staff_vars):
            label = ttk.Label(matrix_frame, text=staff, font=('Arial', 10, 'bold'), anchor='center')
            label.grid(row=0, column=j+1, padx=5, pady=5, sticky='nsew')
            
        for i, bench in enumerate(self.benches):
            label = ttk.Label(matrix_frame, text=bench.replace('Bench_', ''), font=('Arial', 10, 'bold'), anchor='center')
            label.grid(row=i+1, column=0, padx=5, pady=5, sticky='nsew')
            
            for j, staff in enumerate(self.staff_vars):
                value = 0
                for k, v in self.allocations.items():
                    if isinstance(k, tuple):
                        if len(k) == 4 and k[0] == staff and k[1] == self.current_court and k[2] == self.current_mun and k[3] == bench:
                            value = int(v.x) if hasattr(v, 'x') else int(v)
                            break
                        elif len(k) == 3 and k[0] == staff and k[1] == self.current_mun and k[2] == bench:
                            value = int(v.x) if hasattr(v, 'x') else int(v)
                            break
                        elif len(k) == 2 and k[0] == staff and k[1] == bench:
                            value = int(v.x) if hasattr(v, 'x') else int(v)
                            break
                label = ttk.Label(matrix_frame, text=str(value), anchor='center')
                label.grid(row=i+1, column=j+1, padx=5, pady=2, sticky='nsew')

class App:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
        # Initialize model as None (we'll load it when needed)
        self.model = None
        self.status_var.set("Welcome - Ready to optimize")
        self.optimization = OrigOptimization(debug_callback=self.log_debug)

    def setup_ui(self):
        self.root.title("Court Staff Optimizer")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        # Main frame for better organization (centered with spacers)
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.grid(row=0, column=1, sticky="nsew")
        scrollable_frame.grid_rowconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=0)
        scrollable_frame.grid_columnconfigure(2, weight=1)

        # Add left and right spacers to center content
        left_spacer = ttk.Frame(scrollable_frame)
        left_spacer.grid(row=0, column=0, sticky="nsew")
        right_spacer = ttk.Frame(scrollable_frame)
        right_spacer.grid(row=0, column=2, sticky="nsew")
        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(2, weight=1)

        # Control Panel Frame
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", pady=5)
        main_frame.grid_columnconfigure(0, weight=1)

        # Center the control panel contents
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_rowconfigure(4, weight=1)

        # Year input
        ttk.Label(control_frame, text="Year:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.year_var = tk.StringVar()
        self.year_dropdown = ttk.Combobox(control_frame, textvariable=self.year_var)
        self.year_dropdown['values'] = [str(year) for year in data['Year'].unique()]
        self.year_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.year_dropdown.current(0)
        self.year_dropdown.bind('<<ComboboxSelected>>', self.on_year_change)

        # Granularity input
        ttk.Label(control_frame, text="Granularity:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.granularity_var = tk.StringVar(value="Country")
        self.granularity_dropdown = ttk.Combobox(control_frame, textvariable=self.granularity_var)
        self.granularity_dropdown['values'] = ["Country", "Court", "Municipality"]
        self.granularity_dropdown.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        self.granularity_dropdown.current(0)
        self.granularity_dropdown.bind('<<ComboboxSelected>>', self.on_granularity_change)

        # Region selection frame
        self.region_frame = ttk.Frame(control_frame)
        self.region_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        self.region_frame.columnconfigure(1, weight=1)
        
        # Court selection
        self.court_label = ttk.Label(self.region_frame, text="Court:")
        self.court_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        self.court_var = tk.StringVar()
        self.court_dropdown = ttk.Combobox(self.region_frame, textvariable=self.court_var)
        self.court_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.court_dropdown.bind('<<ComboboxSelected>>', self.on_court_change)
        
        # Municipality selection
        self.municipality_label = ttk.Label(self.region_frame, text="Municipality:")
        self.municipality_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.municipality_var = tk.StringVar()
        self.municipality_dropdown = ttk.Combobox(self.region_frame, textvariable=self.municipality_var)
        self.municipality_dropdown.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        self.region_frame.grid_remove()
        
        # Prediction Model selection
        ttk.Label(control_frame, text="Prediction Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="Ridge")
        self.model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_var)
        self.model_dropdown['values'] = ["Ridge", "Linear Regression"]
        self.model_dropdown.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        self.model_dropdown.current(0)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="Run Optimization", command=self.on_run)
        self.run_button.grid(row=4, column=0, columnspan=2, pady=15, sticky='nsew')

        # Results Frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Optimization Results", padding="10")
        self.results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        main_frame.grid_rowconfigure(1, weight=1)

        # Create notebook for results
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.grid(row=0, column=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, sticky="ew", pady=5)
        main_frame.grid_rowconfigure(2, weight=0)

        # Debug console (hidden by default)
        self.debug_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.debug_text.grid(row=3, column=0, sticky="ew", pady=5)
        main_frame.grid_rowconfigure(3, weight=0)
        self.show_debug(False)
        
        # Add debug toggle button
        ttk.Button(main_frame, text="Debug", command=lambda: self.show_debug()).grid(row=4, column=0, pady=5)
        main_frame.grid_rowconfigure(4, weight=0)

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

    def display_allocations(self, allocations, year_data, granularity, selected_court=None, selected_municipality=None):
        for tab in self.results_notebook.tabs():
            self.results_notebook.forget(tab)
            
        if granularity == "Country":
            court_vars = [col for col in year_data.columns if col.startswith('Court_')]
            for court in court_vars:
                court_name = court.replace('Court_', '')
                court_frame = ttk.Frame(self.results_notebook)
                court_frame.grid(row=0, column=0, sticky='nsew')
                self.results_notebook.add(court_frame, text=court_name)
                
                mun_notebook = ttk.Notebook(court_frame)
                mun_notebook.grid(row=0, column=0, sticky='nsew')
                court_frame.grid_rowconfigure(0, weight=1)
                court_frame.grid_columnconfigure(0, weight=1)
                
                court_data = year_data[year_data[court] == 1]
                mun_vars = [col for col in court_data.columns if col.startswith('Municipality_')]
                for mun in mun_vars:
                    if court_data[mun].any():
                        mun_name = mun.replace('Municipality_', '')
                        mun_frame = ttk.Frame(mun_notebook)
                        mun_frame.grid(row=0, column=0, sticky='nsew')
                        mun_notebook.add(mun_frame, text=mun_name)
                        mun_notebook.grid_rowconfigure(0, weight=1)
                        mun_notebook.grid_columnconfigure(0, weight=1)
                        
                        mun_data = court_data[court_data[mun] == 1]
                        bench_vars = [col for col in mun_data.columns if col.startswith('Bench_')]
                        benches = [bench for bench in bench_vars if mun_data[bench].any()]
                        matrix = AllocationMatrix(mun_frame, self.optimization.staff_vars, benches, allocations, court, mun)
                        matrix.grid(row=0, column=0, sticky='nsew')
                        mun_frame.grid_rowconfigure(0, weight=1)
                        mun_frame.grid_columnconfigure(0, weight=1)
                        
        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            mun_vars = [col for col in court_data.columns if col.startswith('Municipality_')]
            for mun in mun_vars:
                if court_data[mun].any():
                    mun_name = mun.replace('Municipality_', '')
                    mun_frame = ttk.Frame(self.results_notebook)
                    mun_frame.grid(row=0, column=0, sticky='nsew')
                    self.results_notebook.add(mun_frame, text=mun_name)
                    self.results_notebook.grid_rowconfigure(0, weight=1)
                    self.results_notebook.grid_columnconfigure(0, weight=1)
                    
                    mun_data = court_data[court_data[mun] == 1]
                    bench_vars = [col for col in mun_data.columns if col.startswith('Bench_')]
                    benches = [bench for bench in bench_vars if mun_data[bench].any()]
                    matrix = AllocationMatrix(mun_frame, self.optimization.staff_vars, benches, allocations, None, mun)
                    matrix.grid(row=0, column=0, sticky='nsew')
                    mun_frame.grid_rowconfigure(0, weight=1)
                    mun_frame.grid_columnconfigure(0, weight=1)
                    
        else:  # Municipality
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[year_data[mun_col] == 1]
            bench_vars = [col for col in mun_data.columns if col.startswith('Bench_')]
            benches = [bench for bench in bench_vars if mun_data[bench].any()]
            mun_frame = ttk.Frame(self.results_notebook)
            mun_frame.grid(row=0, column=0, sticky='nsew')
            self.results_notebook.add(mun_frame, text=selected_municipality or 'Municipality')
            self.results_notebook.grid_rowconfigure(0, weight=1)
            self.results_notebook.grid_columnconfigure(0, weight=1)
            
            matrix = AllocationMatrix(mun_frame, self.optimization.staff_vars, benches, allocations)
            matrix.grid(row=0, column=0, sticky='nsew')
            mun_frame.grid_rowconfigure(0, weight=1)
            mun_frame.grid_columnconfigure(0, weight=1)

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
            
            try:
                decision_variables = self.optimization.create_decision_variables(lp_model, year_data.drop(columns=target_specific), granularity, selected_court, selected_municipality)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create decision variables:\n{str(e)}")
        
            try:
                objective = self.optimization.objective_function(self.model, year_data.drop(columns=target_specific), decision_variables, granularity, selected_court, selected_municipality)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create objective function:\n{str(e)}")

            try:
                lp_model.setObjective(sum(objective), GRB.MAXIMIZE)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set objective function:\n{str(e)}")

            try:
                self.optimization.add_constraints(lp_model, year_data.drop(columns=target_specific), decision_variables, granularity, selected_court, selected_municipality)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add constraints:\n{str(e)}")
                        
            # Run optimization
            result, status, optimal_allocation = self.simulate_optimization(lp_model)
            # Debug: print all decision variable values
            debug_vars = []
            for k, v in decision_variables.items():
                val = v.x if hasattr(v, 'x') else v
                debug_vars.append(f"{k}: {val}")
            self.log_debug("Decision variable values after optimization:\n" + "\n".join(debug_vars))
            ml_prediction = self.ml_model_prediction(year_data.drop(columns=target_specific), decision_variables, granularity, selected_court, selected_municipality)
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
                
                # Display the allocations
                self.display_allocations(decision_variables, year_data, granularity, selected_court, selected_municipality)
            
        except Exception as e:
            self.log_debug(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")

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

        staff_vars = self.optimization.staff_vars
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

