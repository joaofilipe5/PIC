import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from gurobipy import GRB, Model, Var
from typing import List

from src.data_prep import data_prep
from src.optimization import Optimization as OrigOptimization
from src.manual_input import ManualInputNotebook

data, target_specific = data_prep('src/data.xlsx')

# Identify columns for previous year pending and completed cases
# Assuming these columns follow the patterns 'PC_*_prev_year' and 'CC_*_prev_year'
all_data_cols = data.columns.tolist()
prev_year_case_cols = [col for col in all_data_cols if col.endswith('_prev_year') and (col.startswith('PC_') or col.startswith('CC_'))]

class Optimization(OrigOptimization):
    def create_decision_variables(self, *args, **kwargs):
        x_vars = super().create_decision_variables(*args, **kwargs)
        if hasattr(self, 'log_debug') and callable(self.log_debug):
            self.log_debug(f"[DEBUG] Decision variable keys: {list(x_vars.keys())}")
        return x_vars

    def add_constraints(self, lp_model, year_data, decision_vars, granularity, selected_court=None, selected_municipality=None):
        if hasattr(self, 'log_debug') and callable(self.log_debug):
            self.log_debug(f"[DEBUG] Adding constraints for granularity: {granularity}")
            # Staff max calculation is likely not needed here for manual input, or needs adjustment
            # self.log_debug(f"[DEBUG] Staff max: {self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)}")
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
        
        # Initialize manual input data and widgets
        self.manual_input_data = None
        self.manual_input_notebook = None
        self.manual_frame = None # Keep track of the manual input frame
        self.bench_display_frame = None
        self.bench_display_label = None

    def setup_ui(self):
        self.root.title("Court Staff Optimizer")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Use a PanedWindow to separate main content and debug window
        paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Top frame for controls and results
        main_container = ttk.Frame(paned)
        paned.add(main_container, stretch='always')

        # Bottom frame for debug
        debug_container = ttk.Frame(paned)
        paned.add(debug_container, stretch='never', minsize=120)

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
        self.main_frame = main_frame  # Store reference to main frame
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
        self.year_dropdown['values'] = [str(year) for year in data['Year'].unique()] + ["New Year"]
        self.year_dropdown.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        self.year_dropdown.current(0) # Set default to first year in data
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
        self.municipality_dropdown.bind('<<ComboboxSelected>>', self.on_municipality_change) # Bind municipality change
        
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

        # Bench Display Frame (for Municipality New Year mode)
        self.bench_display_frame = ttk.LabelFrame(main_frame, text="Benches in Municipality", padding="10")
        self.bench_display_label = ttk.Label(self.bench_display_frame, text="")
        self.bench_display_label.grid(row=0, column=0, sticky=tk.W)
        self.bench_display_frame.grid_remove() # Initially hidden

        # Results Frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Optimization Results", padding="10")
        self.results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        # main_frame.grid_rowconfigure(1, weight=1) # Will be adjusted by show/hide manual input

        # Create notebook for results
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.grid(row=0, column=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, sticky="ew", pady=5) # Moved to row 3 to accommodate manual input and benches
        main_frame.grid_rowconfigure(3, weight=0)

        # Debug console (always visible, resizable)
        self.debug_text = tk.Text(debug_container, height=10, width=100, state=tk.DISABLED)
        self.debug_text.grid(row=0, column=0, sticky='nsew')
        self.show_debug(True)
        # Add debug clear button
        clear_btn = ttk.Button(debug_container, text="Clear Debug", command=lambda: self.debug_text.config(state=tk.NORMAL) or self.debug_text.delete('1.0', tk.END) or self.debug_text.config(state=tk.DISABLED))
        clear_btn.grid(row=1, column=0, sticky='se', padx=5, pady=2)
        
        # Add debug export button
        export_btn = ttk.Button(debug_container, text="Export Debug", command=self.export_debug_log)
        export_btn.grid(row=1, column=0, sticky='sw', padx=5, pady=2)
        
        debug_container.grid_rowconfigure(0, weight=1)
        debug_container.grid_columnconfigure(0, weight=1)

        # Initial dropdown update
        self.update_dropdowns()

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

    def export_debug_log(self):
        """Export the debug console content to a text file"""
        try:
            debug_content = self.debug_text.get('1.0', tk.END).strip()
            if not debug_content:
                messagebox.showinfo("Export Debug", "Debug console is empty.")
                return

            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Debug Log As"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(debug_content)
                messagebox.showinfo("Export Debug", f"Debug log successfully exported to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export debug log:\n{str(e)}")

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
        self.log_debug("[DEBUG] display_allocations called.")
        self.log_debug(f"[DEBUG] Results frame mapped: {self.results_frame.winfo_ismapped()}")

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
            # For manual input case, use the selected court and municipality
            if self.year_var.get() == "New Year":
                court_col = f'Court_{selected_court}'
                mun_col = f'Municipality_{selected_municipality}'
                # Use the year_data DataFrame created in on_run (which has the correct structure for manual input)
                mun_data_for_benches = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]
                bench_vars = [col for col in mun_data_for_benches.columns if col.startswith('Bench_')]
                benches = [bench for bench in bench_vars if mun_data_for_benches[bench].any()]
                current_court = court_col
                current_mun = mun_col
                self.log_debug(f"[DEBUG] Display Allocations - New Year Municipality: Found {len(benches)} benches: {benches}")

            else:
                # For historical years, filter from the full year_data based on selected municipality
                mun_col = f'Municipality_{selected_municipality}'
                mun_data_filtered = year_data[year_data[mun_col] == 1]
                bench_vars = [col for col in mun_data_filtered.columns if col.startswith('Bench_')]
                benches = [bench for bench in bench_vars if mun_data_filtered[bench].any()]
                # Determine the court for historical data
                court_cols = [col for col in year_data.columns if col.startswith('Court_')]
                current_court = None
                for col in court_cols:
                     if mun_data_filtered[col].any():
                          current_court = col
                          break
                current_mun = mun_col
                self.log_debug(f"[DEBUG] Display Allocations - Historical Municipality: Found {len(benches)} benches: {benches}")

            mun_frame = ttk.Frame(self.results_notebook)
            mun_frame.grid(row=0, column=0, sticky='nsew')
            self.results_notebook.add(mun_frame, text=selected_municipality or 'Municipality')
            self.results_notebook.grid_rowconfigure(0, weight=1)
            self.results_notebook.grid_columnconfigure(0, weight=1)

            # Pass current_court and current_mun to AllocationMatrix
            matrix = AllocationMatrix(mun_frame, self.optimization.staff_vars, benches, allocations, current_court, current_mun)
            matrix.grid(row=0, column=0, sticky='nsew')
            mun_frame.grid_rowconfigure(0, weight=1)
            mun_frame.grid_columnconfigure(0, weight=1)
        
        self.log_debug(f"[DEBUG] Number of result tabs created: {len(self.results_notebook.tabs())}")

    def on_run(self):
        """Handle the optimization request"""
        try:
            self.log_debug("\n=== Starting Optimization ===")

            # Check if we're using manual input
            selected_year_option = self.year_var.get()
            if selected_year_option == "New Year":
                if not self.manual_input_data:
                    messagebox.showerror("Input Error", "Please enter staff values for all regions")
                    return

                # Create a DataFrame with the manual input data
                year_data = self.create_manual_input_dataframe()
                # Use a placeholder for the year in logging and model naming
                display_year = "New Year"

            else:
                # Validate inputs
                try:
                    year = int(selected_year_option)
                    if year <= 0:
                        raise ValueError("Year must be positive")
                except ValueError as e:
                    self.log_debug(f"Input validation error: {str(e)}")
                    messagebox.showerror("Input Error", "Please enter a valid positive year number")
                    return

                # Get data for selected year
                try:
                    year_data = data[data['Year'] == year].drop(columns=['Year'])
                    if year_data.empty:
                        raise ValueError(f"No data found for year {year}")
                    display_year = str(year)

                except Exception as e:
                    self.log_debug(f"Data preparation error: {str(e)}")
                    messagebox.showerror("Data Error", f"Failed to prepare data for year {year}:\n{str(e)}")
                    return

            granularity = self.granularity_var.get()
            selected_court = self.court_var.get() if granularity in ["Court", "Municipality"] else None
            selected_municipality = self.municipality_var.get() if granularity == "Municipality" else None

            # Load model if not already loaded
            if self.model is None:
                if not self.load_model():
                    self.log_debug("Failed to load model")
                    return

            # Disable UI during processing
            self.run_button.config(state=tk.DISABLED)
            self.status_var.set("Optimizing...")
            self.root.update()

            self.log_debug("Disclaimer: The optimization targets the machine learning model prediction, not the real value. The real value is only used for comparison. It might not be possible to reach the real value.")
            self.log_debug(f"Running optimization for year {display_year}, granularity {granularity}")
            if selected_court:
                self.log_debug(f"Selected court: {selected_court}")
            if selected_municipality:
                self.log_debug(f"Selected municipality: {selected_municipality}")

            try:
                # Use display_year for model name
                if granularity == "Country":
                    lp_model = Model(f"Staff_Optimization_{display_year}")
                elif granularity == "Court":
                    lp_model = Model(f"Staff_Optimization_{display_year}_{selected_court}")
                elif granularity == "Municipality":
                    lp_model = Model(f"Staff_Optimization_{display_year}_{selected_court}_{selected_municipality}")
                else:
                    raise ValueError("Invalid granularity selected")
            except Exception as e:
                self.log_debug(f"Model creation error: {str(e)}")
                messagebox.showerror("Model Error", f"Failed to create optimization model:\n{str(e)}")
                return

            try:
                self.log_debug("Creating decision variables...")
                # Pass the relevant columns for decision variables creation
                cols_for_vars = [col for col in year_data.columns if not col in target_specific]
                decision_variables = self.optimization.create_decision_variables(lp_model, year_data[cols_for_vars], granularity, selected_court, selected_municipality)
                self.log_debug(f"Created {len(decision_variables)} decision variables")
            except Exception as e:
                self.log_debug(f"Decision variable creation error: {str(e)}")
                messagebox.showerror("Variable Error", f"Failed to create decision variables:\n{str(e)}")
                return

            try:
                self.log_debug("Creating objective function...")
                # Pass the relevant columns for objective function creation
                cols_for_objective = [col for col in year_data.columns if not col in target_specific]
                objective = self.optimization.objective_function(self.model, year_data[cols_for_objective], decision_variables, granularity, selected_court, selected_municipality)
                self.log_debug(f"Created objective function with {len(objective)} terms")
            except Exception as e:
                self.log_debug(f"Objective function error: {str(e)}")
                messagebox.showerror("Objective Error", f"Failed to create objective function:\n{str(e)}")
                return

            try:
                self.log_debug("Setting objective function...")
                lp_model.setObjective(sum(objective), GRB.MAXIMIZE)
            except Exception as e:
                self.log_debug(f"Objective setting error: {str(e)}")
                messagebox.showerror("Objective Error", f"Failed to set objective function:\n{str(e)}")
                return

            try:
                self.log_debug("Adding constraints...")
                # Pass the relevant columns for constraint creation
                cols_for_constraints = [col for col in year_data.columns if not col in target_specific]
                self.optimization.add_constraints(lp_model, year_data[cols_for_constraints], decision_variables, granularity, selected_court, selected_municipality)
                self.log_debug("Constraints added successfully")
            except Exception as e:
                self.log_debug(f"Constraint error: {str(e)}")
                messagebox.showerror("Constraint Error", f"Failed to add constraints:\n{str(e)}")
                return

            # Run optimization
            self.log_debug("Starting optimization...")
            result, status, optimal_allocation = self.simulate_optimization(lp_model)

            if result is None:
                error_msg = "Optimization failed"
                if status == GRB.INFEASIBLE:
                    error_msg = "Model is infeasible - no solution exists that satisfies all constraints"
                elif status == GRB.UNBOUNDED:
                    error_msg = "Model is unbounded - objective can grow infinitely"
                elif status == GRB.INF_OR_UNBD:
                    error_msg = "Model is either infeasible or unbounded"
                elif status == GRB.CUTOFF:
                    error_msg = "Model objective is worse than specified cutoff"
                elif status == GRB.ITERATION_LIMIT:
                    error_msg = "Optimization stopped due to iteration limit"
                elif status == GRB.NODE_LIMIT:
                    error_msg = "Optimization stopped due to node limit"
                elif status == GRB.TIME_LIMIT:
                    error_msg = "Optimization stopped due to time limit"
                elif status == GRB.SOLUTION_LIMIT:
                    error_msg = "Optimization stopped due to solution limit"
                elif status == GRB.INTERRUPTED:
                    error_msg = "Optimization was interrupted"
                elif status == GRB.NUMERIC:
                    error_msg = "Optimization stopped due to numerical issues"
                elif status == GRB.SUBOPTIMAL:
                    error_msg = "Optimization stopped with a suboptimal solution"
                elif status == GRB.INPROGRESS:
                    error_msg = "Optimization is still in progress"
                elif status == GRB.USER_OBJ_LIMIT:
                    error_msg = "Optimization stopped due to user objective limit"

                self.log_debug(f"Optimization failed with status {status}: {error_msg}")
                messagebox.showerror("Optimization Failed", error_msg)
                return
            else:
                # Debug: print all decision variable values only on success
                debug_vars = []
                for k, v in decision_variables.items():
                    val = v.x if hasattr(v, 'x') else None
                    debug_vars.append(f"{k}: {val}")
                self.log_debug("Decision variable values after optimization:\n" + "\n".join(debug_vars))

                messagebox.showinfo("Result", f"Optimization complete!\nObjective value: {result:.2f}")
                self.log_debug('Optimization complete!')
                self.log_debug(f"Optimized objective value: {result}")

                # Only show ML prediction and real value for historical years
                if selected_year_option != "New Year":
                    # Pass the full year_data (including targets for real_value calculation)
                    ml_prediction = self.ml_model_prediction(year_data, granularity, selected_court, selected_municipality)
                    real_value = self.real_value(year_data, granularity, selected_court, selected_municipality)

                    self.log_debug(f"ML model prediction (with default allocations): {ml_prediction}")
                    self.log_debug(f"Real value: {real_value}")

                # Hide manual input and display the allocations
                if selected_year_option == "New Year":
                    self.hide_manual_input()

                # Display the allocations
                # Pass the full year_data here as display_allocations needs it for structure
                self.display_allocations(decision_variables, year_data, granularity, selected_court, selected_municipality)

        except Exception as e:
            self.log_debug(f"Unexpected error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
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

            status = lp_model.status

            if status == GRB.OPTIMAL:
                return lp_model.objVal, lp_model.status, lp_model.getVars()
            elif status == GRB.INFEASIBLE:
                self.log_debug("Model is infeasible. Computing IIS...")
                lp_model.computeIIS()
                self.log_debug("\nIIS Constraints:")
                for c in lp_model.getConstrs():
                    if c.IISConstr: self.log_debug(f"{c.ConstrName}")
                self.log_debug("\nIIS Variables:")
                for v in lp_model.getVars():
                    if v.IISLB or v.IISUB: self.log_debug(f"{v.VarName}")
                return None, status, None
            elif status == GRB.UNBOUNDED:
                self.log_debug("Model is unbounded!")
                return None, status, None
            else:
                self.log_debug(f"Model status: {lp_model.status}")
                return None, lp_model.status, None
        finally:
            # Ensure stdout is restored even if an error occurs
            sys.stdout = old_stdout

    def ml_model_prediction(self, year_data, granularity, selected_court=None, selected_municipality=None):
        """Predict the target variable using the machine learning model with the default allocations"""
        # Exclude target columns when making predictions
        feature_data = year_data.drop(columns=target_specific)

        if granularity == "Country":
            ml_prediction = self.model.predict(feature_data).sum().sum()

        elif granularity == "Court":
            court_col = f'Court_{selected_court}'
            court_data = feature_data[feature_data[court_col] == 1]
            ml_prediction = self.model.predict(court_data).sum().sum()

        elif granularity == "Municipality":
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = feature_data[feature_data[mun_col] == 1]
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
            court_col = f'Court_{selected_court}'
            mun_col = f'Municipality_{selected_municipality}'
            mun_data = year_data[(year_data[court_col] == 1) & (year_data[mun_col] == 1)]
            real_value = mun_data[target_specific].sum().sum()

        return real_value

    def update_dropdowns(self):
        """Update dropdown options based on selected year"""
        selected_year = self.year_var.get()
        
        if selected_year == "New Year":
            # For "New Year", use the most recent year's data to populate dropdowns
            year_data = data[data['Year'] == data['Year'].max()]
        else:
            year_data = data[data['Year'] == int(selected_year)]
        
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
        selected_year = self.year_var.get()
        selected_court = self.court_var.get()
        
        if selected_court:
            if selected_year == "New Year":
                # For "New Year", use the most recent year's data
                year_data = data[data['Year'] == data['Year'].max()]
            else:
                year_data = data[data['Year'] == int(selected_year)]
            
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
        selected_granularity = self.granularity_var.get()
        selected_year = self.year_var.get()
        self.log_debug(f"[DEBUG] Granularity changed to: {selected_granularity} (Year: {selected_year})")

        if selected_granularity == "Country":
            self.region_frame.grid_remove()
            self.court_var.set("")
            self.municipality_var.set("")
            self.hide_manual_input() # Hide manual input when switching to Country
            self.hide_benches()
        else:
            self.region_frame.grid()
            self.court_label.grid()
            self.court_dropdown.grid()

            # Only show municipality selection for Municipality granularity
            if selected_granularity == "Municipality":
                self.municipality_label.grid()
                self.municipality_dropdown.grid()
                self.update_municipality_dropdown()

                # Show manual input and benches immediately if in "New Year" mode
                if selected_year == "New Year":
                     self.log_debug("[DEBUG] In New Year mode and Municipality granularity. Calling show_manual_input and display_benches.")
                     self.show_manual_input()
                     self.display_benches()

            else: # Court granularity
                self.municipality_label.grid_remove()
                self.municipality_dropdown.grid_remove()
                self.municipality_var.set("")
                self.hide_manual_input() # Hide manual input for Court granularity
                self.hide_benches()

                # Show manual input immediately if in "New Year" mode (though this path should be rare now)
                # if selected_year == "New Year": # This condition should not be met with New Year limited to Municipality
                #      self.show_manual_input() # Keep commented

            self.update_dropdowns()

    def on_year_change(self, event=None):
        """Handle year selection change"""
        selected_year = self.year_var.get()
        self.log_debug(f"[DEBUG] Year changed to: {selected_year}")

        # Clear previous results when year changes
        for tab in self.results_notebook.tabs():
            self.results_notebook.forget(tab)

        if selected_year == "New Year":
            self.log_debug("[DEBUG] New Year selected. Setting granularity to Municipality only.")
            # Only allow Municipality granularity for New Year
            self.granularity_dropdown['values'] = ["Municipality"]
            self.granularity_dropdown.current(0)
            self.granularity_var.set("Municipality")

            # Show region selection
            self.region_frame.grid()
            self.court_label.grid()
            self.court_dropdown.grid()
            self.municipality_label.grid()
            self.municipality_dropdown.grid()

            # Update dropdowns
            self.update_dropdowns()

            # show_manual_input and display_benches will be called by on_granularity_change and on_municipality_change
            # self.show_manual_input() # Removed direct call

        else:
            self.log_debug(f"[DEBUG] Historical year {selected_year} selected.")
            # Restore all granularities
            self.granularity_dropdown['values'] = ["Country", "Court", "Municipality"]

            # Update dropdowns
            self.update_dropdowns()

            # Hide manual input notebook and benches
            self.hide_manual_input()
            self.hide_benches()

            # Update region frame visibility based on granularity
            if self.granularity_var.get() == "Country":
                self.region_frame.grid_remove()
            else:
                self.region_frame.grid()
                self.court_label.grid()
                self.court_dropdown.grid()
                if self.granularity_var.get() == "Municipality":
                    self.municipality_label.grid()
                    self.municipality_dropdown.grid()
                else:
                    self.municipality_label.grid_remove()
                    self.municipality_dropdown.grid_remove()

    def show_manual_input(self):
        """Show the manual input notebook"""
        selected_year = self.year_var.get()
        selected_granularity = self.granularity_var.get()
        selected_municipality = self.municipality_var.get()
        selected_court = self.court_var.get() # Get selected court
        self.log_debug(f"[DEBUG] show_manual_input called (Year: {selected_year}, Granularity: {selected_granularity}, Municipality: {selected_municipality})")

        # Destroy previous manual input widgets if they exist
        if self.manual_input_notebook:
            self.manual_input_notebook.destroy()
            self.manual_input_notebook = None
        if hasattr(self, 'manual_frame') and self.manual_frame:
             self.manual_frame.destroy()
             self.manual_frame = None

        # Manual input is only for Municipality granularity in New Year mode
        if not (selected_year == "New Year" and selected_granularity == "Municipality"):
            self.log_debug("[DEBUG] show_manual_input: Not in New Year Municipality mode. Hiding input.")
            self.hide_manual_input() # Ensure manual input is hidden if not in correct mode
            return

        if not selected_municipality or not selected_court:
             self.log_debug("[DEBUG] show_manual_input: No municipality or court selected. Hiding input.")
             self.hide_manual_input() # Hide if no municipality selected
             return

        # Get the list of benches for the selected municipality (using 2023 data structure)
        year_data_2023 = data[data['Year'] == 2023]
        court_col = f'Court_{selected_court}'
        mun_col = f'Municipality_{selected_municipality}'

        # Ensure columns exist before filtering
        if court_col not in year_data_2023.columns or mun_col not in year_data_2023.columns:
             self.log_debug(f"[DEBUG] show_manual_input: Court ({court_col}) or Municipality ({mun_col}) column not found in 2023 data for manual input display.")
             # Still show the frame but maybe with a message?
             # For now, hide if benches cannot be determined.
             self.hide_manual_input()
             return

        mun_data_2023 = year_data_2023[(year_data_2023[court_col] == 1) & (year_data_2023[mun_col] == 1)]
        bench_cols = [col for col in mun_data_2023.columns if col.startswith('Bench_') and mun_data_2023[col].any()]

        if not bench_cols:
             self.log_debug(f"[DEBUG] show_manual_input: No benches found for {selected_court} - {selected_municipality} in 2023 data. Hiding input.")
             self.hide_manual_input()
             return

        benches = sorted(bench_cols)
        self.log_debug(f"[DEBUG] show_manual_input: Benches found: {benches}")

        self.log_debug(f"[DEBUG] show_manual_input: Creating manual input for municipality {selected_municipality} and benches {benches}")

        # Combine staff and case variables for manual input
        all_manual_input_vars = self.optimization.staff_vars + prev_year_case_cols
        self.log_debug(f"Manual input variables: {all_manual_input_vars}")

        # Create a frame for manual input between control panel and results/bench display
        manual_frame = ttk.LabelFrame(self.main_frame, text="Manual Staff & Case Input", padding="10")

        # In New Year Municipality mode, manual input should be in row 2 (below control panel and bench display)
        manual_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        self.main_frame.grid_rowconfigure(2, weight=0)

        # Results frame is in row 3
        self.results_frame.grid(row=3, column=0, sticky="nsew", pady=5)
        self.main_frame.grid_rowconfigure(3, weight=1)

        self.manual_frame = manual_frame # Store reference to manual frame

        # Initialize manual_input_data structure if it's None
        if self.manual_input_data is None:
            self.manual_input_data = {}
        # Ensure the municipality key exists and is initialized
        if selected_municipality not in self.manual_input_data:
             self.manual_input_data[selected_municipality] = {var: 0 for var in self.optimization.staff_vars}
        # Ensure bench keys exist and are initialized with case variables
        for bench in benches:
             if bench not in self.manual_input_data:
                  self.manual_input_data[bench] = {var: 0 for var in prev_year_case_cols}

        # Create manual input notebook
        self.manual_input_notebook = ManualInputNotebook(
            manual_frame,
            selected_municipality, # Pass municipality name
            benches,               # Pass list of benches
            all_manual_input_vars, # Pass all variables for initialization
            self.on_manual_input_change
        )
        self.manual_input_notebook.grid(row=0, column=0, sticky="nsew")
        manual_frame.grid_rowconfigure(0, weight=1)
        manual_frame.grid_columnconfigure(0, weight=1)

    def hide_manual_input(self):
        """Hide the manual input notebook and its frame"""
        self.log_debug("[DEBUG] hide_manual_input called.")
        # Destroy manual input widgets if they exist
        if self.manual_input_notebook:
            self.manual_input_notebook.destroy()
            self.manual_input_notebook = None
        # Note: self.manual_input_data is NOT cleared here, preserving entered values

        if hasattr(self, 'manual_frame') and self.manual_frame:
            self.manual_frame.destroy()
            self.manual_frame = None

        # Restore results frame position and adjust row weights
        # Check if bench display is visible to determine results frame position
        if hasattr(self, 'bench_display_frame') and self.bench_display_frame and self.bench_display_frame.winfo_exists() and self.bench_display_frame.winfo_ismapped():
             self.log_debug("[DEBUG] hide_manual_input: Bench display is mapped. Gridding results in row 2.")
             # Results frame is in row 2 (below control panel and bench display)
             self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
             self.main_frame.grid_rowconfigure(2, weight=1)
             # Ensure other rows don't take up results space
             self.main_frame.grid_rowconfigure(0, weight=0) # Control panel
             self.main_frame.grid_rowconfigure(1, weight=0) # Bench display
             self.main_frame.grid_rowconfigure(3, weight=0) # Status bar
        else:
             self.log_debug("[DEBUG] hide_manual_input: Bench display is NOT mapped. Gridding results in row 1.")
             # Results frame is in row 1 (below control panel)
             self.results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
             self.main_frame.grid_rowconfigure(1, weight=1)
             # Ensure other rows don't take up results space
             self.main_frame.grid_rowconfigure(0, weight=0) # Control panel
             self.main_frame.grid_rowconfigure(2, weight=0) # Manual input (was here)
             self.main_frame.grid_rowconfigure(3, weight=0) # Status bar

    def on_manual_input_change(self, region: str, staff_or_case: str, value: int):
        """Handle changes in manual input values"""
        if self.manual_input_data is None:
            self.manual_input_data = {}

        if region not in self.manual_input_data:
            # This should ideally not happen with the initialization in show_manual_input,
            # but as a safeguard, initialize if necessary.
            if region == self.municipality_var.get(): # This is the municipality key
                 self.manual_input_data[region] = {var: 0 for var in self.optimization.staff_vars}
            else: # This is a bench key
                 self.manual_input_data[region] = {var: 0 for var in prev_year_case_cols}

        # Ensure value is an integer
        try:
            value = int(value)
        except ValueError:
            self.log_debug(f"Invalid input value for {staff_or_case} in {region}: {value}. Input must be an integer.")
            # Revert the entry widget to the previous valid value or 0. Requires accessing the widget.
            # For now, we will just log and the invalid input will not update the data.
            return # Do not update manual_input_data with invalid value

        self.manual_input_data[region][staff_or_case] = value
        self.log_debug(f"Manual input updated: Region={region}, Variable={staff_or_case}, Value={value}")

    def on_court_change(self, event=None):
        """Handle court selection change"""
        # Update municipality dropdown if granularity is Municipality
        if self.granularity_var.get() == "Municipality":
            self.update_municipality_dropdown()

        # Trigger manual input and bench display update if in New Year Municipality mode
        # on_municipality_change will be triggered after dropdown update and will call show_manual_input and display_benches
        pass # No direct action needed here as update_municipality_dropdown triggers on_municipality_change

    def on_municipality_change(self, event=None):
        """Handle municipality selection change"""
        selected_year = self.year_var.get()
        selected_granularity = self.granularity_var.get()
        selected_municipality = self.municipality_var.get()
        self.log_debug(f"[DEBUG] Municipality changed to: {selected_municipality} (Year: {selected_year}, Granularity: {selected_granularity})")

        # Update manual input and bench display if in "New Year" mode and Municipality granularity
        if selected_year == "New Year" and selected_granularity == "Municipality" and selected_municipality:
            self.log_debug("[DEBUG] In New Year mode, Municipality granularity, and municipality selected. Calling show_manual_input and display_benches.")
            self.show_manual_input()
            self.display_benches()
        else:
             self.log_debug("[DEBUG] Not in New Year mode, Municipality granularity, or no municipality selected. Hiding manual input and benches.")
             self.hide_manual_input()
             self.hide_benches()

    def display_benches(self):
        """Display the benches for the selected municipality (for New Year Municipality granularity)"""
        selected_year = self.year_var.get()
        selected_granularity = self.granularity_var.get()
        selected_court = self.court_var.get()
        selected_municipality = self.municipality_var.get()
        self.log_debug(f"[DEBUG] display_benches called (Year: {selected_year}, Granularity: {selected_granularity}, Court: {selected_court}, Municipality: {selected_municipality})")

        # Only show benches in New Year Municipality mode with selections
        if not (selected_year == "New Year" and selected_granularity == "Municipality" and selected_court and selected_municipality):
             self.log_debug("[DEBUG] display_benches: Not in correct mode or missing selection. Hiding benches.")
             self.hide_benches() # Ensure benches are hidden if not in correct mode
             return

        # Create bench display frame and label if they don't exist
        if not hasattr(self, 'bench_display_frame') or not self.bench_display_frame:
            self.bench_display_frame = ttk.LabelFrame(self.main_frame, text="Benches in Municipality", padding="10")
            self.bench_display_label = ttk.Label(self.bench_display_frame, text="")
            self.bench_display_label.grid(row=0, column=0, sticky=tk.W)

        # Use 2023 data to find benches
        year_data_2023 = data[data['Year'] == 2023]
        court_col = f'Court_{selected_court}'
        mun_col = f'Municipality_{selected_municipality}'

        # Ensure columns exist before filtering
        if court_col not in year_data_2023.columns or mun_col not in year_data_2023.columns:
             self.log_debug(f"[DEBUG] display_benches: Court ({court_col}) or Municipality ({mun_col}) column not found in 2023 data for bench display.")
             self.bench_display_label.config(text="Error: Could not retrieve bench information.")
             # Always grid the bench display frame when display_benches is called, even on error
             self.bench_display_frame.grid(row=1, column=0, sticky="ew", pady=5)
             self.main_frame.grid_rowconfigure(1, weight=0)
             # Adjust manual input and results frame positions
             if hasattr(self, 'manual_frame') and self.manual_frame and self.manual_frame.winfo_ismapped():
                  self.log_debug("[DEBUG] display_benches: Manual input is mapped. Gridding manual input in row 2 and results in row 3.")
                  self.manual_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                  self.main_frame.grid_rowconfigure(2, weight=0)
                  self.results_frame.grid(row=3, column=0, sticky="nsew", pady=5)
                  self.main_frame.grid_rowconfigure(3, weight=1)
             else:
                  # Manual input should be visible here in New Year Municipality mode, but as a fallback
                  self.log_debug("[DEBUG] display_benches: Manual input is NOT mapped. Gridding results in row 2 (fallback). ")
                  self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                  self.main_frame.grid_rowconfigure(2, weight=1)
                  self.main_frame.grid_rowconfigure(3, weight=0)
             return

        mun_data_2023 = year_data_2023[(year_data_2023[court_col] == 1) & (year_data_2023[mun_col] == 1)]
        bench_cols = [col for col in mun_data_2023.columns if col.startswith('Bench_') and mun_data_2023[col].any()]

        if bench_cols:
            bench_names = sorted([col.replace('Bench_', '') for col in bench_cols])
            benches_text = "Benches: " + ", ".join(bench_names)
            self.bench_display_label.config(text=benches_text)
            # Grid the bench display frame in row 1
            self.bench_display_frame.grid(row=1, column=0, sticky="ew", pady=5)
            self.main_frame.grid_rowconfigure(1, weight=0) # Don't allow bench display to expand
            self.log_debug("[DEBUG] display_benches: Benches found. Gridding bench display in row 1.")
            # Adjust manual input and results frame positions
            if hasattr(self, 'manual_frame') and self.manual_frame and self.manual_frame.winfo_ismapped():
                 self.log_debug("[DEBUG] display_benches: Manual input is mapped. Gridding manual input in row 2 and results in row 3.")
                 self.manual_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(2, weight=0)
                 self.results_frame.grid(row=3, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(3, weight=1)
            else:
                 # Manual input should be visible here in New Year Municipality mode, but as a fallback
                 self.log_debug("[DEBUG] display_benches: Manual input is NOT mapped. Gridding results in row 2 (fallback). ")
                 self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(2, weight=1)
                 self.main_frame.grid_rowconfigure(3, weight=0)
        else:
            self.log_debug("[DEBUG] display_benches: No benches found. Gridding bench display in row 1.")
            self.bench_display_label.config(text="No benches found for this municipality in 2023 data.")
            # Grid the bench display frame in row 1
            self.bench_display_frame.grid(row=1, column=0, sticky="ew", pady=5)
            self.main_frame.grid_rowconfigure(1, weight=0)
            # Adjust manual input and results frame positions
            if hasattr(self, 'manual_frame') and self.manual_frame and self.manual_frame.winfo_ismapped():
                 self.log_debug("[DEBUG] display_benches: Manual input is mapped. Gridding manual input in row 2 and results in row 3.")
                 self.manual_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(2, weight=0)
                 self.results_frame.grid(row=3, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(3, weight=1)
            else:
                 # Manual input should be visible here, but as a fallback
                 self.log_debug("[DEBUG] display_benches: Manual input is NOT mapped. Gridding results in row 2 (fallback).")
                 self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(2, weight=1)
                 self.main_frame.grid_rowconfigure(3, weight=0)

    def hide_benches(self):
        """Hide the bench display frame"""
        self.log_debug("[DEBUG] hide_benches called.")
        if hasattr(self, 'bench_display_frame') and self.bench_display_frame:
            self.bench_display_frame.grid_remove()
            # Adjust manual input and results frame positions if needed
            if hasattr(self, 'manual_frame') and self.manual_frame and self.manual_frame.winfo_ismapped():
                 self.log_debug("[DEBUG] hide_benches: Manual input is mapped. Gridding manual input in row 1 and results in row 2.")
                 self.manual_frame.grid(row=1, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(1, weight=0)
                 self.results_frame.grid(row=2, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(2, weight=1)
                 self.main_frame.grid_rowconfigure(3, weight=0)
            else:
                 self.log_debug("[DEBUG] hide_benches: Manual input is NOT mapped. Gridding results in row 1.")
                 self.results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
                 self.main_frame.grid_rowconfigure(1, weight=1)
                 self.main_frame.grid_rowconfigure(2, weight=0)
                 self.main_frame.grid_rowconfigure(3, weight=0)

    def create_manual_input_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame from manual input data"""
        # Create a base DataFrame with all necessary columns
        base_data = pd.DataFrame()

        # Get the most recent year's data structure to copy
        year_data_structure = data[data['Year'] == data['Year'].max()].drop(columns=target_specific)

        # Add staff and case columns
        manual_input_cols = self.optimization.staff_vars + prev_year_case_cols
        for col in manual_input_cols:
            base_data[col] = 0

        # Add court, municipality, and bench columns based on the structure
        for col in year_data_structure.columns:
             if col.startswith('Court_') or col.startswith('Municipality_') or col.startswith('Bench_'):
                  base_data[col] = 0

        created_rows = []

        # Manual input is only for Municipality granularity in New Year mode
        selected_year = self.year_var.get()
        selected_granularity = self.granularity_var.get()

        if selected_granularity == "Municipality" and selected_year == "New Year":
            court = self.court_var.get()
            municipality = self.municipality_var.get()
            court_col = f'Court_{court}'
            mun_col = f'Municipality_{municipality}'

            # Find benches for this specific municipality in the original data structure (using latest year for structure)
            # Ensure columns exist before filtering
            if court_col not in year_data_structure.columns or mun_col not in year_data_structure.columns:
                 self.log_debug(f"Court ({court_col}) or Municipality ({mun_col}) column not found in latest year data structure for manual input DataFrame creation.")
                 return pd.DataFrame(columns=base_data.columns)

            mun_benches_data = year_data_structure[(year_data_structure[court_col] == 1) & (year_data_structure[mun_col] == 1)]
            bench_cols = [col for col in mun_benches_data.columns if col.startswith('Bench_') and mun_benches_data[col].any()]

            if not bench_cols:
                 self.log_debug(f"No benches found in original data for {court} - {municipality}. Cannot create manual input rows.")
                 # Return an empty DataFrame if no benches are found
                 return pd.DataFrame(columns=base_data.columns)

            # Get manual input values for municipality (staff) and benches (cases)
            municipality_values = self.manual_input_data.get(municipality, {}) # Staff values are under municipality key

            for bench_col in bench_cols:
                row = {col: 0 for col in base_data.columns} # Initialize row with 0s
                row[court_col] = 1
                row[mun_col] = 1
                row[bench_col] = 1

                # Add staff values (from municipality input)
                for var in self.optimization.staff_vars:
                    value = municipality_values.get(var, 0)
                    if var == 'Judges' and value > 0:
                        # Distribute judges - ensure at least one per bench if total > 0
                         num_judges = municipality_values.get('Judges', 0)
                         if num_judges > 0:
                              judges_per_bench = num_judges // len(bench_cols)
                              remainder_judges = num_judges % len(bench_cols)
                              # Assign judges to this bench (distribute remainder first)
                              row[var] = judges_per_bench + (1 if bench_cols.index(bench_col) < remainder_judges else 0)

                    else:
                         # Evenly distribute other staff from municipality total
                         total_staff = municipality_values.get(var, 0)
                         row[var] = total_staff // len(bench_cols)

                # Add case values (from bench input)
                bench_values = self.manual_input_data.get(bench_col, {}) # Case values are under bench key
                for var in prev_year_case_cols:
                    row[var] = bench_values.get(var, 0)

                created_rows.append(row)

        # Create DataFrame from the list of created rows
        if created_rows:
            manual_df = pd.DataFrame(created_rows)
        else:
            # Return an empty DataFrame with correct columns if no rows were created
            manual_df = pd.DataFrame(columns=base_data.columns)

        # Add target columns with zeros
        for target in target_specific:
            manual_df[target] = 0

        self.log_debug(f"Created manual input DataFrame with {manual_df.shape[0]} rows and {manual_df.shape[1]} columns.")
        self.log_debug(f"Manual DataFrame columns: {manual_df.columns.tolist()}")
        self.log_debug(f"Manual DataFrame head:\n{manual_df.head()}")

        return manual_df

