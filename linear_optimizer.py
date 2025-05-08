import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from gurobipy import GRB
from data_prep import data_prep

data, target_specific, target_all = data_prep('data.xlsx')

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        
        # Initialize model as None (we'll load it when needed)
        self.model = None
        self.status_var.set("Welcome - Ready to optimize")

    def setup_ui(self):
        self.root.title("Court Staff Optimizer")
        self.root.geometry("500x300")
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
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.region_frame.columnconfigure(1, weight=1)
        
        # Debug console (hidden by default)
        self.debug_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.debug_text.grid(row=6, column=0, columnspan=2, sticky=tk.EW)
        self.show_debug(False)  # Hide by default
        
        # Add debug toggle button (for troubleshooting)
        ttk.Button(main_frame, text="Debug", command=lambda: self.show_debug()).grid(row=7, column=0, columnspan=2)

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
            model_path = f'{model_name}.joblib'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
            
            self.log_debug(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            self.log_debug(f"Model loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return False

    def add_constraints(self):
        pass

    def calculate_staff_max(self, year_data, granularity, selected_court=None, selected_municipality=None):
        """Calculate maximum staffing possibilities based on granularity level"""
        staff_vars = ['Judges', 'Justice Secretary', 'Law Clerck', 'Auxiliar Clerck', 
                     'Administrative/Technical People', 'Operational/Auxiliar People']
        staff_max = {}

        # Get all court columns
        court_columns = [col for col in year_data.columns if col.startswith('Court_')]
        courts = [col.replace('Court_', '') for col in court_columns]

        if granularity == "Country":
            # Calculate maximums for each court
            for court in courts:
                court_col = f'Court_{court}'
                court_data = year_data[year_data[court_col] == 1]
                for staff in staff_vars:
                    staff_max[(court, staff)] = court_data[staff].sum()

        elif granularity == "Court":
            if not selected_court:
                return {}
            # Only calculate for selected court
            court_col = f'Court_{selected_court}'
            court_data = year_data[year_data[court_col] == 1]
            for staff in staff_vars:
                staff_max[staff] = court_data[staff].sum()

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
            
            # Calculate staff maximums based on granularity
            staff_max = self.calculate_staff_max(year_data, granularity, selected_court, selected_municipality)
            
            if not staff_max:
                messagebox.showerror("Error", "No data available for the selected parameters")
                return
            
            self.log_debug(f"Running optimization for year {year}, granularity {granularity}")
            if selected_court:
                self.log_debug(f"Selected court: {selected_court}")
            if selected_municipality:
                self.log_debug(f"Selected municipality: {selected_municipality}")
            self.log_debug(f"Staff maximums: {staff_max}")
            
            # Run optimization (simplified for debugging)
            result = self.simulate_optimization(year, granularity, selected_court, selected_municipality, staff_max)
            
            messagebox.showinfo("Result", f"Optimization complete!\nObjective value: {result:.2f}")
            
        except Exception as e:
            self.log_debug(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")

    def simulate_optimization(self, year, granularity, selected_court=None, selected_municipality=None, staff_max=None):
        """Simulate optimization for debugging"""
        self.log_debug("Simulating optimization...")
        return 42.0  # Dummy result

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

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        app = OptimizationApp(root)
        
        # Add error handling for the main loop
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    
    except Exception as e:
        messagebox.showerror("Fatal Error", f"The application crashed:\n{str(e)}")
        raise