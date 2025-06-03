import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Callable

class ManualInputFrame(ttk.Frame):
    def __init__(self, parent, variables: List[str], initial_values: Dict[str, int], on_value_change: Optional[Callable] = None):
        super().__init__(parent)
        self.variables = variables # Now accepts all variables (staff and cases)
        self.on_value_change = on_value_change
        self.values: Dict[str, int] = initial_values.copy() # Initialize with provided values
        self.create_widgets()

    def create_widgets(self):
        # Create a main frame for the input fields
        input_frame = ttk.Frame(self, padding="10")
        input_frame.grid(row=0, column=0, sticky='nsew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Organize variables into categories
        staff_vars = [var for var in self.variables if var in ["Judges", "Justice Secretary", "Law Clerck", "Auxiliar Clerck", "Administrative/Technical People", "Operational/Auxiliar People"]]
        pending_cases_vars = [var for var in self.variables if var.startswith('PC_') and var.endswith('_prev_year')]
        completed_cases_vars = [var for var in self.variables if var.startswith('CC_') and var.endswith('_prev_year')]

        current_col = 0

        # Add input fields for staff variables (if present)
        if staff_vars:
            staff_frame = ttk.LabelFrame(input_frame, text="Staff", padding="10")
            staff_frame.grid(row=0, column=current_col, sticky='nsew', padx=5, pady=5)
            input_frame.grid_columnconfigure(current_col, weight=1)

            for i, var_name in enumerate(staff_vars):
                ttk.Label(staff_frame, text=var_name).grid(row=i, column=0, sticky='w', padx=5, pady=2)
                var = tk.StringVar(value=str(self.values.get(var_name, 0)))
                var.trace_add("write", lambda *args, s=var_name, v=var: self._on_value_change(s, v))
                entry = ttk.Entry(staff_frame, textvariable=var, width=10)
                entry.grid(row=i, column=1, sticky='e', padx=5, pady=2)
                staff_frame.grid_columnconfigure(1, weight=1) # Allow entry column to expand
            current_col += 1

        # Add input fields for pending cases (if present)
        if pending_cases_vars:
            pending_cases_frame = ttk.LabelFrame(input_frame, text="Pending Cases (Previous Year)", padding="10")
            pending_cases_frame.grid(row=0, column=current_col, sticky='nsew', padx=5, pady=5)
            input_frame.grid_columnconfigure(current_col, weight=1)

            for i, var_name in enumerate(pending_cases_vars):
                 ttk.Label(pending_cases_frame, text=var_name.replace('_prev_year', '')).grid(row=i, column=0, sticky='w', padx=5, pady=2)
                 var = tk.StringVar(value=str(self.values.get(var_name, 0)))
                 var.trace_add("write", lambda *args, s=var_name, v=var: self._on_value_change(s, v))
                 entry = ttk.Entry(pending_cases_frame, textvariable=var, width=10)
                 entry.grid(row=i, column=1, sticky='e', padx=5, pady=2)
                 pending_cases_frame.grid_columnconfigure(1, weight=1) # Allow entry column to expand
            current_col += 1

        # Add input fields for completed cases (if present)
        if completed_cases_vars:
            completed_cases_frame = ttk.LabelFrame(input_frame, text="Completed Cases (Previous Year)", padding="10")
            completed_cases_frame.grid(row=0, column=current_col, sticky='nsew', padx=5, pady=5)
            input_frame.grid_columnconfigure(current_col, weight=1)

            for i, var_name in enumerate(completed_cases_vars):
                 ttk.Label(completed_cases_frame, text=var_name.replace('_prev_year', '')).grid(row=i, column=0, sticky='w', padx=5, pady=2)
                 var = tk.StringVar(value=str(self.values.get(var_name, 0)))
                 var.trace_add("write", lambda *args, s=var_name, v=var: self._on_value_change(s, v))
                 entry = ttk.Entry(completed_cases_frame, textvariable=var, width=10)
                 entry.grid(row=i, column=1, sticky='e', padx=5, pady=2)
                 completed_cases_frame.grid_columnconfigure(1, weight=1) # Allow entry column to expand
            current_col += 1

    def _on_value_change(self, var_name: str, var: tk.StringVar):
        try:
            value = int(var.get())
            if value < 0:
                value = 0
                var.set("0")
            self.values[var_name] = value
            if self.on_value_change:
                # Pass the variable name instead of staff
                self.on_value_change(var_name, value)
        except ValueError:
            var.set(str(self.values.get(var_name, 0))) # Revert to previous valid value on error

    def get_values(self) -> Dict[str, int]:
        return self.values.copy()

class ManualInputNotebook(ttk.Notebook):
    def __init__(self, parent, municipality: str, benches: List[str], variables: List[str], on_value_change: Optional[Callable] = None):
        super().__init__(parent)
        self.municipality = municipality
        self.benches = benches
        self.variables = variables # Contains both staff and case variables
        self.on_value_change = on_value_change
        self.input_frames: Dict[str, ManualInputFrame] = {}
        # Initialize self.values based on existing data or default to 0
        if not hasattr(parent.master.master.master.master, 'manual_input_data') or parent.master.master.master.master.manual_input_data is None:
             self.values: Dict[str, Dict[str, int]] = {municipality: {var: 0 for var in variables}}
             for bench in benches:
                 self.values[bench] = {var: 0 for var in variables if var.startswith('PC_') or var.startswith('CC_')}
        else:
            app_instance = parent.master.master.master.master # Access the App instance
            self.values = app_instance.manual_input_data # Use the existing data
            # Ensure all necessary keys and variables are present, initializing if needed
            if municipality not in self.values:
                 self.values[municipality] = {var: 0 for var in variables}
            else:
                 for var in variables:
                      if var not in self.values[municipality] and var in ["Judges", "Justice Secretary", "Law Clerck", "Auxiliar Clerck", "Administrative/Technical People", "Operational/Auxiliar People"]:
                           self.values[municipality][var] = 0
            for bench in benches:
                 if bench not in self.values:
                     self.values[bench] = {var: 0 for var in variables if var.startswith('PC_') or var.startswith('CC_')}
                 else:
                      for var in variables:
                           if var not in self.values[bench] and (var.startswith('PC_') or var.startswith('CC_')):
                                self.values[bench][var] = 0


        self.create_tabs()

    def create_tabs(self):
        # Clear existing tabs before creating new ones
        for tab in self.tabs():
            self.forget(tab)

        # Separate staff and case variables
        staff_vars = [var for var in self.variables if var in ["Judges", "Justice Secretary", "Law Clerck", "Auxiliar Clerck", "Administrative/Technical People", "Operational/Auxiliar People"]]
        case_vars = [var for var in self.variables if var.startswith('PC_') or var.startswith('CC_')]

        # Create Municipality tab for staff input
        mun_frame = ttk.Frame(self)
        self.add(mun_frame, text=self.municipality) # Use municipality name for the main tab

        # Pass only staff variables and initial values for the municipality
        input_frame = ManualInputFrame(mun_frame, staff_vars, self.values.get(self.municipality, {}),
                                     lambda var_name, value: self._on_value_change(self.municipality, var_name, value))
        input_frame.grid(row=0, column=0, sticky='nsew')
        mun_frame.grid_rowconfigure(0, weight=1)
        mun_frame.grid_columnconfigure(0, weight=1)
        self.input_frames[self.municipality] = input_frame

        # Create a tab for each bench for case input
        for bench in self.benches:
            bench_frame = ttk.Frame(self)
            self.add(bench_frame, text=bench.replace('Bench_', '')) # Use bench name for tab text

            # Pass only case variables and initial values for the bench
            input_frame = ManualInputFrame(bench_frame, case_vars, self.values.get(bench, {}),
                                         lambda var_name, value: self._on_value_change(bench, var_name, value))
            input_frame.grid(row=0, column=0, sticky='nsew')
            bench_frame.grid_rowconfigure(0, weight=1)
            bench_frame.grid_columnconfigure(0, weight=1)
            self.input_frames[bench] = input_frame

    def _on_value_change(self, region: str, var_name: str, value: int):
        # Update the internal values dictionary
        if region not in self.values:
            self.values[region] = {}
        self.values[region][var_name] = value

        # Call the external handler if provided
        if self.on_value_change:
            self.on_value_change(region, var_name, value)

    def get_values(self) -> Dict[str, Dict[str, int]]:
        # Retrieve values from each frame and update internal dictionary
        for region, frame in self.input_frames.items():
            self.values[region] = frame.get_values()
        return self.values.copy() 