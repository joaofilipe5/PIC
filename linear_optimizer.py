import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from gurobipy import GRB

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
        
        # Year input
        ttk.Label(main_frame, text="Year:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.year_var = tk.StringVar()
        self.year_entry = ttk.Entry(main_frame, textvariable=self.year_var)
        self.year_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Region input
        ttk.Label(main_frame, text="Region (optional):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.region_var = tk.StringVar()
        self.region_entry = ttk.Entry(main_frame, textvariable=self.region_var)
        self.region_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Min Judges
        ttk.Label(main_frame, text="Min Judges per Court:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.min_judges_var = tk.IntVar(value=1)
        self.min_judges_spin = ttk.Spinbox(main_frame, from_=0, to=10, textvariable=self.min_judges_var, width=5)
        self.min_judges_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Run button
        self.run_button = ttk.Button(main_frame, text="Run Optimization", command=self.on_run)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=15)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Debug console (hidden by default)
        self.debug_text = tk.Text(main_frame, height=5, state=tk.DISABLED)
        self.debug_text.grid(row=5, column=0, columnspan=2, sticky=tk.EW)
        self.show_debug(False)  # Hide by default
        
        # Add debug toggle button (for troubleshooting)
        ttk.Button(main_frame, text="Debug", command=lambda: self.show_debug()).grid(row=6, column=0, columnspan=2)

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

    def load_model(self):
        """Load the machine learning model with error handling"""
        try:
            model_path = 'ridge.joblib'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
            
            self.log_debug(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            self.log_debug(f"Model loading failed: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            return False

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
            
            region = self.region_var.get().strip() or None
            min_judges = self.min_judges_var.get()
            
            if min_judges < 0:
                messagebox.showerror("Input Error", "Minimum judges cannot be negative")
                return
            
            # Load model if not already loaded
            if self.model is None and not self.load_model():
                return
            
            # Disable UI during processing
            self.run_button.config(state=tk.DISABLED)
            self.status_var.set("Optimizing...")
            self.root.update()
            
            # Run optimization (simplified for debugging)
            self.log_debug(f"Running optimization for year {year}, region {region}, min judges {min_judges}")
            
            # Simulate optimization (replace with your actual code)
            result = self.simulate_optimization(year, region, min_judges)
            
            messagebox.showinfo("Result", f"Optimization complete!\nObjective value: {result:.2f}")
            
        except Exception as e:
            self.log_debug(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.status_var.set("Ready")

    def simulate_optimization(self, year, region, min_judges):
        """Simulate optimization for debugging"""
        self.log_debug("Simulating optimization...")
        return 42.0  # Dummy result

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