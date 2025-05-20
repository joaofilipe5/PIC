import tkinter as tk
from tkinter import messagebox

from src.app import App

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        app = App(root)
        
        # Add error handling for the main loop
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    
    except Exception as e:
        messagebox.showerror("Fatal Error", f"The application crashed:\n{str(e)}")
        raise