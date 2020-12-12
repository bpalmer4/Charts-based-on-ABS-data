# python_env.py

# --- initialisation

import sys
import platform
import psutil

import numpy as np
import pandas as pd
import matplotlib as mpl

# --- utility functions

def python_env():
    N = 50
    print('-'*N)
    print(f"System:     {platform.system()}")
    print(f"Release:    {platform.release()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Processor:  {platform.processor()}")
    print(f"RAM:        {round(psutil.virtual_memory().total / 1024**3)}GB")
    print('-'*N)
    print(f"Python:     {platform.python_version()}")
    print(f"Psutil:     {psutil.__version__}")
    print(f"Pandas:     {pd.__version__}")
    print(f"Numpy:      {np.__version__}")
    print(f"Matplotlib: {mpl.__version__}")
    print('-'*N)
