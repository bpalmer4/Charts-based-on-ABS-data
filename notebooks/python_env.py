# python_env.py

# --- initialisation

import sys
import platform
import psutil


# -- constants

line_size = 50


# --- functions

def python_system():
    """Print to standard output the key features of the system architecture.
       There are no arguments for this function.
       The function returns None."""
    
    print('System:')
    print('=' * line_size)
    print(f"System:     {platform.system()}")
    print(f"Release:    {platform.release()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Processor:  {platform.processor()}")
    print(f"RAM:        {round(psutil.virtual_memory().total / 1024**3)}GB")
    print(f"Python:     {platform.python_version()}")

    return None


def python_modules():
    """Print to standard output the headline imported modules with version 
       numbers. Note, modules without version numbers are excluded. 
       Sub-modules are excluded. Internal modules (beginning with an
       underscore) are excluded.
       There are no arguments for this function. 
       The function returns None."""

    print('Modules:')
    print('-' * line_size)
    modules = {}
    for m in sys.modules.keys():
        if m.startswith('_') or '.' in m:
            continue
        if hasattr(sys.modules[m], '__version__'):
            modules[m] = sys.modules[m] 
            
    for m in sorted(modules.keys(), key=str.casefold):
        print(f"{m}: {modules[m].__version__}")

    return None


def python_env():
    """Print to standard output the key features of the python environment.
       There are no arguments for this function.
       The function returns None."""
    
    python_system()
    print('')
    python_modules()
    return None