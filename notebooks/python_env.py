# python_env.py

# --- initialisation

import sys
import platform
import psutil

## --- utility functions

def python_env():
    """This function prints the key system architecture and lists
       the headline imported modules with version numnbers.
       There are no arguments for this function.
       The function returns None."""
    
    N = 50
    
    # system architecture
    print('System:')
    print('='*N)
    print(f"System:     {platform.system()}")
    print(f"Release:    {platform.release()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Processor:  {platform.processor()}")
    print(f"RAM:        {round(psutil.virtual_memory().total / 1024**3)}GB")
    print(f"Python:     {platform.python_version()}")

    # headline modules with version numbers
    print('\nModules:')
    print('-'*N)
    modules = {}
    for m in sys.modules.keys():
        if m.startswith('_') or '.' in m:
            continue
        if hasattr(sys.modules[m], '__version__'):
            modules[m] = sys.modules[m] 

    for m in sorted(modules.keys(), key=str.casefold):
        print(f"{m}: {modules[m].__version__}")

    return(None)