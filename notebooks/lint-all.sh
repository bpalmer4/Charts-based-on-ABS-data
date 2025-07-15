#!/bin/zsh

# Apply a set of lint checks (mypy, ruff) to the calling arguments.i
# Note ruff looks to rules in ../pyproject.toml 
# Works for Python files and Jupyter notebooks.

for arg in "$@"
do
    echo "========================================"
    if [[ ! -e "$arg" ]]; then
        echo "File or directory ($arg) not found, skipping ..."
        continue
    fi
    echo "Linting \"$arg\" ..."
    if [[ "$arg" == *.ipynb ]]; then
        echo "whch is a Jupyter notebook ..."
        echo "ruff ..."
        nbqa ruff --fix "$arg"
        echo "mypy ..."
        nbqa mypy "$arg"
        continue
    fi
    if [[ "$arg" == *.py ]]; then
        echo "which is a Python file ..."
        echo "ruff ..."
        ruff check --fix"$arg"
        ruff format "$arg"
        echo "mypy ..."
        mypy "$arg"
        continue
    fi
    echo "But file type not supported, skipping ..."
done
echo "========================================"
