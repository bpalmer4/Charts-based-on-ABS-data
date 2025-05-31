#!/bin/zsh

# Apply a set of lint checks (black, mypy, pylint, ruff) to 
# the calling arguments. Works for Python files and Jupyter 
# notebooks.

for arg in "$@"
do
    echo "========================================"
    if [[ ! -e "$arg" ]]; then
        echo "File or directory ($arg) not found, skipping ..."
        continue
    fi
    echo "Linting \"$arg\" ..."
    if [[ "$arg" == *.ipynb ]]; then
        echo "which is a Jupyter notebook ..."
        echo "black ..."
        nbqa black "$arg"
        echo "pylint ..."
        nbqa pylint "$arg"
        echo "ruff ..."
        nbqa ruff "$arg"
        echo "\n\nmypy ..."
        nbqa mypy "$arg"
        echo "\n\nChecking for type and pylint overrides ..."
        grep "# type: " "$arg"
        grep "# pylint: " "$arg"
        grep --regexp="from typing import .*cast" "$arg"
        grep "cast(" "$arg"
        continue
    fi
    if [[ "$arg" == *.py ]]; then
        echo "which is a Python file ..."
        echo "black ..."
        black "$arg"
        echo "pylint ..."
        pylint "$arg"
        echo "ruff ..."
        ruff check "$arg"
        echo "mypy ..."
        mypy "$arg"
        echo "Checking for type and pylint overrides ..."
        grep "# type: " "$arg"
        grep "# pylint: " "$arg"
        grep --regexp="from typing import .*cast" "$arg"
        grep "cast(" "$arg"
       continue
    fi
    echo "But file type not supported, skipping ..."
done
echo "========================================"
