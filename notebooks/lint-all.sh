#!/bin/zsh

# Loop through each argument passed to the script
for arg in "$@"
do
    # Perform linting on the file or directory specified by the argument
    if [[ ! -e "$arg" ]]; then
        echo "File or directory ($arg) not found, skipping..."
        continue
    fi
    echo "----------------------------------------"
    if [[ "$arg" == *.ipynb ]]; then
        echo "Linting Jupyter notebook ($arg) ..."
        nbqa black "$arg"
        nbqa mypy "$arg"
        nbqa pylint "$arg"
        nbqa ruff "$arg"
        # check to see if check has been disabled for the notebook
        grep "# type: " "$arg"
        grep "# pylint: " "$arg"
        continue
    fi
    if [[ "$arg" == *.py ]]; then
        echo "Linting Python file ($arg) ..."
        black "$arg"
        mypy "$arg"
        pylint "$arg"
        ruff check "$arg"
        # check to see if check has been disabled for the file
        grep "# type: " "$arg"
        grep "# pylint: " "$arg"
    fi
    echo "===> Done with $arg <==="
    echo "----------------------------------------"
done