# Do the python files first
for i in *.py; do
    [ -f "$i" ] || break
    echo "$i"
    pylint "$i"
done

# then jupyter notebooks
for i in *.ipynb; do
    [ -f "$i" ] || break
    echo "$i"
    nbqa pylint --disable=C0114,W0106,W0104 "$i"
done

# Note, these are the pylint codes that are disabled for 
#       jupyter notebooks:
# C0114: Missing module docstring [Not practical for jupyter notebooks]
# W0106: Expression is assigned to nothing [Often used in jupyter notebooks]
# W0104: Statement seems to have no effect [Often used in jupyter notebooks]
