micromamba activate 312
for i in *.ipynb; do
    [ -f "$i" ] || break
    echo "$i"
    nbqa mypy "$i"
done
for i in *.py; do
    [ -f "$i" ] || break
    echo "$i"
    mypy "$i"
done

