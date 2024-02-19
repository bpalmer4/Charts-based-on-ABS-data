micromamba activate 311
for i in *.ipynb; do
    [ -f "$i" ] || break
    echo "$i"
    nbqa black "$i"
done
for i in *.py; do
    [ -f "$i" ] || break
    echo "$i"
    black "$i"
done

