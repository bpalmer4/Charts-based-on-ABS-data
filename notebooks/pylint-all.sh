for i in *.ipynb; do
    [ -f "$i" ] || break
    echo "$i"
    nbqa pylint --disable=C0114,CO103 "$i"
done
for i in *.py; do
    [ -f "$i" ] || break
    echo "$i"
    pylint "$i"
done

