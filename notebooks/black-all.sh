micromamba activate 311
for i in *.ipynb; do
    [ -f "$i" ] || break
    black "$i"
done