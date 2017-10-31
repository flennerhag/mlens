for f in ../*; do
    [ "$f" == "../source" ] && continue
    rm -r $f
done
