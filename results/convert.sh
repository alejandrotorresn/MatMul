#/bin/bash

total_files=`ls *.svg | wc -l`
echo $total_files

filenames=`ls ./*.svg`
i=1
for file in $filenames
do
    filename_with_ext=$(basename "$file")
    filename="${filename_with_ext%.*}"
    extension="${filename_with_ext##*.}"
    cairosvg $filename_with_ext -f eps -o "${filename}.eps"
    echo "File: ${i} of ${total_files}"
    i=$((i+1))
    rm $filename_with_ext
done
