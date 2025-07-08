#!/bin/bash

SEARCH_DIR="." 

echo "Replacing 'segmented' with 'seg'..."

find "$SEARCH_DIR" -depth -name "*segmented*" -print0 | while IFS= read -r -d $'\0' file; do
    dir=$(dirname "$file")
    filename=$(basename "$file")
    
    new_filename=$(echo "$filename" | sed 's/segmented/seg/g')
    
    new_file="$dir/$new_filename"
    
    if [ "$file" != "$new_file" ]; then
        mv "$file" "$new_file"
        echo "Renamed: $filename -> $new_filename"
    else
        echo "Skipped: $filename (no change needed or already renamed)"
    fi
done

echo "Renaming process complete."
