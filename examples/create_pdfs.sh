#!/bin/bash

# Check if Graphviz is installed
if ! command -v dot &> /dev/null
then
    echo "Error: Graphviz (dot) is not installed. Install it first."
    exit 1
fi

# Loop through all .dot files in the current directory
for file in *.dot; do
    # Check if there are any .dot files
    if [ ! -e "$file" ]; then
        echo "No .dot files found."
        exit 1
    fi
    
    # Extract the filename without extension
    filename="${file%.dot}"
    
    # Convert to PDF
    dot -Tpdf "$file" -o "$filename.pdf"
    
    echo "Converted $file -> $filename.pdf"
done
