#!/bin/bash

vmd_flag="false"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
    --vmd)
        vmd_flag="true"
        shift
        ;;
    *)
        combined_files="$combined_files $(cat $1)"
        shift
        ;;
    esac
done

combined_files=$(echo "$combined_files" | grep "^[0-9]")

line_number=1
pi=3.141592653589793238462643383279502884197169399375105820974944592307816406286

while read -r line; do
    read -r nAtoms x y z alpha beta gamma <<<$(echo "$line")
    if [[ "$vmd_flag" == "false" ]]; then
        echo $line_number $x $y $z
    else
        echo "Box" $line_number $x $y $z
        echo ""
        echo "X" $((-x / 2)) 0 0
        echo "X" $((+x / 2)) 0 0
        echo "X" $(echo "-0.5 * s($gamma * $pi / 180) * $y" | bc -l) $(echo "0.5 * c($gamma * $pi / 180) * $y" | bc -l) 0

    fi
    line_number=$((line_number + 1))
done <<<"$combined_files"
