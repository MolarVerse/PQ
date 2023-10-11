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
        xx=$(echo "$x" | bc -l)
        xy=$(echo "c($gamma * $pi / 180) * $y" | bc -l)
        xz=$(echo "c($beta * $pi / 180) * $z" | bc -l)
        yy=$(echo "$y*s($gamma * $pi / 180)" | bc -l)
        yz=$(echo "(c($alpha * $pi / 180) - c($beta * $pi / 180) * c($gamma * $pi / 180)) * $z / s($gamma * $pi / 180)" | bc -l)
        zz=$(echo "sqrt($z*$z - $xz*$xz - $yz*$yz)" | bc -l)

        echo "X" $(echo " 0.5 * $xx + 0.5 * $xy + 0.5 * $xz" | bc -l) $(echo " 0.5 * $yy + 0.5 * $yz" | bc -l) $(echo " 0.5 * $zz" | bc -l)
        echo "X" $(echo " 0.5 * $xx + 0.5 * $xy - 0.5 * $xz" | bc -l) $(echo " 0.5 * $yy - 0.5 * $yz" | bc -l) $(echo "-0.5 * $zz" | bc -l)
        echo "X" $(echo " 0.5 * $xx - 0.5 * $xy + 0.5 * $xz" | bc -l) $(echo "-0.5 * $yy + 0.5 * $yz" | bc -l) $(echo " 0.5 * $zz" | bc -l)
        echo "X" $(echo " 0.5 * $xx - 0.5 * $xy - 0.5 * $xz" | bc -l) $(echo "-0.5 * $yy - 0.5 * $yz" | bc -l) $(echo "-0.5 * $zz" | bc -l)
        echo "X" $(echo "-0.5 * $xx + 0.5 * $xy + 0.5 * $xz" | bc -l) $(echo " 0.5 * $yy + 0.5 * $yz" | bc -l) $(echo " 0.5 * $zz" | bc -l)
        echo "X" $(echo "-0.5 * $xx + 0.5 * $xy - 0.5 * $xz" | bc -l) $(echo " 0.5 * $yy - 0.5 * $yz" | bc -l) $(echo "-0.5 * $zz" | bc -l)
        echo "X" $(echo "-0.5 * $xx - 0.5 * $xy + 0.5 * $xz" | bc -l) $(echo "-0.5 * $yy + 0.5 * $yz" | bc -l) $(echo " 0.5 * $zz" | bc -l)
        echo "X" $(echo "-0.5 * $xx - 0.5 * $xy - 0.5 * $xz" | bc -l) $(echo "-0.5 * $yy - 0.5 * $yz" | bc -l) $(echo "-0.5 * $zz" | bc -l)

    fi
    line_number=$((line_number + 1))
done <<<"$combined_files"
