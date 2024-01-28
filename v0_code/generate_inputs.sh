#!/bin/bash

module load gcc/9.2.0


gcc crete_file.c -o create_file_executable -std=c99

for n in 1120 1600 2080 2560 3040; do
 for k in 10 100 500 1000;do

        ./create_file_executable "input_n${n}_k${k}.txt" "$n"

    done

done


