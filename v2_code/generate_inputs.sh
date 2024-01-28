#!/bin/bash



module load gcc/9.2.0





gcc crete_file.c -o create_file_executable -std=c99



for n in  4800 8000 12800 16000; do
       for k in 1 10 20 50;do

        ./create_file_executable "input_n${n}_k${k}.txt" "$n"



    done



done
