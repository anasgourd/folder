#!/bin/bash



#for n in  4800 8000 12800 16000; do







 #    for k in 1 10 20 50;do



for n in 1120 1600 2080 2560 3040; do 



for k in 10 100 500 1000;do



          sbatch my_job.sbatch  "$n" "$k"



    done




done


