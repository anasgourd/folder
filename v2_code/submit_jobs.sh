#!/bin/bash





for n in  1600 2080 2560 3040;do



     for k in 10 100 500 1000;do

          sbatch my_job.sbatch  "$n" "$k"

    done



done




