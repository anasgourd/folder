#!/bin/bash



#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --time=7:10:00

#SBATCH --mem=4G

#SBATCH --job-name=YourJobName






module load gcc/9.2.0

gcc vocode.c -o vocode_executable_n${1}_k${2} -std=c99

./vocode_executable "input_n${1}_k${2}.txt" "output_n${1}_k${2}.txt" "$1" "$2" > "output_n${1}_k${2}.log" 2>&1

