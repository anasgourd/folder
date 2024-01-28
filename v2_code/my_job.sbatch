#!/bin/bash



#SBATCH --nodes=1

#SBATCH --partition=gpu



#SBATCH --time=0:30:00


#SBATCH --job-name=YourJobName



#SBATCH --gres=gpu:1  # Request 1 GPU







module load gcc/9.2.0



module load cuda/11.1.0

nvcc v2code.cu -o v2code_executable_n${1}_k${2} -std=c++11  -O3 








./v2code_executable_n${1}_k${2}  "input_n${1}_k${2}.txt" "output_n${1}_k${2}.txt" "$1" "$2" > "output_n${1}_k${2}.log" 2>&1
