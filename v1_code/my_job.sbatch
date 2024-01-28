#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu

#SBATCH --time=0:10:00



#SBATCH --job-name=YourJobName

#SBATCH --gres=gpu:1  # Request 1 GPU



module load gcc/9.2.0

module load cuda/11.1.0
nvcc v1code.cu -o v1code_executable_n${1}_k${2} -std=c++11  -O3 




./v1code_executable_n${1}_k${2}  "input_n${1}_k${2}.txt" "output_n${1}_k${2}.txt" "$1" "$2" > "output_n${1}_k${2}.log" 2>&1
:'nvidia-smi --loop=300 > nvidia-smi.log &



function terminate_monitor_gpu_status()

{

  nvidia_smi_pid=$(pgrep -u $USER nvidia-smi)

  kill $nvidia_smi_pid

}



trap 'terminate_monitor_gpu_status' KILL TERM EXIT
'
