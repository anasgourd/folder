# parallel-ising-cuda

This repository contains a **parallel GPU implementation** of the **2D Ising model**. It demonstrates sequential CPU execution as well as multiple CUDA parallel versions with increasing optimization.It was developed for the Parallel and Distributed Systems course at ECE AUTH (2023–2024).

## Contents

- **`v0_sequential/`** – Sequential CPU implementation (baseline).  
- **`v1_cuda/`** – Naive CUDA kernel: 1 thread per cell using global memory.  
- **`v2_cuda/`** – CUDA kernel with thread coarsening / block tiling(each thread processes more than one cell).  
- **`v3_cuda/`** – CUDA kernel with shared memory & halo cells (each block loads its portion of the grid plus surrounding neighbors into fast shared memory, where threads perform all computations).
- **`create_file.c` / `generate_inputs.sh`** – Generate input grids for the simulations.  

## Requirements

- NVIDIA GPU (tested on GTX 1050)  
- CUDA Toolkit 11.x or higher  
- GCC compiler (for input generation)


Using `my_job.sbatch` together with `submit_jobs.sh`, the execution of multiple simulations with different grid sizes (`n`) and iteration counts (`k`) was automated, producing the results shown in [RESULTS](https://github.com/anasgourd/folder/tree/main/RESULTS), which demonstrate the speedup achieved with GPU parallelization and optimized CUDA kernels.
