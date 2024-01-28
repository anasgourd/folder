#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <stdio.h>

#include <stdlib.h>

#include <time.h>




__global__ void updateIsingModelKernel(int* currentGrid, int* newGrid, int n, int thread_block_size) {

    int block_i = blockIdx.y * blockDim.y;

    int block_j = blockIdx.x * blockDim.x;



    int thread_i = threadIdx.y;

    int thread_j = threadIdx.x;



    for (int i = 0; i < thread_block_size; ++i) {

        for (int j = 0; j < thread_block_size; ++j) {

            int global_i = block_i + thread_i * thread_block_size + i;

            int global_j = block_j + thread_j * thread_block_size + j;



            if (global_i < n && global_j < n) {

                int i_prev = (global_i - 1 + n) % n;

                int j_prev = (global_j - 1 + n) % n;

                int i_next = (global_i + 1) % n;

                int j_next = (global_j + 1) % n;



                int sum = currentGrid[i_prev * n + global_j] +

                    currentGrid[global_i * n + j_prev] +

                    currentGrid[global_i * n + global_j] +

                    currentGrid[i_next * n + global_j] +

                    currentGrid[global_i * n + j_next];



                newGrid[global_i * n + global_j] = (sum > 0) ? 1 : -1;

            }

        }

    }

}



void initializeIsingModelFromFile(int* grid, int n, const char* filename) {

    FILE* file = fopen(filename, "r");

    if (file == NULL) {

        perror("Error opening file");

        exit(EXIT_FAILURE);

    }



    for (int i = 0; i < n * n; ++i) {

        if (fscanf(file, "%d", &grid[i]) != 1) {

            perror("Error reading from file");

            fclose(file);

            exit(EXIT_FAILURE);

        }

    }



    fclose(file);

}



void writeIsingModelToFile(int* grid, int n, const char* filename) {

    FILE* file = fopen(filename, "w");

    if (file == NULL) {

        perror("Error opening file");

        exit(EXIT_FAILURE);

    }



    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            fprintf(file, "%2d ", grid[i * n + j]);

        }

        fprintf(file, "\n");

    }



    fclose(file);

}



void printIsingModel(int* grid, int n) {

    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            printf("%2d ", grid[i * n + j]);

        }

        printf("\n");

    }

    printf("\n");

}

int main(int argc, char* argv[]) {

    if (argc != 5) {

        fprintf(stderr, "Usage: %s <input_filename> <output_filename> <num_of_iterations>\n", argv[0]);

        exit(1);

    }
  
      const char* inputFilename = argv[1];

    const char* outputFilename = argv[2];

    int n = atoi(argv[3]);

    int k = atoi(argv[4]);
    int* currentGrid, * newGrid;

    cudaMallocManaged(&currentGrid, n * n * sizeof(int));

    cudaMallocManaged(&newGrid, n * n * sizeof(int));
     clock_t start_time = clock();

    int thread_block_size = n/32;  // Change this to the desired thread block size

   int block_size = 32;
	    // Initialize the Ising model from a file

    initializeIsingModelFromFile(currentGrid, n, inputFilename);



    // Perform k iterations of the Ising model using CUDA

    for (int iteration = 0; iteration < k; ++iteration) {

        // Print the current state (optional)

        printf("Iteration %d:\n", iteration);



        dim3 blockDim(block_size, block_size);

        int grid_size_x = (n + blockDim.x - 1) / blockDim.x;

        int grid_size_y = (n + blockDim.y - 1) / blockDim.y;



        // Limit the grid size if needed to avoid too many threads

        int max_grid_size = 65535;  // Maximum allowed grid size for a single dimension

        grid_size_x = (grid_size_x > max_grid_size) ? max_grid_size : grid_size_x;

        grid_size_y = (grid_size_y > max_grid_size) ? max_grid_size : grid_size_y;



        dim3 gridDim(grid_size_x, grid_size_y);



        updateIsingModelKernel << <gridDim, blockDim >> > (currentGrid, newGrid, n, thread_block_size);



        // Synchronize to ensure the kernel completes before swapping pointers

        cudaDeviceSynchronize();



        // Swap the pointers for the next iteration

        int* temp = currentGrid;

        currentGrid = newGrid;

        newGrid = temp;

    }



    // Print the final state after k iterations

    printf("Final state:\n");

    //printIsingModel(currentGrid, n);

     clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Total time for n=%d, k=%d: %f seconds\n", n, k, elapsed_time);

    writeIsingModelToFile(currentGrid, n, outputFilename);



    // Free allocated memory

    cudaFree(currentGrid);

    cudaFree(newGrid);



    return 0;

}


