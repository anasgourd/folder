

#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#define THREADS_PER_BLOCK_X 32

#define THREADS_PER_BLOCK_Y 32






__global__ void updateIsingModelKernel(int* currentGrid, int* newGrid, int n) {



     

    __shared__ int sharedGrid[THREADS_PER_BLOCK_Y + 2][THREADS_PER_BLOCK_X + 2];



     

    int block_i = blockIdx.y * blockDim.y;

    int block_j = blockIdx.x * blockDim.x;



    

    int thread_i = threadIdx.y;

    int thread_j = threadIdx.x;



     

    int shared_i = thread_i + 1;  // Add 1 to skip the top row of shared memory

    int shared_j = thread_j + 1;  // Add 1 to skip the leftmost column of shared memory



     

    int i = block_i + thread_i;

    int j = block_j + thread_j;



     

    int flat_index_global = i * n + j;



    // Ensure all threads have loaded data into shared memory

    __syncthreads();



     

    if (i < n && j < n) {

        sharedGrid[shared_i][shared_j] = currentGrid[flat_index_global];

    }



     

    if (thread_j == 0) {

        // Leftmost column

        sharedGrid[shared_i][0] = currentGrid[i * n + (j - 1 + n) % n];

    }

    else if (thread_j == blockDim.x - 1) {

        // Rightmost column

        sharedGrid[shared_i][blockDim.x + 1] = currentGrid[i * n + (j + 1) % n];

    }



     

    if (thread_i == 0) {

        

        sharedGrid[0][shared_j] = currentGrid[(i - 1 + n) % n * n + j];

    }

    else if (thread_i == blockDim.y - 1) {

         

        sharedGrid[blockDim.y + 1][shared_j] = currentGrid[(i + 1) % n * n + j];

    }



    

    // Synchronize to make sure all threads have copied data before proceeding

    __syncthreads();



    // Calculate indices for neighbors in the shared memory array with periodic boundary conditions

    int i_prev = shared_i - 1;

    int j_prev = shared_j - 1;

    int i_next = shared_i + 1;

    int j_next = shared_j + 1;



    // Accessing neighbors from shared memory

    int sum = sharedGrid[i_prev][shared_j] +     // Accessing the element above in shared memory

        sharedGrid[shared_i][j_prev] +     // Accessing the element to the left in shared memory

        sharedGrid[shared_i][shared_j] +  // Accessing the current element in shared memory

        sharedGrid[i_next][shared_j] +     // Accessing the element below in shared memory

        sharedGrid[shared_i][j_next];      // Accessing the element to the right in shared memory



    // Update Ising model

    if (i < n && j < n) {

        newGrid[flat_index_global] = (sum > 0) ? 1 : -1;

    }



    // Synchronize to make sure all threads have completed their work

    __syncthreads();

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

    



    // Allocate managed memory for two grids (current and new state)

    int *currentGrid, *newGrid;



   

    

    if (argc != 5) {



        fprintf(stderr, "Usage: %s <input_filename> <output_filename> <num_of_iterations>\n", argv[0]);



        exit(1);



    }

  

    const char* inputFilename = argv[1];



    const char* outputFilename = argv[2];



    int n = atoi(argv[3]);



    int k = atoi(argv[4]);

    clock_t start_time = clock();

    cudaMallocManaged(&currentGrid, n * n * sizeof(int));

    cudaMallocManaged(&newGrid, n * n * sizeof(int));

    

    initializeIsingModelFromFile(currentGrid, n, inputFilename);



    // Calculate an appropriate block size based on the Ising model size

    int max_block_size = 32;  // Set a maximum block size

    int block_size = (n < max_block_size) ? n : max_block_size;



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



        updateIsingModelKernel << <gridDim, blockDim >> > (currentGrid, newGrid, n);



        // Synchronize to ensure the kernel completes before swapping pointers

        cudaDeviceSynchronize();



        // Swap the pointers for the next iteration

        int* temp = currentGrid;

        currentGrid = newGrid;

        newGrid = temp;

    }



    // Print the final state after k iterations

    printf("Final state:\n");

    // printIsingModel(currentGrid, n);

    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;



    printf("Total time for n=%d, k=%d: %f seconds\n", n, k, elapsed_time);

    

     writeIsingModelToFile(currentGrid, n, outputFilename);



    // Free allocated memory

    cudaFree(currentGrid);

    cudaFree(newGrid);



    return 0;

}


