#include <cuda_runtime.h>



#include <device_launch_parameters.h>



#include <stdio.h>



#include <stdlib.h>



#include <time.h>







#define THREADS_PER_BLOCK_X 32



#define THREADS_PER_BLOCK_Y 32







__global__ void updateIsingModelKernel(int* currentGrid, int* newGrid, int n) {



    int i = blockIdx.y * blockDim.y + threadIdx.y;



    int j = blockIdx.x * blockDim.x + threadIdx.x;







    if (i < n && j < n) {



        int i_prev = (i - 1 + n) % n;



        int j_prev = (j - 1 + n) % n;



        int i_next = (i + 1) % n;



        int j_next = (j + 1) % n;







        int sum = currentGrid[i_prev * n + j] +



                  currentGrid[i * n + j_prev] +



                  currentGrid[i * n + j] +



                  currentGrid[i_next * n + j] +



                  currentGrid[i * n + j_next];







        newGrid[i * n + j] = (sum > 0) ? 1 : -1;



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







    // Allocate managed memory for two grids (current and new state)



    int* currentGrid, * newGrid;



    cudaMallocManaged(&currentGrid, n * n * sizeof(int));



    cudaMallocManaged(&newGrid, n * n * sizeof(int));







    clock_t start_time = clock();







    // Initialize the Ising model from a file



    initializeIsingModelFromFile(currentGrid, n, inputFilename);







    // Perform k iterations of the Ising model using CUDA



    for (int iteration = 0; iteration < k; ++iteration) {



        // Print the current state (optional)



        printf("Iteration %d:\n", iteration);







        // Launch the CUDA kernel for the Ising model update



        dim3 blockDim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);



        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);







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



    //printIsingModel(currentGrid, n);







    // Write the final state to an output file



    writeIsingModelToFile(currentGrid, n, outputFilename);







    // Free allocated memory



    cudaFree(currentGrid);



    cudaFree(newGrid);







    clock_t end_time = clock();



    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;



    printf("Total time for n=%d, k=%d: %f seconds\n", n, k, elapsed_time);







    return 0;



}


