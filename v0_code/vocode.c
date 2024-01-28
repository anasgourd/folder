#include <stdio.h>

#include <stdlib.h>
#include <time.h>


void initializeIsingModelFromFile(const char* filename, int** grid, int n) {

    FILE* file = fopen(filename, "r");



    if (file == NULL) {

        fprintf(stderr, "Error opening file: %s\n", filename);

        exit(1);

    }



    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            fscanf(file, "%d", &grid[i][j]);

        }

    }



    fclose(file);

}



void printIsingModelToFile(const char* filename, int** grid, int n) {

    FILE* file = fopen(filename, "w");



    if (file == NULL) {

        fprintf(stderr, "Error opening file: %s\n", filename);

        exit(1);

    }



    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            fprintf(file, "%2d ", grid[i][j]);

        }

        fprintf(file, "\n");

    }



    fclose(file);

}



void updateIsingModel(int** currentGrid, int** newGrid, int n) {

    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            

            int i_prev = (i - 1 + n) % n;

            int j_prev = (j - 1 + n) % n;

            int i_next = (i + 1) % n;

            int j_next = (j + 1) % n;



            // Compute the sum of spins of the four neighbors and itself

            int sum = currentGrid[i_prev][j] + currentGrid[i][j_prev] +

                      currentGrid[i][j] + currentGrid[i_next][j] +

                      currentGrid[i][j_next];


            newGrid[i][j] = (sum > 0) ? 1 : -1;

        }

    }

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


     

    int** currentGrid = (int**)malloc(n * sizeof(int*));

    int** newGrid = (int**)malloc(n * sizeof(int*));

    for (int i = 0; i < n; ++i) {

        currentGrid[i] = (int*)malloc(n * sizeof(int));

        newGrid[i] = (int*)malloc(n * sizeof(int));

    }



    clock_t start_time = clock();

    initializeIsingModelFromFile(inputFilename, currentGrid, n);



    // Perform k iterations of the Ising model

    for (int iteration = 0; iteration < k; ++iteration) {

        // Print the current state (optional)

        printf("Iteration %d:\n", iteration);



        

        updateIsingModel(currentGrid, newGrid, n);



        // Swap the pointers for the next iteration

        int** temp = currentGrid;

        currentGrid = newGrid;

        newGrid = temp;

    }



  clock_t end_time = clock();
  double elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
  printf("Total time for n=%d, k=%d: %f seconds\n", n, k, elapsed_time); 

    printIsingModelToFile(outputFilename, currentGrid, n);

    

    for (int i = 0; i < n; ++i) {

        free(currentGrid[i]);

        free(newGrid[i]);

    }


    free(currentGrid);

    free(newGrid);



    return 0;

}


