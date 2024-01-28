#include <stdio.h>

#include <stdlib.h>

#include <time.h>



void generateAndWriteToFile(const char* filename, int n) {

    FILE* file = fopen(filename, "w");



    if (file == NULL) {

        fprintf(stderr, "Error opening file: %s\n", filename);

        exit(1);

    }



    srand(time(NULL));



    



    for (int i = 0; i < n; ++i) {

        for (int j = 0; j < n; ++j) {

            int value = (rand() % 2) * 2 - 1; // Set to either +1 or -1

            fprintf(file, "%d ", value);

        }

        fprintf(file, "\n");

    }



    fclose(file);

}



void readFromFile(const char* filename, int* array, int* n) {

    FILE* file = fopen(filename, "r");



    if (file == NULL) {

        fprintf(stderr, "Error opening file: %s\n", filename);

        exit(1);

    }



    fscanf(file, "%d", n); 



    for (int i = 0; i < *n; ++i) {

        for (int j = 0; j < *n; ++j) {

            fscanf(file, "%d", &array[i * (*n) + j]);

        }

    }



    fclose(file);

}


int main(int argc, char* argv[]) {

    if (argc != 3) {

        fprintf(stderr, "Usage: %s <filename> <size_of_array>\n", argv[0]);

        exit(1);

    }



    const char* filename = argv[1];

    int n = atoi(argv[2]);



    generateAndWriteToFile(filename, n);



    return 0;

}




