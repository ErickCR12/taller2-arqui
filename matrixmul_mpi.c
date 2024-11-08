#include <stdio.h>
#include <mpi.h>

#define N 4  // Tamaño de la matriz (4x4)

// Función principal
int main(int argc, char **argv) {
    int rank, num_procs, i, j, k;
    int A[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int B[N][N] = {
        {16, 15, 14, 13},
        {12, 11, 10, 9},
        {8, 7, 6, 5},
        {4, 3, 2, 1}
    };
    int C[N][N];  // Matriz resultado
    int local_result[N][N] = {0};  // Submatriz de resultado para cada proceso

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Cada proceso se encarga de una fila de la matriz C
    int rows_per_proc = N / num_procs;  // Filas calculadas por cada proceso
    int start_row = rank * rows_per_proc;  // Fila inicial para este proceso
    int end_row = (rank + 1) * rows_per_proc;  // Fila final para este proceso

    // Calcular la submatriz correspondiente de C
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < N; j++) {
            local_result[i][j] = 0;
            for (k = 0; k < N; k++) {
                local_result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Imprimir la submatriz calculada por cada proceso
    printf("Proceso %d calculó las siguientes filas de la matriz resultado:\n", rank);
    for (i = start_row; i < end_row; i++) {
        printf("Fila %d: ", i);
        for (j = 0; j < N; j++) {
            printf("%d ", local_result[i][j]);
        }
        printf("\n");
    }

    // Recopilar los resultados en el proceso raíz
    MPI_Gather(local_result[start_row], rows_per_proc * N, MPI_INT,
               C, rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    // Imprimir la matriz resultado completa en el proceso raíz
    if (rank == 0) {
        printf("\nMatriz Resultado:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
