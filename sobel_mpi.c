#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <math.h>
#include <mpi.h>  // Incluir la librería de MPI

// Definir los kernels de Sobel
int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int Gy[3][3] = {
    {-1, -2, -1},
    {0,  0,  0},
    {1,  2,  1}
};

// Función para leer una imagen JPEG
unsigned char* read_JPEG_file(const char *filename, int *width, int *height) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *infile = fopen(filename, "rb");
    if (infile == NULL) {
        fprintf(stderr, "No se pudo abrir el archivo %s\n", filename);
        exit(1);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    int channels = cinfo.output_components;

    unsigned char *data = (unsigned char*)malloc(*width * *height * channels);
    unsigned char *rowptr = data;

    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
        rowptr += *width * channels;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return data;
}

// Función para guardar una imagen JPEG
void write_JPEG_file(const char *filename, unsigned char *image, int width, int height) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile = fopen(filename, "wb");
    if (outfile == NULL) {
        fprintf(stderr, "No se pudo crear el archivo %s\n", filename);
        exit(1);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1; // Imagen en escala de grises
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    unsigned char *rowptr = image;
    while (cinfo.next_scanline < cinfo.image_height) {
        jpeg_write_scanlines(&cinfo, &rowptr, 1);
        rowptr += width;
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

// Convertir imagen a escala de grises (si no lo está)
void convert_to_grayscale(unsigned char *data, int width, int height, int channels) {
    if (channels == 1) return;

    for (int i = 0; i < width * height; i++) {
        int gray = 0.299 * data[i * channels] + 0.587 * data[i * channels + 1] + 0.114 * data[i * channels + 2];
        data[i] = (unsigned char)gray;
    }
}

// Función para aplicar el filtro Sobel
void sobel_filter(unsigned char *input, unsigned char *output, int width, int height, int start_row, int end_row) {
    for (int y = start_row; y < end_row; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0, sumY = 0;

            // Aplicar los kernels Gx y Gy
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = input[(y + i) * width + (x + j)];
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }

            int gradient = abs(sumX) + abs(sumY);
            if (gradient > 255) gradient = 255;
            output[y * width + x] = (unsigned char)gradient;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int width, height;
    unsigned char *data, *output, *local_data;
    double start_time, end_time;

    // Inicialización de MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Obtiene el rank del proceso
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Obtiene el número de procesos

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <imagen_entrada.jpg> <imagen_salida.jpg>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Medir el tiempo total de ejecución
    if (rank == 0) {
        start_time = MPI_Wtime();  // Inicio del tiempo en el proceso maestro
    }

    if (rank == 0) {
        // Leer la imagen en el proceso maestro
        data = read_JPEG_file(argv[1], &width, &height);
        output = (unsigned char*)calloc(width * height, sizeof(unsigned char));

        // Convertir a escala de grises si es necesario
        convert_to_grayscale(data, width, height, 3);
    }

    // Distribuir el tamaño de la imagen entre los procesos
    int rows_per_process = height / size;
    int remainder = height % size;

    // Determinar el rango de filas que cada proceso debe procesar
    int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

    // Asignar memoria para los datos locales
    local_data = (unsigned char*)malloc(width * (end_row - start_row) * sizeof(unsigned char));
    
    // Enviar y recibir bloques de la imagen entre los procesos
    MPI_Scatter(data + start_row * width, width * (end_row - start_row), MPI_UNSIGNED_CHAR, 
                local_data, width * (end_row - start_row), MPI_UNSIGNED_CHAR, 
                0, MPI_COMM_WORLD);

    // Aplicar el filtro Sobel en el bloque local
    unsigned char *local_output = (unsigned char*)malloc(width * (end_row - start_row) * sizeof(unsigned char));
    sobel_filter(local_data, local_output, width, height, start_row, end_row);

    // Recopilar los resultados del filtro Sobel
    MPI_Gather(local_output, width * (end_row - start_row), MPI_UNSIGNED_CHAR, 
               output, width * (end_row - start_row), MPI_UNSIGNED_CHAR, 
               0, MPI_COMM_WORLD);

    // El proceso maestro guarda la imagen resultante
    if (rank == 0) {
        write_JPEG_file(argv[2], output, width, height);
        end_time = MPI_Wtime();  // Fin del tiempo en el proceso maestro
        printf("Filtro Sobel aplicado exitosamente.\n");
        printf("Tiempo total de procesamiento: %f segundos.\n", end_time - start_time);
    }

    // Liberar la memoria
    free(local_data);
    free(local_output);
    if (rank == 0) {
        free(data);
        free(output);
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}
