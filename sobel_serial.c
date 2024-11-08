#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <math.h>
#include <time.h> // Incluir la librería para medir el tiempo

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

// Aplicar el filtro Sobel
void sobel_filter(unsigned char *input, unsigned char *output, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
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
    if (argc != 3) {
        printf("Uso: %s <imagen_entrada.jpg> <imagen_salida.jpg>\n", argv[0]);
        return 1;
    }

    int width, height;
    unsigned char *data = read_JPEG_file(argv[1], &width, &height);
    unsigned char *output = (unsigned char*)calloc(width * height, sizeof(unsigned char));

    // Convertir a escala de grises si es necesario
    convert_to_grayscale(data, width, height, 3);

    // Medir el tiempo de ejecución del filtro Sobel
    clock_t start_time = clock();  // Tiempo de inicio

    // Aplicar filtro Sobel
    sobel_filter(data, output, width, height);

    clock_t end_time = clock();  // Tiempo de fin

    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;  // Calcular el tiempo en segundos
    printf("Tiempo de procesamiento del filtro Sobel: %f segundos.\n", time_taken);

    // Guardar imagen filtrada
    write_JPEG_file(argv[2], output, width, height);

    printf("Filtro Sobel aplicado exitosamente.\n");

    free(data);
    free(output);
    return 0;
}
