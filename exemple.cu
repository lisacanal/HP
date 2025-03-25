// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION

// #include "stb_image.h"
// #include "stb_image_write.h"
// #include <stdlib.h>
// #include <math.h>
// #include <stdio.h>

// // Gaussian function
// double gaussian(double x, double sigma) {
//     return exp(-(x * x) / (2.0 * sigma * sigma));
// }

// // Function to precompute spatial Gaussian weights
// void compute_spatial_weights(double *spatial_weights, int d, double sigma_space) {
//     int radius = d / 2;
//     for (int i = 0; i < d; i++) {
//         for (int j = 0; j < d; j++) {
//             int x = i - radius, y = j - radius;
//             spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
//         }
//     }
// }

// // Manual bilateral filter
// void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
//     int radius = d / 2;

//     // Precompute spatial Gaussian weights
//     double *spatial_weights = (double *)malloc(d * d * sizeof(double));
//     if (!spatial_weights) {
//         printf("Memory allocation for spatial weights failed!\n");
//         return;
//     }
//     compute_spatial_weights(spatial_weights, d, sigma_space);

//     // Process image
//     for (int y = radius; y < height - radius; y++) {
//         for (int x = radius; x < width - radius; x++) {
//             double weight_sum[3] = {0.0, 0.0, 0.0};
//             double filtered_value[3] = {0.0, 0.0, 0.0};

//             // Get center pixel pointer
//             unsigned char *center_pixel = src + (y * width + x) * channels;

//             // Iterate over local window
//             for (int i = 0; i < d; i++) {
//                 for (int j = 0; j < d; j++) {
//                     int nx = x + j - radius;
//                     int ny = y + i - radius;

//                     // Bounds check to ensure we're within the image
//                     if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
//                         continue;
//                     }

//                     // Get neighbor pixel pointer
//                     unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

//                     for (int c = 0; c < channels; c++) {
//                         // Compute range weight
//                         double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
//                         double weight = spatial_weights[i * d + j] * range_weight;

//                         // Accumulate weighted sum
//                         filtered_value[c] += neighbor_pixel[c] * weight;
//                         weight_sum[c] += weight;
//                     }
//                 }
//             }

//             // Normalize and store result
//             unsigned char *output_pixel = dst + (y * width + x) * channels;
//             for (int c = 0; c < channels; c++) {
//                 output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Avoid division by zero
//             }
//         }
//     }

//     free(spatial_weights);
// }

// // Main function
// int main(int argc, char *argv[]) {
//     if (argc < 3) {
//         printf("Usage: %s <input_image> <output_image>\n", argv[0]);
//         return 1;
//     }

//     int width, height, channels;
//     unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
//     if (!image) {
//         printf("Error loading image!\n");
//         return 1;
//     }

//     // Ensure that image is not too small for bilateral filter (at least radius of d/2 around edges)
//     if (width <= 5 || height <= 5) {
//         printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
//         stbi_image_free(image);
//         return 1;
//     }

//     // Allocate memory for output image
//     unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
//     if (!filtered_image) {
//         printf("Memory allocation for filtered image failed!\n");
//         stbi_image_free(image);
//         return 1;
//     }
    
//     // Apply the bilateral filter
//     bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

//     // Save the output image
//     if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
//         printf("Error saving the image!\n");
//         free(filtered_image);
//         stbi_image_free(image);
//         return 1;
//     }

//     // Free memory
//     stbi_image_free(image);
//     free(filtered_image);

//     printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
//     return 0;
// }

// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION

// #include "stb_image.h"
// #include "stb_image_write.h"
// #include <stdlib.h>
// #include <math.h>
// #include <stdio.h>
// #include <cuda_runtime.h>
// #include <math_constants.h>

// // Gaussian function
// __device__ double gaussian(double x, double sigma) {
//     return exp(-(x * x) / (2.0 * sigma * sigma));
// }

// // CUDA kernel to compute spatial weights
// __global__ void compute_spatial_weights_kernel(double *spatial_weights, int d, double sigma_space, int radius) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < d && j < d) {
//         int x = i - radius, y = j - radius;
//         spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
//     }
// }

// // CUDA kernel to apply the bilateral filter
// __global__ void bilateral_filter_kernel(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space, double *spatial_weights) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int radius = d / 2;

//     if (x < radius || x >= width - radius || y < radius || y >= height - radius) {
//         return;  // Ignore borders
//     }

//     double weight_sum[3] = {0.0, 0.0, 0.0};
//     double filtered_value[3] = {0.0, 0.0, 0.0};

//     unsigned char *center_pixel = src + (y * width + x) * channels;

//     for (int i = -radius; i <= radius; i++) {
//         for (int j = -radius; j <= radius; j++) {
//             int nx = x + j;
//             int ny = y + i;
            
//             if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
//                 continue; // Skip out-of-bounds pixels
//             }

//             unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

//             for (int c = 0; c < channels; c++) {
//                 double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
//                 double weight = spatial_weights[(i + radius) * d + (j + radius)] * range_weight;
//                 filtered_value[c] += neighbor_pixel[c] * weight;
//                 weight_sum[c] += weight;
//             }
//         }
//     }

//     unsigned char *output_pixel = dst + (y * width + x) * channels;
//     for (int c = 0; c < channels; c++) {
//         output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
//     }
// }

// // Function to apply the bilateral filter using CUDA
// void bilateral_filter_cuda(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
//     int img_size = width * height * channels;
//     unsigned char *d_src, *d_dst;
//     double *d_spatial_weights;
    
//     cudaMalloc((void**)&d_src, img_size);
//     cudaMalloc((void**)&d_dst, img_size);
//     cudaMalloc((void**)&d_spatial_weights, d * d * sizeof(double));
//     cudaMemcpy(d_src, src, img_size, cudaMemcpyHostToDevice);
    
//     dim3 threadsPerBlock(16, 16);
//     dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
//                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     compute_spatial_weights_kernel<<<numBlocks, threadsPerBlock>>>(d_spatial_weights, d, sigma_space, d / 2);
//     bilateral_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height, channels, d, sigma_color, sigma_space, d_spatial_weights);
//     cudaMemcpy(dst, d_dst, img_size, cudaMemcpyDeviceToHost);
    
//     cudaFree(d_src);
//     cudaFree(d_dst);
//     cudaFree(d_spatial_weights);
// }

// // Main function
// int main(int argc, char *argv[]) {
//     if (argc < 3) {
//         printf("Usage: %s <input_image> <output_image>\n", argv[0]);
//         return 1;
//     }

//     int width, height, channels;
//     unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
//     if (!image) {
//         printf("Error loading image!\n");
//         return 1;
//     }

//     if (width <= 5 || height <= 5) {
//         printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
//         stbi_image_free(image);
//         return 1;
//     }

//     unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
//     if (!filtered_image) {
//         printf("Memory allocation for filtered image failed!\n");
//         stbi_image_free(image);
//         return 1;
//     }
    
//     bilateral_filter_cuda(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

//     if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
//         printf("Error saving the image!\n");
//         free(filtered_image);
//         stbi_image_free(image);
//         return 1;
//     }

//     stbi_image_free(image);
//     free(filtered_image);

//     printf("Bilateral filtering complete using CUDA. Output saved as %s\n", argv[2]);
//     return 0;
// }

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Fonction gaussienne
__device__ float calcul_gaussien(float distance, float ecart_type) {
    return expf(-(distance * distance) / (2.0f * ecart_type * ecart_type));
}

// Filtre bilatéral en CUDA
__global__ void filtre_bilateral_cuda(unsigned char *entree, unsigned char *sortie, int largeur, int hauteur, int canaux, int taille_fenetre, float sigma_couleur, float sigma_espace) {
    int coord_x = blockIdx.x * blockDim.x + threadIdx.x;
    int coord_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (coord_x >= largeur || coord_y >= hauteur) return;

    int rayon = taille_fenetre / 2;

    float somme_valeurs[3] = {0.0f, 0.0f, 0.0f};
    float somme_poids[3] = {0.0f, 0.0f, 0.0f};

    unsigned char *pixel_central = entree + (coord_y * largeur + coord_x) * canaux;

    for (int dy = -rayon; dy <= rayon; dy++) {
        for (int dx = -rayon; dx <= rayon; dx++) {
            int voisin_x = coord_x + dx;
            int voisin_y = coord_y + dy;

            if (voisin_x >= 0 && voisin_x < largeur && voisin_y >= 0 && voisin_y < hauteur) {
                unsigned char *pixel_voisin = entree + (voisin_y * largeur + voisin_x) * canaux;

                for (int c = 0; c < canaux; c++) {
                    float poids_spatial = calcul_gaussien(sqrtf((float)(dx * dx + dy * dy)), sigma_espace);
                    float poids_couleur = calcul_gaussien(fabsf((float)pixel_voisin[c] - (float)pixel_central[c]), sigma_couleur);
                    float poids_total = poids_spatial * poids_couleur;

                    somme_valeurs[c] += pixel_voisin[c] * poids_total;
                    somme_poids[c] += poids_total;
                }
            }
        }
    }

    unsigned char *pixel_sortie = sortie + (coord_y * largeur + coord_x) * canaux;
    for (int c = 0; c < canaux; c++) {
        pixel_sortie[c] = (unsigned char)(somme_valeurs[c] / (somme_poids[c] + 1e-6f));
    }
}

// Fonction principale
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_entree> <image_sortie>\n", argv[0]);
        return 1;
    }

    int largeur, hauteur, canaux;
    unsigned char *image_entree = stbi_load(argv[1], &largeur, &hauteur, &canaux, 0);
    if (!image_entree) {
        printf("Erreur lors du chargement de l’image !\n");
        return 1;
    }

    unsigned char *image_filtrée = (unsigned char *)malloc(largeur * hauteur * canaux);

    unsigned char *gpu_entree, *gpu_sortie;
    cudaMalloc(&gpu_entree, largeur * hauteur * canaux);
    cudaMalloc(&gpu_sortie, largeur * hauteur * canaux);

    cudaMemcpy(gpu_entree, image_entree, largeur * hauteur * canaux, cudaMemcpyHostToDevice);

    dim3 taille_bloc(16, 16);
    dim3 taille_grille(32, 32);
    //dim3 taille_grille((largeur + taille_bloc.x - 1) / taille_bloc.x, (hauteur + taille_bloc.y - 1) / taille_bloc.y);

    filtre_bilateral_cuda<<<taille_grille, taille_bloc>>>(gpu_entree, gpu_sortie, largeur, hauteur, canaux, 5, 15.0f, 5.0f);
    cudaDeviceSynchronize();
}