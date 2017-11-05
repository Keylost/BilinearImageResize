#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

int* resizeBilinear_gpu(int* pixels, int w, int h, int w2, int h2);