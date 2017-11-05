#include <stdint.h>
#include <omp.h>
#include <stdio.h>

int* resizeBilinear_omp(int* pixels, int w, int h, int w2, int h2);