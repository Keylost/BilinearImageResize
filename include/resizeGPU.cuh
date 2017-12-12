#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

int* resizeBilinear_gpu(int w, int h, int w2, int h2);

void initGPU(const int maxResolutionX, const int maxResolutionY);
void deinitGPU();

void reAllocPinned(int w, int h, int w2, int h2, int32_t* dataSource);
void freePinned();