
#include "resizeGPU.cuh"
//#define _DEBUG

#define BLOCK_DIM 32
#define threadNum 1024
#define WARP_SIZE 32
#define elemsPerThread 1

int32_t* deviceDataResized; //отмасштабированное изображение в памяти GPU
int32_t* deviceData; //оригинальное изображение в памяти GPU
int32_t* hostOriginalImage;
int32_t* hostResizedImage;

void reAllocPinned(int w, int h, int w2, int h2, int32_t* dataSource)
{
	cudaMallocHost((void**)&hostOriginalImage, w*h* sizeof(int32_t)); // host pinned
	cudaMallocHost((void**)&hostResizedImage, w2*h2 * sizeof(int32_t)); // host pinned
	memcpy(hostOriginalImage, dataSource, w*h * sizeof(int32_t));

	return;
}

void freePinned()
{
	cudaFreeHost(hostOriginalImage);
	cudaFreeHost(hostResizedImage);

	return;
}

void initGPU(const int maxResolutionX, const int maxResolutionY)
{
	cudaMalloc((void**)&deviceDataResized, maxResolutionX*maxResolutionY * sizeof(int32_t));
	cudaMalloc((void**)&deviceData, maxResolutionX*maxResolutionY * sizeof(int32_t));

	return;
}

void deinitGPU()
{
	cudaFree(deviceData);
	cudaFree(deviceDataResized);

	return;
}

__global__ void SomeKernel(int32_t* originalImage, int32_t* resizedImage, int w, int h, int w2, int h2/*, float x_ratio, float y_ratio*/)
{
	__shared__ int32_t tile[1024];
	const float x_ratio = ((float)(w - 1)) / w2;
	const float y_ratio = ((float)(h - 1)) / h2;
	//const int blockbx = blockIdx.y * w2 + blockIdx.x*BLOCK_DIM;
	//unsigned int threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x;
	unsigned int threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread;
	//__shared__ float result[threadNum*elemsPerThread];
	unsigned int shift = 0;
	//int32_t a, b, c, d, x, y, index;
	while((threadId < w2*h2 && shift<elemsPerThread))
	{
		const int32_t i = threadId / w2;
		const int32_t j = threadId - (i*w2);
		//float x_diff, y_diff, blue, red, green;
		
		const int32_t x = (int)(x_ratio * j);
		const int32_t y = (int)(y_ratio * i);
		const float x_diff = (x_ratio * j) - x;
		const float y_diff = (y_ratio * i) - y;
		const int32_t index = (y*w + x);
		const int32_t a = originalImage[index];
		const int32_t b = originalImage[index + 1];
		const int32_t c = originalImage[index + w];
		const int32_t d = originalImage[index + w + 1];
		// blue element
		// Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
		const float blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
			(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

		// green element
		// Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
		const float green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

		// red element
		// Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
		const float red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

		/*
		resizedImage[threadId] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);
		*/
		tile[threadIdx.x] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);

		threadId++;
		//threadId+= WARP_SIZE;
		shift++;
	}
	
	__syncthreads();
	threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread;
	resizedImage[threadId] = tile[threadIdx.x];
	/*
	shift--;
	threadId = blockIdx.x * threadNum*elemsPerThread + threadIdx.x*elemsPerThread+ shift;

	while (shift >= 0)
	{
		resizedImage[threadId] = tile[shift];
		shift--;
		threadId--;
	}
	*/
}



int32_t* resizeBilinear_gpu(int w, int h, int w2, int h2)
{
#ifdef _DEBUG
	cudaError_t error; //store cuda error codes
#endif
	int length = w2 * h2;

	// Копирование исходных данных в GPU для обработки
	cudaMemcpy(deviceData, hostOriginalImage, w*h * sizeof(int32_t), cudaMemcpyHostToDevice);
	//cudaMemcpy2D(deviceData, w * sizeof(int32_t), hostOriginalImage, w * sizeof(int32_t), w * sizeof(int32_t), h, cudaMemcpyHostToDevice);
	//error = cudaMemcpyToSymbol(deviceData, pixels, w*h * sizeof(int32_t),0, cudaMemcpyHostToDevice);
#ifdef _DEBUG
	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (pixels->deviceData), returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
#endif

	dim3 threads = dim3(threadNum, 1,1); //block size 32,32,x
	dim3 blocks = dim3(w2*h2/ threadNum*elemsPerThread, 1,1);
	//printf("Blockdim.x %d\n", blocks.x);
	//printf("thrdim.x %d\n", threads.x);

	// Запуск ядра из (length / 256) блоков по 256 потоков,
	// предполагая, что length кратно 256
	SomeKernel << <blocks, threads >> >(deviceData, deviceDataResized, w, h, w2, h2/*, x_ratio, y_ratio*/);


	cudaDeviceSynchronize();
	// Считывание результата из GPU
	cudaMemcpy(hostResizedImage, deviceDataResized, length * sizeof(int32_t), cudaMemcpyDeviceToHost);

	return hostResizedImage;
}