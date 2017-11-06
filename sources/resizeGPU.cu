
#include "resizeGPU.cuh"
//#define _DEBUG

__global__ void SomeKernel(int32_t* originalImage, int32_t* resizedImage, int w, int h, int w2, int h2, float x_ratio, float y_ratio)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t a, b, c, d, x, y, index;
	int i = threadId / w2;
	int j = threadId - (i*w2);
	float x_diff, y_diff, blue, red, green;

	if (threadId < w2*h2)
	{
		x = (int)(x_ratio * j);
		y = (int)(y_ratio * i);
		x_diff = (x_ratio * j) - x;
		y_diff = (y_ratio * i) - y;
		index = (y*w + x);
		a = originalImage[index];
		b = originalImage[index + 1];
		c = originalImage[index + w];
		d = originalImage[index + w + 1];
		// blue element
		// Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
		blue = (a & 0xff)*(1 - x_diff)*(1 - y_diff) + (b & 0xff)*(x_diff)*(1 - y_diff) +
			(c & 0xff)*(y_diff)*(1 - x_diff) + (d & 0xff)*(x_diff*y_diff);

		// green element
		// Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
		green = ((a >> 8) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 8) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 8) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 8) & 0xff)*(x_diff*y_diff);

		// red element
		// Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
		red = ((a >> 16) & 0xff)*(1 - x_diff)*(1 - y_diff) + ((b >> 16) & 0xff)*(x_diff)*(1 - y_diff) +
			((c >> 16) & 0xff)*(y_diff)*(1 - x_diff) + ((d >> 16) & 0xff)*(x_diff*y_diff);

		resizedImage[threadId] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);
	}
}

int32_t* resizeBilinear_gpu(int32_t* pixels, int w, int h, int w2, int h2)
{
	cudaError_t error; //store cuda error codes
	float x_ratio = ((float)(w - 1)) / w2;
	float y_ratio = ((float)(h - 1)) / h2;

	int length = w2 * h2;

	// Выделение оперативной памяти под заресайзеннное изображение(для CPU)
	int32_t* hostData = (int32_t*)malloc(length * sizeof(int32_t));

	int32_t* deviceDataResized;
	cudaMalloc((void**)&deviceDataResized, w2*h2 * sizeof(int32_t));

	// Выделение памяти GPU для оригинального изображения
	int32_t* deviceData;
	cudaMalloc((void**)&deviceData, w*h * sizeof(int32_t));
	// Копирование исходных данных в GPU для обработки
	error = cudaMemcpy(deviceData, pixels, w*h * sizeof(int32_t), cudaMemcpyHostToDevice);
	//error = cudaMemcpyToSymbol(deviceData, pixels, w*h * sizeof(int32_t),0, cudaMemcpyHostToDevice);
#ifdef _DEBUG
	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (pixels->deviceData), returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
#endif

	dim3 threads = dim3(256);
	dim3 blocks = dim3(w2*h2 / 256);
	//printf("Blockdim.x %d\n", blocks.x);

	// Запуск ядра из (length / 256) блоков по 256 потоков,
	// предполагая, что length кратно 256
	SomeKernel << <blocks, threads >> >(deviceData, deviceDataResized, w, h, w2, h2, x_ratio, y_ratio);


	cudaDeviceSynchronize();
	// Считывание результата из GPU
	cudaMemcpy(hostData, deviceDataResized, length * sizeof(int32_t), cudaMemcpyDeviceToHost);

	cudaFree(deviceData);
	cudaFree(deviceDataResized);

	return hostData;
}