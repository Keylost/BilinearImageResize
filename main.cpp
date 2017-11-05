#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "resizeGPU.cuh"
#include "resizeCPU.hpp"
#include "resizeOMP.hpp"
#include "converter.hpp"

#define RESIZE_CALLS_NUM 1000

int main(int argc, char **argv)
{
	cv::Mat image;
	cv::Mat image_resized;
	int32_t *argb = NULL;
	int32_t *argb_res = NULL;
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime = 0;
	cv::Size newSz(1280, 1280);

	if (argc < 2)
	{
		printf("Usage:\n\t %s path_to_image\n", argv[0]);
		exit(0);
	}

	image = cv::imread(argv[1], 1);
	if (image.empty())
	{
		printf("Can't load image %s\n", argv[1]);
	}

	argb = cvtMat2Int32(image);

	//gpu block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res;
		argb_res = resizeBilinear_gpu(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time GPU: %f\n", cpu_ElapseTime);
	//gpu block end

	//cpu (no OpenMP) block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res;
		argb_res = resizeBilinear_cpu(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CPU: %f\n", cpu_ElapseTime);
	//cpu (no OpenMP) block end

	//OpenMP block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res;
		argb_res = resizeBilinear_omp(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CPU (OpenMP): %f\n", cpu_ElapseTime);
	//OpenMP block end

	image_resized = cv::Mat(newSz, CV_8UC3);
	cvtInt322Mat(argb_res, image_resized);

	cv::imshow("Original", image);
	cv::imshow("Resized", image_resized);
	cv::waitKey(0);

	delete[] argb_res;
	delete[] argb;

	return 0;
}