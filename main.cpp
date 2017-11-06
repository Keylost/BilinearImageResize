#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "resizeGPU.cuh"
#include "resizeCPU.hpp"
#include "resizeOMP.hpp"
#include "converter.hpp"

#define RESIZE_CALLS_NUM 10

int main(int argc, char **argv)
{
	cv::Mat image;
	cv::Mat image_resized_gpu;
	cv::Mat image_resized_cpu;
	cv::Mat image_resized_omp;
	int32_t *argb = NULL;
	int32_t *argb_res_gpu = NULL;
	int32_t *argb_res_cpu = NULL;
	int32_t *argb_res_omp = NULL;
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
		delete[] argb_res_gpu;
		argb_res_gpu = resizeBilinear_gpu(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time GPU: %f\n", cpu_ElapseTime);
	//gpu block end

	//cpu (no OpenMP) block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res_cpu;
		argb_res_cpu = resizeBilinear_cpu(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CPU: %f\n", cpu_ElapseTime);
	//cpu (no OpenMP) block end

	//OpenMP block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res_omp;
		argb_res_omp = resizeBilinear_omp(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CPU (OpenMP): %f\n", cpu_ElapseTime);
	//OpenMP block end

	//show result images of each module
	image_resized_gpu = cv::Mat(newSz, CV_8UC3);
	image_resized_cpu = cv::Mat(newSz, CV_8UC3);
	image_resized_omp = cv::Mat(newSz, CV_8UC3);
	cvtInt322Mat(argb_res_gpu, image_resized_gpu);
	cvtInt322Mat(argb_res_cpu, image_resized_cpu);
	cvtInt322Mat(argb_res_omp, image_resized_omp);
	cv::imshow("Original", image);
	cv::imshow("Resized_GPU", image_resized_gpu);
	cv::imshow("Resized_CPU", image_resized_cpu);
	cv::imshow("Resized_OMP", image_resized_omp);
	cv::waitKey(0);

	//free memory
	delete[] argb_res_gpu, argb_res_cpu, argb_res_omp;
	delete[] argb;

	return 0;
}