#include <stdint.h>
#include <opencv2/opencv.hpp>

int32_t* cvtMat2Int32(const cv::Mat& srcImage);
void cvtInt322Mat(int32_t *pxArray, cv::Mat& outImage);