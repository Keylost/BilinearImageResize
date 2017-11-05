#include "converter.hpp"

int32_t* cvtMat2Int32(const cv::Mat& srcImage)
{
	int32_t *result = new int32_t[srcImage.cols*srcImage.rows];
	int offset = 0;

	for (int i = 0; i<srcImage.cols*srcImage.rows * 3; i += 3)
	{
		int32_t blue = srcImage.data[i];
		int32_t green = srcImage.data[i + 1];
		int32_t red = srcImage.data[i + 2];
		result[offset++] =
			0xff000000 |
			((((int32_t)red) << 16) & 0xff0000) |
			((((int32_t)green) << 8) & 0xff00) |
			((int32_t)blue);
	}

	return result;
}

void cvtInt322Mat(int32_t *pxArray, cv::Mat& outImage)
{
	int offset = 0;
	for (int i = 0; i<outImage.cols*outImage.rows * 3; i += 3)
	{
		int32_t a = pxArray[offset++];
		int32_t blue = a & 0xff;
		int32_t green = ((a >> 8) & 0xff);
		int32_t red = ((a >> 16) & 0xff);
		outImage.data[i] = blue;
		outImage.data[i + 1] = green;
		outImage.data[i + 2] = red;
	}
	return;
}