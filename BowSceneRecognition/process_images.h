#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat applyFilterSeq(Mat image, Mat filter);

void processImagesSeq();

Mat applyFilterPar_pixel(Mat image, Mat filter);

void processImagesPar_pixel();

vector<Mat> applyFilterPar(int idx, float* deviceImages, int* imageOffsets, int* imageHeights, int* imageWidths,
    float* deviceFilters, int* devicefilterOffsets, int* devicefilterSizes, int numFilters, int filterDataSize);

void processImagesPar();
