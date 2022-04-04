#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void loadImages(vector<string>& labels, vector<Mat>& images);
void loadFilters(vector<Mat>& filters);

Mat applyFilterSeq(Mat image, Mat filter);
Mat applyFilterPar_pixel(Mat image, Mat filter);
vector<Mat> applyFilterPar(Mat image, float* deviceFilters, int* devicefilterOffsets, int* devicefilterSizes, int numFilters, int filterDataSize);

void processImagesSeq();
void processImagesPar_pixel();
void processImagesPar();