#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;
using namespace std;
namespace fs = filesystem;

void loadImages(vector<string>& paths, vector<string>& labels, vector<Mat>& images);

void loadFilters(vector<Mat>& filters);

void loadPaths(vector<string>& paths);

void saveResponses(const vector<string>& paths, const vector<vector<Mat>>& responses);

void loadResponses(const vector<string>& paths, vector<vector<Mat>>& responses, const int numFilters);

void saveCenters(const vector<float*>& centers, const int numFilters);

void loadCenters(vector<float*>& centers);

float distEuc(float* x, float* y, int n);

void saveFeatures(const vector<string>& paths, const vector<float*>& features, const int numCenters);

void loadFeatures(vector<string>& labels, vector<string>& trainLabels, vector<string>& testLabels,
    vector<float*>& trainFeatures, vector<float*>& testFeatures, int& numCenters, float pTest);