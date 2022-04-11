#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void kMeansSeq(const vector<float*>& points, vector<float*> centers, int K, int n, int k);

void randomSamples(const vector<vector<Mat>>& responses, vector<float*>& points, int m);

void saveCenters(const vector<float*>& centers, const int numFilters);

void createDictionarySeq();

void kMeansPar(const vector<float*>& points, vector<float*> centers, int K, int n, int k);

void createDictionaryPar();