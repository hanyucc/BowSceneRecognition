#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace fs = filesystem;
using namespace cv;
using namespace std;

void loadImages(vector<string>& paths, vector<string>& labels, vector<Mat>& images);

void loadFilters(vector<Mat>& filters);

void loadPaths(vector<string>& paths);

void saveResponses(const vector<string>& paths, const vector<vector<Mat>>& responses);

void loadResponses(const vector<string>& paths, vector<vector<Mat>>& responses, const int numFilters);