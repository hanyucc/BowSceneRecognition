#include <fstream>
#include <filesystem>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "create_dictionary.h"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace chrono;
namespace fs = filesystem;


void kMeansSeq(const vector<vector<float>>& samples, vector<vector<float>> centers, int k = 100)
{

}


void randomSamples(const vector<vector<Mat>>& responses, vector<vector<float>>& samples, int n = 100)
{

}


void saveCenters(const vector<vector<float>>& centers) 
{

}


void createDictionary()
{
	int numFilters = 32;
	int n = 100, k = 100;

	vector<string> paths;
	vector<vector<Mat>> responses;

	loadPaths(paths);
	loadResponses(paths, responses, numFilters);

	vector<vector<float>> samples;
	randomSamples(responses, samples, n);

	vector<vector<float>> centers;
	kMeansSeq(samples, centers, k);

	assert(centers.size() == k);

	saveCenters();
}