#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "classify_images.h"
#include "utils.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace cv;
using namespace std;
using namespace chrono;


string kNearestNeighbors(float*& x, vector<float*>& features, vector<string>& labels, int n = 100, int K = 32, int k = 10)
{

	vector<pair<float, string>> dists;

	for (int i = 0; i < features.size(); i += 1) {
		dists.push_back(make_pair(distEuc(x, features[i], K), labels[i]));
	}

	auto comp = [](pair<float, string> p1, pair<float, string> p2) {
		return p1.first < p2.first;
	};

	sort(dists.begin(), dists.end(), comp);

	map<string, int> cnt;

	for (int i = 0; i < k; i += 1) {
		string s = dists[i].second;
		if (cnt.find(s) == cnt.end()) {
			cnt[s] = 1;
		}
		else {
			cnt[s] += 1;
		}
	}

	int maxCnt = 0;
	string label;
	for (auto l : cnt) {
		if (l.second > maxCnt) {
			maxCnt = l.second;
			label = l.first;
		}
	}

	return label;
	 
}


void classifyImagesSeq()
{
	int numFilters = 32;

	vector<string> paths;
	vector<string> labels;
	vector<Mat> images;

	vector<float*> trainFeatures;
	vector<float*> testFeatures;
	vector<string> trainLabels;
	vector<string> testLabels;
	int numCenters;

	loadImages(paths, labels, images);
	loadFeatures(labels, trainLabels, testLabels, trainFeatures, testFeatures, numCenters, 0.2);

	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	vector<string> predLabels;

	int correct = 0, total = 0;

#pragma omp parallel for
	for (int i = 0; i < testFeatures.size(); i += 1) {
		string label = kNearestNeighbors(testFeatures[i], trainFeatures, trainLabels, numCenters, numFilters, 25);
		predLabels.push_back(label);
		if (label.compare(testLabels[i]) == 0) {
#pragma omp atomic
			correct += 1;
		}
#pragma omp atomic
		total += 1;
	}

	cout << correct * 1.f / total << endl;

	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "create features computation time (par): " << (end - start) / 1000.0 << endl;

}
