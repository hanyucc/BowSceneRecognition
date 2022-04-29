#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "create_features.h"
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


/*
* 100 images
* Computation time: 459.172s / 4.592s per image  (with OpenMP)
*							   36.73s per image  (without OpenMP estimated)
*/
void createFeaturesSeq()
{
	int numFilters = 32;

	vector<string> paths;
	vector<string> labels;
	vector<Mat> images;

	vector<vector<Mat>> responses;
	vector<float*> centers;

	loadImages(paths, labels, images);
	loadResponses(paths, responses, numFilters);

	loadCenters(centers);
	int numCenters = centers.size();

	vector<float*> features(responses.size());


	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

#pragma omp parallel for
	for (int i = 0; i < responses.size(); i += 1) {
		int h = responses[i][0].size().height;
		int w = responses[i][0].size().width;

		float* cnt = new float[numCenters]();

#pragma omp parallel for
		for (int y = 0; y < h; y += 1) {
#pragma omp parallel for
			for (int x = 0; x < w; x += 1) {

				float* c = new float[numFilters];
#pragma omp parallel for
				for (int j = 0; j < numFilters; j += 1) {
					c[j] = ((float*)responses[i][j].data)[y * w + x];
				}

				float minDist = 1e20;
				int nearest = 0;

				for (int j = 0; j < numCenters; j += 1) {
					float dist = distEuc(c, centers[j], numFilters);
					if (dist < minDist) {
						minDist = dist;
						nearest = j;
					}
				}
#pragma omp atomic 
				cnt[nearest] += 1;

			}
		}

#pragma omp parallel for
		for (int j = 0; j < numCenters; j += 1) {
			cnt[j] /= h * w;
		}
		features[i] = cnt;

		cout << i << endl;
	}
	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "create features computation time (seq): " << (end - start) / 1000.0 << endl;

}


__device__ float deviceDistEuc2(float* x, float* y, int n)
{
	float sum = 0;

	for (int i = 0; i < n; i += 1) {
		sum += pow((x[i] - y[i]), 2);
	}

	return sqrt(sum);

}


__global__ void createFeaturesHelper(float* centers, float* response, int* cnt, int h, int w, int K, int k)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= h || j >= w) {
		return;
	}

	int idx = j * h + i;

	float* r = response + idx * K;

	int nearest = 0;
	float minDist = 1e20;
	for (int j = 0; j < k; j += 1) {
		float dist = deviceDistEuc2(r, centers + j * K, K);
		if (dist < minDist) {
			minDist = dist;
			nearest = j;
		}
	}

	atomicAdd(cnt + nearest, 1);

}

/*
* 100 images
* Computation time: 76.373s / 0.764s per image
*/
void createFeaturesPar()
{
	int BLOCK_SIZE = 8;
	int MAX_SIZE = 1024;

	int numFilters = 32;

	vector<string> paths;
	vector<string> labels;
	vector<Mat> images;

	vector<vector<Mat>> responses;
	vector<float*> centers;

	loadImages(paths, labels, images);
	loadResponses(paths, responses, numFilters);

	loadCenters(centers);
	int numCenters = centers.size();

	float* centersData = new float[numCenters * numFilters];

	for (int i = 0; i < numCenters; i += 1) {
		float* c = centers[i];
		copy(c, c + numFilters, centersData + i * numFilters);
	}

	float* deviceCenters;
	cudaMalloc(&deviceCenters, sizeof(float) * numCenters * numFilters);
	cudaMemcpy(deviceCenters, centersData, sizeof(float) * numCenters * numFilters, cudaMemcpyHostToDevice);

	int* deviceCnt;
	cudaMalloc(&deviceCnt, sizeof(int) * numCenters);

	float* deviceResponse;
	cudaMalloc(&deviceResponse, sizeof(float) * MAX_SIZE * MAX_SIZE * numFilters);

	vector<float*> features;

	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	for (int i = 0; i < responses.size(); i += 1) {
		int h = responses[i][0].size().height;
		int w = responses[i][0].size().width;

		float* responseData = new float[h * w * numFilters];

		for (int y = 0; y < h; y += 1) {
			for (int x = 0; x < w; x += 1) {
				int index = y * w + x;
				for (int j = 0; j < numFilters; j += 1) {
					responseData[index * numFilters + j] = ((float*)responses[i][j].data)[index];
				}
			}
		}
		cudaMemcpy(deviceResponse, responseData, sizeof(float) * h * w * numFilters, cudaMemcpyHostToDevice);
		cudaMemset(deviceCnt, 0, sizeof(int) * numCenters);

		dim3 grid((h + BLOCK_SIZE - 1) / BLOCK_SIZE, (w + BLOCK_SIZE - 1) / BLOCK_SIZE);
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);

		createFeaturesHelper<<<grid, block>>>(deviceCenters, deviceResponse, deviceCnt, h, w, numFilters, numCenters);

		int* cnt = new int[numCenters];
		cudaMemcpy(cnt, deviceCnt, sizeof(int) * numCenters, cudaMemcpyDeviceToHost);

		float* feature = new float[numCenters];
		for (int i = 0; i < numCenters; i += 1) {
			feature[i] = 1.0 * cnt[i] / (h * w);
		}

		features.push_back(feature);

		cout << i << endl;
	}

	cudaFree(deviceResponse);
	cudaFree(deviceCnt);
	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "create features computation time (par): " << (end - start) / 1000.0 << endl;

	saveFeatures(paths, features, numCenters);

}