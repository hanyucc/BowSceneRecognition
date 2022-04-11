#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <unordered_map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "create_dictionary.h"
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
namespace fs = filesystem;

float distEuc(float* x, float* y, int n) 
{
	float sum = 0;
	
	//#pragma omp parallel for shared(sum) reduction(+:sum)
	for (int i = 0; i < n; i += 1) {
		sum += pow((x[i] - y[i]), 2);
	}

	return sqrt(sum);
	
}


/*
* Sequential baseline algorithm, iterates over data points and cluster centers naively
* 200 images, 100 points/image, 50 clusters
*
* Computation time: 74.615s (83 iterations) / 0.899s per iteration
*/
void kMeansSeq(const vector<float*>& points, vector<float*> centers, int K = 32, int k = 100)
{
	int l = points.size();

	centers = vector<float*>(k, nullptr);

	// initialize arbitrarily
	for (int i = 0; i < k; i += 1) {
		centers[i] = new float[K];
		copy(points[l / k * i], points[l / k * i] + K, centers[i]);
	}

	// cluster each points belongs to
	vector<int> cluster(l);
	vector<int> oldCluster(l);

	bool firstIter = true;
	int cnt = 0;

	// number of points in each cluster
	vector<int> numPoints(k, 0);
	// sum of points in each cluster
	vector<float*> sumPoints(k, nullptr);

	for (int i = 0; i < k; i += 1) {
		sumPoints[i] = new float[K]();
	}

	while (true) {

		fill(numPoints.begin(), numPoints.end(), 0);
		for (int i = 0; i < k; i += 1) {
			float* t = sumPoints[i];
			fill(t, t + K, 0);
		}

		float totalDist = 0;

		//#pragma omp parallel for
		for (int i = 0; i < l; i += 1) {
			int nearest = 0;
			float minDist = 1e20;
			for (int j = 0; j < k; j += 1) {
				float dist = distEuc(points[i], centers[j], K);
				if (dist < minDist) {
					minDist = dist;
					nearest = j;
				}
			}
			cluster[i] = nearest;
			//#pragma omp atomic
			{
				numPoints[nearest] += 1;
				totalDist += minDist;
				for (int j = 0; j < K; j += 1) {
					sumPoints[nearest][j] += points[i][j];
				}
			}
		}

		if (!firstIter) {
			bool flag = true;
			for (int i = 0; i < l; i += 1) {
				if (cluster[i] != oldCluster[i]) {
					flag = false;
					break;
				}
			}
			if (flag) {
				break;
			}
		}
		else {
			firstIter = false;
		}

		for (int i = 0; i < k; i += 1) {
			for (int j = 0; j < K; j += 1) {
				centers[i][j] = sumPoints[i][j] / numPoints[i];
			}
		}

		copy(cluster.begin(), cluster.end(), oldCluster.begin());

		cout << "iteration " << cnt << " total dist: " << totalDist << endl;
		cnt += 1;

	}

}


void randomSamples(const vector<vector<Mat>>& responses, vector<float*>& points, int m = 100)
{
	random_device rd;
	mt19937 rng(rd());

	int numImages = responses.size();
	int numFilters = responses[0].size();

	points = vector<float*>(numImages * m, nullptr);

	for (int i = 0; i < numImages; i += 1) {
		int h = responses[i][0].size().height;
		int w = responses[i][0].size().width;

		uniform_int_distribution<int> uni(0, h * w);
		vector<int> samples;
		for (int i = 0; i < m; i += 1) {
			samples.push_back(uni(rng));
		}

		//#pragma omp parallel for
		for (int j = 0; j < m; j += 1) {
			int s = samples[j];
			int hh = s / w;
			int ww = s % w;

			points[i * m + j] = new float[numFilters];

			for (int k = 0; k < numFilters; k += 1) {
				points[i * m + j][k] = ((float*)responses[i][k].data)[hh * w + ww];
			}
		}
	}
}


void saveCenters(const vector<float*>& centers, const int numFilters) 
{
	ofstream fout("centers.txt");

	fout << centers.size() << " " << numFilters << endl;

	for (int i = 0; i < centers.size(); i += 1) {
		for (int j = 0; j < numFilters; j += 1) {
			fout << centers[i][j] << " ";
		}
		fout << endl;
	}

	fout.close();
}


void createDictionarySeq()
{

	int numFilters = 32;
	int m = 100, k = 50;

	vector<string> paths;
	vector<vector<Mat>> responses;

	cout << "loading reponses" << endl;
	loadPaths(paths);
	loadResponses(paths, responses, numFilters);

	vector<float*> points;
	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	randomSamples(responses, points, m);

	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "sampling time: " << (end - start) / 1000.0 << endl;

	cout << "running k-means on " << points.size() << " points with k = " << k << endl;
	vector<float*> centers;
	start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	kMeansSeq(points, centers, numFilters, k);

	end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "k-means computation time (seq): " << (end - start) / 1000.0 << endl;

	assert(centers.size() == k);
	saveCenters(centers, numFilters);

}



__device__ float deviceDistEuc(float* x, float* y, int n)
{
	float sum = 0;

	for (int i = 0; i < n; i += 1) {
		sum += pow((x[i] - y[i]), 2);
	}

	return sqrt(sum);

}


__global__ void kMeansHelper1(float* points, float* centers, int* cluster, 
	int* numPoints, float* sumPoints, int l, int k, int K, float* totalDist)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < l) {
		int nearest = 0;
		float minDist = 1e20;
		for (int j = 0; j < k; j += 1) {
			float dist = deviceDistEuc(points + idx * K, centers + j * K, K);
			if (dist < minDist) {
				minDist = dist;
				nearest = j;
			}
		}

		atomicAdd(numPoints + nearest, 1);
		atomicAdd(totalDist, minDist);
		cluster[idx] = nearest;

		for (int j = 0; j < K; j += 1) {
			atomicAdd(sumPoints + nearest * K + j, points[idx * K + j]);
		}
	}
}



__global__ void kMeansHelper2(float* centers, int* cluster, int* oldCluster,
	int* numPoints, float* sumPoints, int l, int k, int K, bool* notDone)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < l && cluster[idx] != oldCluster[idx]) {
		*notDone = true;
	}

	if (idx < k) {
		for (int j = 0; j < K; j += 1) {
			centers[idx * K + j] = sumPoints[idx * K + j] / numPoints[idx];
		}
	}
}

/*
* Parallelizes over data points for both phases of the algorithm
* 200 images, 100 points/image, 50 clusters
*
* Block size 1:		120.614s	(192 iterations)	0.6282s per iteration
* Block size 16:	6.418s		(153 iterations)	0.0419s per iteration
* Block size 64:	2.733s		(129 iterations)	0.0212s per iteration
* Block size 256:	3.670s		(170 iterations)	0.0216s per iteration
* Block size 1024:	5.102s		(121 iterations)	0.0422s per iteration
*/
void kMeansPar(const vector<float*>& points, vector<float*> centers, int K, int k)
{
	int BLOCK_SIZE = 256;

	centers = vector<float*>(k);

	int l = points.size();

	float* devicePoints;
	float* deviceCenters;
	float* deviceSumPoints;
	int* deviceCluster;
	int* deviceOldCluster;
	int* deviceNumPoints;
	float* deviceTotalDist;
	bool* deviceNotDone;

	cudaMalloc(&devicePoints, sizeof(float) * l * K);
	cudaMalloc(&deviceCenters, sizeof(float) * k * K);
	cudaMalloc(&deviceSumPoints, sizeof(float) * k * K);
	cudaMalloc(&deviceCluster, sizeof(int) * l);
	cudaMalloc(&deviceOldCluster, sizeof(int) * l);
	cudaMalloc(&deviceNumPoints, sizeof(float) * k);
	cudaMalloc(&deviceTotalDist, sizeof(float));
	cudaMalloc(&deviceNotDone, sizeof(bool));

	float* initCenters = new float[k * K];
	for (int i = 0; i < k; i += 1) {
		copy(points[l / k * i], points[l / k * i] + K, initCenters + i * K);
	}

	float* pointsData = new float[l * K];
	for (int i = 0; i < l; i += 1) {
		copy(points[i], points[i] + K, pointsData + i * K);
	}

	cudaMemcpy(deviceCenters, initCenters, sizeof(float) * k * K, cudaMemcpyHostToDevice);
	cudaMemcpy(devicePoints, pointsData, sizeof(float) * l * K, cudaMemcpyHostToDevice);

	dim3 grid((l + BLOCK_SIZE + 1) / BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);

	bool notDone;
	int cnt = 0;

	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	while (true) {

		cudaMemset(deviceNumPoints, 0, sizeof(int) * k);
		cudaMemset(deviceSumPoints, 0, sizeof(float) * k * K);
		cudaMemset(deviceTotalDist, 0, sizeof(float));
		cudaMemset(deviceNotDone, 0, sizeof(bool));

		kMeansHelper1 << <grid, block >> > (devicePoints, deviceCenters, deviceCluster, deviceNumPoints, deviceSumPoints, l, k, K, deviceTotalDist);
		kMeansHelper2 << <grid, block >> > (deviceCenters, deviceCluster, deviceOldCluster, deviceNumPoints, deviceSumPoints, l, k, K, deviceNotDone);

		cudaMemcpy(deviceOldCluster, deviceCluster, sizeof(int) * l, cudaMemcpyDeviceToDevice);

		//int* cluster = new int[l];
		//cudaMemcpy(cluster, deviceCluster, sizeof(int) * l, cudaMemcpyDeviceToHost);

		//for (int i = 0; i < l; i += 1) {
		//	cout << cluster[i] << " ";
		//}
		//cout << endl;

		float dist;
		cudaMemcpy(&dist, deviceTotalDist, sizeof(float), cudaMemcpyDeviceToHost);
		cout << cnt++ << " " << dist << endl;

		cudaMemcpy(&notDone, deviceNotDone, sizeof(bool), cudaMemcpyDeviceToHost);

		if (!notDone) {
			break;
		}

	}

	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "k-means computation time (par): " << (end - start) / 1000.0 << endl;

	float* c = new float[k * K];
	gpuErrchk(cudaMemcpy(c, deviceCenters, sizeof(float) * k * K, cudaMemcpyDeviceToHost));

	for (int i = 0; i < k; i += 1) {
		centers[i] = new float[K];
		copy(c + i * K, c + (i + 1) * K, centers[i]);
	}

}

void createDictionaryPar()
{

	int numFilters = 32;
	int m = 100, k = 50;

	vector<string> paths;
	vector<vector<Mat>> responses;

	cout << "loading reponses" << endl;
	loadPaths(paths);
	loadResponses(paths, responses, numFilters);

	vector<float*> points;
	auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	randomSamples(responses, points, m);

	auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	cout << "sampling time: " << (end - start) / 1000.0 << endl;

	cout << "running k-means on " << points.size() << " points with k = " << k << endl;
	vector<float*> centers;
	kMeansPar(points, centers, numFilters, k);

	assert(centers.size() == k);
	saveCenters(centers, numFilters);

}