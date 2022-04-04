#include <fstream>
#include <filesystem>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "process_images.h"

using namespace cv;
using namespace std;
using namespace chrono;
namespace fs = filesystem;

void loadImages(vector<string>& labels, vector<Mat>& images)
{
    string path = "data";
    for (const auto& e1 : fs::directory_iterator(path)) {

        string label = e1.path().string();
        label = label.substr(5);

        for (const auto& e2 : fs::directory_iterator(e1.path())) {
            string path = e2.path().string();
            auto im = imread(path, cv::IMREAD_GRAYSCALE);
            im.convertTo(im, CV_32FC1, 1 / 255.0);

            images.push_back(im);
            labels.push_back(label);
        }
    }
}

void loadFilters(vector<Mat>& filters)
{
    string path = "filters.csv";
    ifstream fin(path);

    while (!fin.eof()) {
        int size;
        fin >> size;


        Mat filter(size, size, CV_32FC1);
        for (int i = 0; i < size; i += 1) {
            for (int j = 0; j < size; j += 1) {
                fin >> filter.at<float>(i, j);
            }

        }
        filters.push_back(filter);
    }
}


/*
* Sequential baseline algorithm, iterates over image pixels and filter pixels naively
* Computation time: 0.61532
*/
Mat applyFilterSeq(Mat image, Mat filter)
{
    Mat response(image.size(), CV_32FC1);

    int h, w, s;
    h = image.size().height;
    w = image.size().width;
    s = filter.size().height;

    int pad = (s - 1) / 2;
    cv::copyMakeBorder(image, image, pad, pad, pad, pad, BORDER_REPLICATE);

    float* imageData = (float*)(image.data);
    float* filterData = (float*)(filter.data);
    float* responseData = (float*)(response.data);

    for (int i = 0; i < h; i += 1) {
        for (int j = 0; j < w; j += 1) {

            float r = 0;

            for (int k = 0; k < s; k += 1) {
                for (int l = 0; l < s; l += 1) {
                    r += imageData[(i + k) * (w + 2 * pad) + j + l] * filterData[k * s + l];
                }
            }
            responseData[i * w + j] = r;
        }
    };

    return response;
}

void processImagesSeq()
{
    vector<string> labels;
    vector<Mat> images;

    vector<Mat> filters;

    loadImages(labels, images);
    loadFilters(filters);

    vector<vector<Mat>> responses(images.size(), vector<Mat>(filters.size()));

    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < 100; i += 1) {
        for (int j = 0; j < filters.size(); j += 1) {
            Mat r = applyFilterSeq(images[i], filters[j]);
            responses[i][j] = r;
        }
        if (i % 10 == 0) {
            cout << "completed " << i << endl;
        }
    }
    auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cout << "computation time per image (seq): " << (end - start) / 1000.0 / 100 << endl;
}


/*
* Parallelized only over image pixels
* 
*   grid(h, w) block(1, 1)              0.05730
*   grid(h/4, w/4) block(4, 4)          0.02671
*   grid(h/8, w/8) block(8, 8)          0.02579
*   grid(h/16, w/16) block(16, 16)      0.02731
*   grid(h/32, w/32) block(32, 32)      0.02891
*/
__global__ void filterHelper_pixel(float* image, float* filter, float* response, int h, int w, int s)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    float r = 0;

    if (i >= h || j >= w) {
        return;
    }

    for (int k = 0; k < s; k += 1) {
        for (int l = 0; l < s; l += 1) {
            r += image[(i + k) * (w + s - 1) + j + l] * filter[k * s + l];
        }
    }

    response[i * w + j] = r;

}

Mat applyFilterPar_pixel(Mat image, Mat filter)
{
    int BLOCK_SIZE = 8;

    Mat response(image.size(), CV_32FC1);

    int h, w, s;
    h = image.size().height;
    w = image.size().width;
    s = filter.size().height;

    dim3 grid((h + BLOCK_SIZE + 1) / BLOCK_SIZE, (w + BLOCK_SIZE + 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    int pad = (s - 1) / 2;
    cv::copyMakeBorder(image, image, pad, pad, pad, pad, BORDER_REPLICATE);

    float* imageData = (float*)(image.data);
    float* filterData = (float*)(filter.data);
    float* responseData = (float*)(response.data);

    float* deviceImage;
    float* deviceFilter;
    float* deviceResponse;

    cudaMalloc(&deviceImage, sizeof(float) * (h + 2 * pad) * (w + 2 * pad));
    cudaMalloc(&deviceFilter, sizeof(float) * s * s);
    cudaMalloc(&deviceResponse, sizeof(float) * h * w);

    cudaMemcpy(deviceImage, imageData, sizeof(float) * (h + 2 * pad) * (w + 2 * pad), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFilter, filterData, sizeof(float) * s * s, cudaMemcpyHostToDevice);

    filterHelper_pixel << <grid, block >> > (deviceImage, deviceFilter, deviceResponse, h, w, s);

    cudaMemcpy(responseData, deviceResponse, sizeof(float) * h * w, cudaMemcpyDeviceToHost);

    cudaFree(deviceImage);
    cudaFree(deviceFilter);
    cudaFree(deviceResponse);

    return response;
}

void processImagesPar_pixel()
{
    vector<string> labels;
    vector<Mat> images;

    vector<Mat> filters;

    loadImages(labels, images);
    loadFilters(filters);

    vector<vector<Mat>> responses(images.size(), vector<Mat>(filters.size()));

    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < 500; i += 1) {
        for (int j = 0; j < filters.size(); j += 1) {
            Mat r = applyFilterPar_pixel(images[i], filters[j]);
            responses[i][j] = r;
        }
        if (i % 10 == 0) {
            cout << "completed " << i << endl;
        }
    }
    auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cout << "computation time per image (par): " << (end - start) / 1000.0 / 500 << endl;
}


/*
* Parallelized over image pixels & filters
* All filters & images are sent to GPU memory in once as a 1-d array
* One image patch with one filter per block
* 
*   grid(h, w) block(1, 1)              0.04536
*   grid(h/4, w/4) block(4, 4)          0.01180
*   grid(h/8, w/8) block(8, 8)          0.01102
*   grid(h/16, w/16) block(16, 16)      0.01491
*   grid(h/32, w/32) block(32, 32)      0.02260
*/
__global__ void filterHelper(float* image, float* filterData, int* filterOffsets, int* filterSizes, float* response, int h, int w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.z;
    float r = 0;

    if (i >= h || j >= w) {
        return;
    }

    float* filter = filterData + filterOffsets[n];
    int s = filterSizes[n];

    int pad = (s - 1) / 2;

    for (int k = 0; k < s; k += 1) {
        for (int l = 0; l < s; l += 1) {
            int ii = min(max(i + k - pad, 0), h - 1);
            int jj = min(max(j + l - pad, 0), w - 1);

            r += image[ii * w + jj] * filter[k * s + l];
        }
    }

    response[n * h * w + i * w + j] = r;
}

vector<Mat> applyFilterPar(int idx, float* deviceImages, int* imageOffsets, int* imageHeights, int* imageWidths,
    float* deviceFilters, int* devicefilterOffsets, int* devicefilterSizes, int numFilters, int filterDataSize)
{
    int BLOCK_SIZE = 32;

    int h, w;
    h = imageHeights[idx];
    w = imageWidths[idx];

    dim3 grid((h + BLOCK_SIZE + 1) / BLOCK_SIZE, (w + BLOCK_SIZE + 1) / BLOCK_SIZE, numFilters);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    float* responseData = new float[numFilters * h * w];

    float* deviceImage = deviceImages + imageOffsets[idx];
    float* deviceResponse;

    cudaMalloc(&deviceResponse, sizeof(float) * numFilters * h * w);

    filterHelper << <grid, block >> > (deviceImage, deviceFilters, devicefilterOffsets, devicefilterSizes, deviceResponse, h, w);

    cudaMemcpy(responseData, deviceResponse, sizeof(float) * numFilters * h * w, cudaMemcpyDeviceToHost);
    cudaFree(deviceResponse);

    vector<Mat> response;

    for (int i = 0; i < numFilters; i += 1) {
        Mat r(h, w, CV_32FC1);
        copy(responseData + h * w * i, responseData + h * w * (i + 1), (float*)r.data);
        response.push_back(r);
    }

    return response;
}

void processImagesPar()
{
    vector<string> labels;
    vector<Mat> images;

    vector<Mat> filters;

    loadImages(labels, images);
    loadFilters(filters);

    vector<vector<Mat>> responses;

    // send all filters to GPU
    int* filterOffsets = new int[filters.size()];
    int* filterSizes = new int[filters.size()];
    int filterDataSize = 0;

    for (int i = 0; i < filters.size(); i += 1) {
        filterOffsets[i] = filterDataSize;
        filterSizes[i] = filters[i].size().height;
        filterDataSize += filterSizes[i] * filterSizes[i];
    }

    float* filterData = new float[filterDataSize];

    for (int i = 0; i < filters.size(); i += 1) {
        copy((float*)filters[i].data, ((float*)filters[i].data) + filterSizes[i] * filterSizes[i], filterData + filterOffsets[i]);
    }

    float* deviceFilters;
    int* devicefilterOffsets;
    int* devicefilterSizes;

    cudaMalloc(&deviceFilters, sizeof(float) * filterDataSize);
    cudaMalloc(&devicefilterOffsets, sizeof(int) * filters.size());
    cudaMalloc(&devicefilterSizes, sizeof(int) * filters.size());

    cudaMemcpy(deviceFilters, filterData, sizeof(float) * filterDataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicefilterOffsets, filterOffsets, sizeof(int) * filters.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(devicefilterSizes, filterSizes, sizeof(int) * filters.size(), cudaMemcpyHostToDevice);

    // send all images to GPU
    int* imageOffsets = new int[images.size()];
    int* imageHeights = new int[images.size()];
    int* imageWidths = new int[images.size()];
    int imageDataSize = 0;

    for (int i = 0; i < images.size(); i += 1) {
        imageOffsets[i] = imageDataSize;
        imageHeights[i] = images[i].size().height;
        imageWidths[i] = images[i].size().width;
        imageDataSize += imageHeights[i] * imageWidths[i];
    }

    float* imageData = new float[imageDataSize];
    for (int i = 0; i < images.size(); i += 1) {
        copy((float*)images[i].data, ((float*)images[i].data) + imageHeights[i] * imageWidths[i], imageData + imageOffsets[i]);
    }

    float* deviceImages;
    cudaMalloc(&deviceImages, sizeof(float) * imageDataSize);
    cudaMemcpy(deviceImages, imageData, sizeof(float) * imageDataSize, cudaMemcpyHostToDevice);

    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    for (int i = 0; i < 500; i += 1) {
        vector<Mat> r = applyFilterPar(i, deviceImages, imageOffsets, imageHeights, imageWidths, deviceFilters, devicefilterOffsets, devicefilterSizes, filters.size(), filterDataSize);
        responses.push_back(r);
    }
    auto end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cout << "computation time per image (par): " << (end - start) / 1000.0 / 500 << endl;

    cudaFree(deviceFilters);
    cudaFree(devicefilterOffsets);
    cudaFree(devicefilterSizes); 
}
