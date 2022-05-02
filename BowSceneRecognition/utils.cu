#include "utils.h"

using namespace cv;
using namespace std;

void loadImages(vector<string>& paths, vector<string>& labels, vector<Mat>& images)
{
    paths.clear();
    labels.clear();
    images.clear();

    string dir = "data";
    for (const auto& e1 : fs::directory_iterator(dir)) {

        string label = e1.path().string();
        label = label.substr(5);

        for (const auto& e2 : fs::directory_iterator(e1.path())) {
            string path = e2.path().string();

            if (path.substr(path.length() - 3).compare("jpg") != 0) {
                continue;
            }

            auto im = imread(path, cv::IMREAD_GRAYSCALE);
            im.convertTo(im, CV_32FC1, 1 / 255.0);

            paths.push_back(path);
            images.push_back(im);
            labels.push_back(label);
        }
    }
}

void loadFilters(vector<Mat>& filters)
{
    filters.clear();
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

void loadPaths(vector<string>& paths)
{
    paths.clear();
    string dir = "data";
    for (const auto& e1 : fs::directory_iterator(dir)) {
        for (const auto& e2 : fs::directory_iterator(e1.path())) {
            string path = e2.path().string();
            if (path.substr(path.length() - 3).compare("jpg") != 0) {
                continue;
            }
            paths.push_back(path);
        }
    }
}

void saveResponses(const vector<string>& paths, const vector<vector<Mat>>& responses)
{
    string dir = "responses\\";
    for (int i = 0; i < paths.size(); i += 1) {
        string dirPath = paths[i].substr(5, paths[i].size() - 9);
        dirPath = dir + dirPath;

        fs::create_directories(dirPath);

        for (int j = 0; j < responses[i].size(); j += 1) {
            imwrite(dirPath + "\\" + to_string(j) + ".exr", responses[i][j]);
        }

        cout << i << " " << dirPath << endl;
    }
}

void loadResponses(const vector<string>& paths, vector<vector<Mat>>& responses, const int numFilters)
{
    responses.clear();
    string dir = "responses\\";
    for (int i = 0; i < paths.size(); i += 1) {
        string dirPath = paths[i].substr(5, paths[i].size() - 9);
        dirPath = dir + dirPath;

        vector<Mat> r;

        for (int j = 0; j < numFilters; j += 1) {
            r.push_back(imread(dirPath + "\\" + to_string(j) + ".exr", IMREAD_ANYCOLOR | IMREAD_ANYDEPTH));
        }

        responses.push_back(r);
        cout << i << " " << dirPath << endl;
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


void loadCenters(vector<float*>& centers)
{
    ifstream fin("centers.txt");
    int n, m;

    fin >> n >> m;

    for (int i = 0; i < n; i += 1) {
        float* c = new float[m];
        for (int j = 0; j < m; j += 1) {
            fin >> c[j];
        }
        centers.push_back(c);
    }

    fin.close();
}


float distEuc(float* x, float* y, int n)
{
    float sum = 0;

    for (int i = 0; i < n; i += 1) {
        sum += pow((x[i] - y[i]), 2);
    }

    return sqrt(sum);

}


void saveFeatures(const vector<string>& paths, const vector<float*>& features, const int numCenters)
{
    ofstream fout("features.txt");

    fout << features.size() << " " << numCenters << endl;

    for (int i = 0; i < features.size(); i += 1) {
        fout << paths[i] << endl;
        for (int j = 0; j < numCenters; j += 1) {
            fout << features[i][j] << " ";
        }
        fout << endl;
    }

    fout.close();
}


void loadFeatures(vector<string>& labels, vector<string>& trainLabels, vector<string>& testLabels,
    vector<float*>& trainFeatures, vector<float*>& testFeatures, int& numCenters, float pTest = 0.1)
{
    ifstream fin("features.txt");

    int numFeatures;
    fin >> numFeatures >> numCenters;

    int idx = 0;

    while (!fin.eof()) {
        string path;
        fin >> path;

        if (path.empty()) {
            break;
        }

        float* feature = new float[numCenters];
        for (int j = 0; j < numCenters; j += 1) {
            fin >> feature[j];
        }

        float r = 1.f * rand() / RAND_MAX;
        if (r > pTest) {
            trainLabels.push_back(labels[idx]);
            trainFeatures.push_back(feature);
        }
        else {
            testLabels.push_back(labels[idx]);
            testFeatures.push_back(feature);
        }

        idx += 1;
    }

    fin.close();
}


void saveConfusionMap(vector<vector<int>> cmap)
{
    ofstream fout("features.txt");


}