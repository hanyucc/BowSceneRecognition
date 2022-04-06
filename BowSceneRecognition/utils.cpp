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
