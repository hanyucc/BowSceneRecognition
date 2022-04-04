#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include "process_images.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    assert(argc == 1);

    string s = argv[0];

    processImagesPar();
    //processImagesSeq();

    return 0;

}
