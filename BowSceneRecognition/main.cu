#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

#include "process_images.h"
#include "create_dictionary.h"
#include "create_features.h"
#include "classify_images.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    assert(argc == 1);

    string s = argv[0];

    if (s.compare("p") == 0) {
        processImagesPar();
    }
    else if (s.compare("d") == 0) {
        createDictionaryPar();
    }
    classifyImagesSeq();
    //processImagesSeq();

    return 0;

}
