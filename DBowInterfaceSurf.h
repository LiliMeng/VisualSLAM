*
 * DBowInterfaceSurf.h
 *
 *  Created on: 13 Dec 2011
 *      Author: thomas
 */

#ifndef DBOWINTERFACESURF_H_
#define DBOWINTERFACESURF_H_

#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <DVision/DVision.h>
#include <DBoW2/FSurf64.h>
#include <DLoopDetector/TemplatedLoopDetector.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif

class DBowInterfaceSurf
{
    public:
        DBowInterfaceSurf(int width, int height, int mode, const std::string & filename);
        virtual ~DBowInterfaceSurf();

        void detectSURF(const cv::Mat & im, std::vector<float> & imageDescriptor, std::vector<cv::KeyPoint> & imageKeyPoints);
        DLoopDetector::DetectionResult detectLoop();
        void computeExportVocab();

        const int width;
        const int height;

        const int currentMode;
        const std::string filename;

        enum Mode
        {
            VOCAB_CREATION = 0,
            LOOP_DETECTION = 1
        };

        const static int surfDescriptorLength = 64;

        void reset();

    private:
        std::vector<cv::KeyPoint> keys;
        std::vector<FSurf64::TDescriptor> descriptors;
        std::vector<std::vector<FSurf64::TDescriptor> > descriptorsCollection;
        DLoopDetector::DetectionResult result;

        DVision::SurfSet surfExtractor;
        DBoW2::TemplatedVocabulary<FSurf64::TDescriptor, FSurf64> * vocab;
        DLoopDetector::TemplatedLoopDetector<FSurf64::TDescriptor, FSurf64> * surfLoopDetector;
};

#endif /* DBOWINTERFACESURF_H_ */
