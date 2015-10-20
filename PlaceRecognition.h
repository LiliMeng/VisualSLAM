#ifndef PLACERECOGNITION_H_
#define PLACERECOGNITION_H_

#include "MotionEstimation/LoopClosure.h"
#include "../Utils/Projection.h"
#include "Surf3DTools.h"
#include "DBoWInterfaceSurf.h"
#include "PlaceRecognitionInput.h"

#include <sstream>
#include <string>
#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/filters/voxel_grid.h>

class PlaceRecognition
{
public:
    PlaceRecognition();

    virtual ~PlaceRecognition();

    bool detectLoop(unsigned char * rgbImage,
                    unsigned short * depthData,
                    int64_t timestamp,
                    int64_t & matchTime,
                    Eigen::Matrix4d & transformation,
                    Eigen::Matrixxd & cov);

private:
    bool processLoopClosureDetection(int matchId,
                                     int64_t & matchTime,
                                     Eigen::Matrix4d & transformation,
                                     Eigen::MatrixXd & cov);

    Eigen::Matrix4f icpDepthFrames(Eigen::Matrix4f & bootstrap, unsigned short * frame1, unsigned short * frame2, float & score);

    static const int PR_BUFFER_SIZE = 3000;
    PlaceRecognitionInput placeRecognitionBuffer[PR_BUFFER_SIZE];
    int numEntries;

    DBowInterfaceSurf * dbowInterface;
    cv::Mat * oldImage;
    cv::Mat * newImage;
    cv::Mat * imageGray;
    cv::Mat * depthMapNew;
    cv::Mat * depthMapOld;
    KinectCamera * kinectCamera;

};

#endif /* PLACERECOGNITION_H_ */
