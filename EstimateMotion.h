#ifndef _ESTIMATEMOTION_H
#define _ESTIMATEMOTION_H

#include <utility>
#include <map>
#include <algorithm>
#include <iomanip>
#include <iostream>

//Boost includes for uBLAS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/quaternion.hpp>

#include <opencv2/opencv.hpp>

#include "Map.h"
#include "math_util.h"
#include "PnP.h"
#include "../KinectCamera.h"

#include <complex>

#include "lcmtypes/mrlcm_obj_collection_t.h"
#include "lcmtypes/so_point_feature_track_list_t.h"
#include "lcmtypes/so_point_feature_with_disparity_t.h"

#include <boost/shared_ptr.hpp>

using namespace std;
using boost::shared_ptr;

struct FeatureTracker;
struct PointTrack;

class EstimateMotion;

typedef struct lm_projectPoints_data_s
{
        boost::numeric::ublas::matrix<float> R_base;
        KinectCamera * camera;
        std::vector<cv::Point3f> worldPoints;
        std::vector<bool> *inlier_bitmap;
        EstimateMotion *_EstimateMotion;
} lm_projectPoints_data_t;

class EstimateMotion
{
public:
    EstimateMotion(KinectCamera * kinectCamera, PnP * pnp);

    virtual ~EstimateMotion()
    {
        delete _pnp;
    }

    float computeMotion(const feature_list &features,
                        const feature_list &features2,
                        const landmark_list &landmarks,
                        boost::numeric::ublas::vector<float> &L,
                        boost::math::quaternion<float> &Q,
                        int max_indices[4],
                        float *covariance,
                        std::vector<std::pair<int2, int2> > &inliers);

}
class EstimateMotion
{
    public:
        EstimateMotion(KinectCamera * kinectCamera, PnP * pnp);

        virtual ~EstimateMotion()
        {
            delete _pnp;
        }

        float computeMotion(const feature_list &features,
                            const feature_list &features2,
                            const landmark_list &landmarks,
                            boost::numeric:ublas::vector<float> &L,
                            boost::math::quaternion<float> &Q,
                            int max_indices[4],
                            float *covariance,
                            std::vector<std::pair<int2, int2> > & inliers);
    private:
        bool pickPoints(std::vector<int> &indices,
                        int num_points,
                        const vector<so_point_feature_with_disparity_t *> &features,
                        float proximity_threshold = 20, int max_choices = 100);

        int ransac_projectPoints(const feature_list &features,
                                 const landmark_list &landmarks,
                                 std::vector<bool> &inlier_bitmap);

        KinectCamera * _camera;
        PnP * _pnp;

        const int MAX_ITER;
        const float RANSAC_THRESH, OUTLIER_THRESH;

        lm_projectPoints_data_t projectPoints_t;
};

#endif
