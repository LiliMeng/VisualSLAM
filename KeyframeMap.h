
#ifndef KEYFRAMEMAP_H_
#define KEYFRAMEMAP_H_

#include "PoseGraph/iSAMInterface.h"
#include "Utils/Resolution.h"
#include "Utils/Projection.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

class KeyframeMap
{
public:
    KeyframeMap(bool filter = false);
    virtual ~KeyframeMap();

    void addKeyframe(unsigned char* rgbImage,
                     unsigned short * depthData,
                     Eigen::Matrix3f Rcurr,
                     Eigen::Vector3f tcurr,
                     uint64_t time);

    void applyPoses(iSAMInterface & isam);

    pcl::PointCloud<pcl::PointXYZRGB>* getMap();

    class Keyframe
    {
        public:
            Keyframe(pcl::PointCloud<pcl::PointXYZRGB>::Ptr points,
                     Eigen::Matrix4f pose,
                     uint64_t timestamp)
            : points(points),
              pose(pose),
              timestamp(timestamp)
            {}

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr points;
            Eigen::Matrix4f pose;
            uint64_t timestamp;
    };
private:
    bool filter;
    std::vector<Keyframe> map;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToXYZRGBPointCloud(unsigned char * rgbImage, unsigned short * depthData);
};

#endif /* KEYFRAMEMAP_H */

};
