#include "CloudKeyframeMap.h"
KeyframeMap::KeyframeMap(bool filter) : filter(filter)
{

}
KeyframeMap::~KeyframeMap()
{

}

void KeyframeMap::addKeyframe(unsigned char * rgbImage,
                              unsigned short * depthData,
                              Eigen::Matrix3f Rcurr,
                              Eigen::Vector3f tcurr,
                              uint64_t time)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr frameCloud = Projection::convertToXYZRGBPointCloud(rgbImage, depthData);

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();

    transformation.topLeftCorner(3, 3) = Rcurr;
    transformation.topRightCorner(3, 1) = tcurr;

    cloudKeyframeMap.push_back(Keyframe(frameCloud, transformation, time));
}

void KeyframeMap::applyPoses(iSAMInterface & isam)
{
    for(size_t i=0; i<map.size(); i++)
    {
        cloudKeyframeMap.at(i).pose = isam.getCameraPose(cloudKeyframeMap.at(i).timestamp);
        pcl::transformPointCloud(*cloudKeyframeMap.at(i).points, *cloudKeyframeMap.at(i).points, cloudKeyframeMap.at(i).pose);
        Cloud.insert(cloud.end(), cloudKeyframeMap.at(i).points->begin(), cloudKeyframeMap.at(i).points->end());
    }

    if(filter)
    {
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud.makeShared());
        sor.setLeafSize(0.005, 0.005, 0.005);
        sor.filter(cloud);
    }

   cloudKeyframeMap.clear();
}



pcl::PointCloud<pcl::PointXYZRGB> * KeyframeMap::getMap()
{
    return &cloud;
}
