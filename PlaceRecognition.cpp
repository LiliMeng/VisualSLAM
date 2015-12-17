/*
 * PlaceRecognition.cpp
 *
 *  Created on: 5 May 2012
 *      Author: thomas
 */

#include "PlaceRecognition.h"

PlaceRecognition::PlaceRecognition(cv::Mat * intrinsicMatrix)
 : numEntires(0),
   kinectCamera(new KinectCamera(intrinsicMatrix))
{
    dbowInterface = new DBowInterfaceSurf(Resolution::getInstance().width(), Resolution::getInstance().height(), DBowInterfaceSurf::LOOP_DETECTION, "../vocab.yml.gz");
    oldImage = new cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
    newImage = new cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
    imageGray = new cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8U);
    depthMapNew = new cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_16U);
    depthMapOld = new cv::Mat(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_16U);
}

PlaceRecognition::~PlaceRecognition()
{
    delete dbowInterface;
    delete oldImage;
    delete newImage;
    delete imageGray;
    delete depthMapNew;
    delete depthMapOld;
    delete kinectCamera;
}


bool PlaceRecognition::detectLoop(unsigned char * rgbImage,
                                  unsigned short * depthData,
                                  int64_t timestamp,
                                  int64_t & matchTime,
                                  Eigen::Matrix4d & transformation,
                                  Eigen::MatrixXd & cov)
{
    assert(numEntires + 1 < PR_BUFFER_SIZE && "PlaceRecognition full!");

    unsigned char * depthPr = new unsigned char[Resolution::getInstance().numPixels() * 2];
    unsigned char * imgPr = new unsigned char[Resolution::getInstance().numPixels() * 3];

    memcpy(depthPr, depthData, Resolution::getInstance().numPixels() * 2);
    memcpy(imgPr, rgbImage, Resolution::getInstance().numPixels() * 3);

    placeRecognitionBuffer[numEntires].rgbImage = imgPr;
    placeRecognitionBuffer[numEntires].imageSize = Resolution::getInstance().numPixels() * 3;
    placeRecognitionBuffer[numEntires].depthMap = (unsigned short *)depthPr;
    placeRecognitionBuffer[numEntires].depthSize = Resolution::getInstance().numPixels() * 2;
    placeRecognitionBuffer[numEntires].isCompressed = false;
    placeRecognitionBuffer[numEntires].utime = timestamp;

    memcpy(newImage->data, rgbImage, Resolution::getInstance().numPixels() * 3);

    placeRecognitionBuffer[numEntires].compress();

    cv::cvtColor(*newImage, *imageGray, CV_RGB2GRAY);

    DLoopDetector::DetectionResult result;

    dbowInterface->detectSURF(*imageGray,
                              placeRecognitionBuffer[numEntires].descriptor,
                              placeRecognitionBuffer[numEntires].keyPoints);

    bool gotLoop = false;

    result = dbowInterface->detectLoop();

    if(result.detection()==1)
    {
        cout<<"The loop is detected from dBow, and now process transformation check"<<endl;
        gotLoop = processLoopClosureDetection(result.match,
                                              matchTime,
                                              transformation,
                                              cov);

        cout<<"processLoopClosureDetection is OK"<<endl;
    }
    else
    {
        cout<<"cannot detect the loop by dBow"<<endl;

    }

    numEntires++;

    return gotLoop;
}

bool PlaceRecognition::processLoopClosureDetection(int matchId,
                                                   int64_t & matchTime,
                                                   Eigen::Matrix4d & transformation,
                                                   Eigen::MatrixXd & cov)
{
    bool result = false;
    vector<vector<float> > matches1, matches2;
    std::vector<InterestPoint *> iP1, iP2;

   //For new image
    placeRecognitionBuffer[numEntires].decompressDepthTo(depthMapNew->data);
    placeRecognitionBuffer[numEntires].decompressImgTo(newImage->data);

    Surf3DTools::Surf3DImage *  image3DSurfOne = Surf3DTools::calculate3dPointsSURF(kinectCamera,
                                                                                    depthMapNew,
                                                                                    placeRecognitionBuffer[numEntires].descriptor,
                                                                                    placeRecognitionBuffer[numEntires].keyPoints);

    cout<<"The numEntires in the process of processLoopClosureDetection is  (new coming image)"<<numEntires<<endl;
    cout<<"The matchID in the placeRecognitionBuffer is (the old image which needs to be matched)" <<matchId<<endl;

    placeRecognitionBuffer[matchId].decompressImgTo(oldImage->data);
    cv::cvtColor(*oldImage, *imageGray, CV_RGB2GRAY);
    placeRecognitionBuffer[matchId].decompressDepthTo(depthMapOld->data);

    Surf3DTools::Surf3DImage *  image3DSurfTwo = Surf3DTools::calculate3dPointsSURF(kinectCamera,
                                                                                    depthMapOld,
                                                                                    placeRecognitionBuffer[matchId].descriptor,
                                                                                    placeRecognitionBuffer[matchId].keyPoints);

    Surf3DTools::surfMatch3D(image3DSurfOne, image3DSurfTwo, matches1, matches2);

    assert(matches1.size() == matches2.size());

    if(matches1.size() < 20)
    {
        delete image3DSurfOne;
        delete image3DSurfTwo;
        return result;
    }

    for (std::vector<std::vector<float> >::const_iterator m = matches1.begin(); m != matches1.end(); m++)
    {
        iP1.push_back(new InterestPoint((*m)[0],(*m)[1],(*m)[2],(*m)[3],(*m)[4]));
    }


    for (std::vector<std::vector<float> >::const_iterator m = matches2.begin(); m != matches2.end(); m++)
    {
        iP2.push_back(new InterestPoint((*m)[0],(*m)[1],(*m)[2],(*m)[3],(*m)[4]));
    }

    std::vector<std::pair<int2, int2> > inliers;

    isam::Pose3d pose;
    PNPSolver * pnpSolver = new PNPSolver(kinectCamera);
    pnpSolver->getRelativePose(pose, inliers, iP1, iP2);

    std::cout<<"inliers.size() from solvePnPRansac  "<<inliers.size()<<endl;
    std::cout << "Loop found with image " << matchId << ", matches: " << matches1.size() << ", inliers: " << (double)inliers.size()/matches1.size() << "%... ";
    std::cout.flush();

    if((double)inliers.size()/matches1.size()  > 0.3)
    {
        float score = 0;

        Eigen::Matrix4d T;
        T.topLeftCorner(3,3) = pose.rot().wRo();
        T.col(3).head(3) << pose.x(), pose.y(), pose.z();
        T.row(3) << 0., 0., 0., 1.;

        Eigen::Matrix4f bootstrap = T.cast<float>().inverse();

        Eigen::Matrix4f icpTrans = icpDepthFrames(bootstrap,
                                                  (unsigned short *)depthMapOld->data,
                                                  (unsigned short *)depthMapNew->data,
                                                  score);

        if(score < 0.3)
        {
            std::cout << "accepted!" << std::endl;

            matchTime = placeRecognitionBuffer[matchId].utime;
            transformation = icpTrans.cast<double>();
           // cov = covariance;
            result = true;

            Surf3DTools::displayMatches(newImage, oldImage, matches1, matches2, pose, false);
        }
        else
        {
            std::cout << "rejected on ICP score" << std::endl;
        }
    }
    else
    {
        std::cout << "rejected on inlier percentage" << std::endl;
    }

    for(unsigned int i = 0; i < matches1.size(); i++)
    {
        delete iP1.at(i);
        delete iP2.at(i);
    }

    delete image3DSurfOne;
    delete image3DSurfTwo;
    delete  pnpSolver;

    return result;
}

Eigen::Matrix4f PlaceRecognition::icpDepthFrames(Eigen::Matrix4f & bootstrap, unsigned short * frame1, unsigned short * frame2, float & score)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOne = kinectCamera->convertToXYZPointCloud(frame1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTwo = kinectCamera->convertToXYZPointCloud(frame2);

    pcl::VoxelGrid<pcl::PointXYZ> sor;

    sor.setLeafSize(0.03, 0.03, 0.03);

    sor.setInputCloud(cloudOne);
    sor.filter(*cloudOne);

    sor.setInputCloud(cloudTwo);
    sor.filter(*cloudTwo);

    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned (new pcl::PointCloud <pcl::PointXYZ>);

    pcl::transformPointCloud(*cloudOne, *cloudOne, bootstrap);

    icp.setInputCloud(cloudOne);
    icp.setInputTarget(cloudTwo);
    icp.align(*aligned);

    std::cout << "score: " << icp.getFitnessScore() << ", ";
    std::cout.flush();

    Eigen::Matrix4f d = icp.getFinalTransformation() * bootstrap;

    score = icp.getFitnessScore();

    return d;
}
