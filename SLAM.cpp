#include "RawLogReader.h"
#include "KeyframeMap.h"
#include "PoseGraph/iSAMInterface.h"
#include "PlaceRecognition/PlaceRecognition.h"
#include "Odometry/FOVISOdometry.h"
#include "Odometry/DVOdometry.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <vector>
#include <fstream>

void drawPoses(std::vector<std::pair<uint64_t, Eigen::Matrix4f> > & poses,
               pcl::visualization::PCLVisualizer & cloudViewer, double r, double g, double b)
{
    static int count = 0;

    for(size_t i = 1; i < poses.size(); i++)
    {
        pcl::PointXYZ p1, p2;

        p1.x = poses.at(i - 1).second(0, 3);
        p1.y = poses.at(i - 1).second(1, 3);
        p1.z = poses.at(i - 1).second(2, 3);

        p2.x = poses.at(i).second(0, 3);
        p2.y = poses.at(i).second(1, 3);
        p2.z = poses.at(i).second(2, 3);

        std::stringstream strs;

        strs << "l" << count++;

        cloudViewer.addLine(p1, p2, r, g, b, strs.str());
    }
}

int main(int argc, char * argv[])
{
    int width = 640;
    int height = 480;

    Resolution::getInstance(width, height);

    Intrinsics::getInstance(528, 528, 320, 240);

    cv::Mat intrinsicMatrix = cv::Mat(3,3,CV_64F);

    intrinsicMatrix.at<double>(0,0) = Intrinsics::getInstance().fx();
    intrinsicMatrix.at<double>(1,1) = Intrinsics::getInstance().fy();

    intrinsicMatrix.at<double>(0,2) = Intrinsics::getInstance().cx();
    intrinsicMatrix.at<double>(1,2) = Intrinsics::getInstance().cy();

    intrinsicMatrix.at<double>(0,1) =0;
    intrinsicMatrix.at<double>(1,0) =0;

    intrinsicMatrix.at<double>(2,0) =0;
    intrinsicMatrix.at<double>(2,1) =0;
    intrinsicMatrix.at<double>(2,2) =1;

     cout<<"IntrinsicsMatrix OK"<<endl;
    Bytef * decompressionBuffer = new Bytef[Resolution::getInstance().numPixels() * 2];
    IplImage * deCompImage = 0;

    std::string logFile="/home/lili/Kinect_Logs/2015-11-05.00.klg";
   // assert(pcl::console::parse_argument(argc, argv, "-l", logFile) > 0 && "Please provide a log file");

    RawLogReader logReader(decompressionBuffer,
                           deCompImage,
                           logFile,
                           true);

   cout<<"logReader OK"<<endl;
    cv::Mat1b tmp(height, width);
    cv::Mat3b depthImg(height, width);

    PlaceRecognition placeRecognition(&intrinsicMatrix);

    iSAMInterface isam;

    //Keyframes
    KeyframeMap map(true);
    Eigen::Vector3f lastPlaceRecognitionTrans = Eigen::Vector3f::Zero();
    Eigen::Matrix3f lastPlaceRecognitionRot = Eigen::Matrix3f::Identity();
    int64_t lastTime = 0;

    OdometryProvider * odom = 0;


    if(true)
    {
        odom = new FOVISOdometry;

        if(logReader.hasMore())
        {
            logReader.getNext();

            Eigen::Matrix3f Rcurr = Eigen::Matrix3f::Identity();
            Eigen::Vector3f tcurr = Eigen::Vector3f::Zero();

            odom->getIncrementalTransformation(tcurr,
                                               Rcurr,
                                               logReader.timestamp,
                                               (unsigned char *)logReader.deCompImage->imageData,
                                               (unsigned short *)&decompressionBuffer[0]);

        }
    }
    else
    {
        odom = new DVOdometry;

        if(logReader.hasMore())
        {
            logReader.getNext();

            DVOdometry * dvo = static_cast<DVOdometry *>(odom);

            dvo->firstRun((unsigned char *)logReader.deCompImage->imageData,
                          (unsigned short *)&decompressionBuffer[0]);
        }
    }

    ofstream fout1("camera_pose_FOVIS.csv");
    ofstream fout2("camera_pose_Keyframe.csv");
    ofstream fout3("camera_pose_after_optimization.csv");

    while(logReader.hasMore())
    {
        logReader.getNext();

        cv::Mat3b rgbImg(height, width, (cv::Vec<unsigned char, 3> *)logReader.deCompImage->imageData);

        cv::Mat1w depth(height, width, (unsigned short *)&decompressionBuffer[0]);

        cv::normalize(depth, tmp, 0, 255, cv::NORM_MINMAX, 0);

        cv::cvtColor(tmp, depthImg, CV_GRAY2RGB);

        cv::imshow("RGB", rgbImg);

        cv::imshow("Depth", depthImg);

        char key = cv::waitKey(1);

        if(key == 'q')
        {
            break;
        }
        else if(key == ' ')
        {
            key = cv::waitKey(0);
        }

        Eigen::Matrix3f Rcurr = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcurr = Eigen::Vector3f::Zero();

//        #1
        odom->getIncrementalTransformation(tcurr,
                                           Rcurr,
                                           logReader.timestamp,
                                          (unsigned char *)logReader.deCompImage->imageData,
                                          (unsigned short *)&decompressionBuffer[0]);


       fout1<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<endl;

        Eigen::Matrix3f Rdelta = Rcurr.inverse() * lastPlaceRecognitionRot;
        Eigen::Vector3f tdelta = tcurr - lastPlaceRecognitionTrans;

        Eigen::MatrixXd covariance = odom->getCovariance();

        if((Projection::rodrigues2(Rdelta).norm() + tdelta.norm())  >= 0.05)
        {
            isam.addCameraCameraConstraint(lastTime,
                                           logReader.timestamp,
                                           lastPlaceRecognitionRot,
                                           lastPlaceRecognitionTrans,
                                           Rcurr,
                                           tcurr,
                                           covariance);

            lastTime = logReader.timestamp;

            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = tcurr;

//            #2
            map.addKeyframe((unsigned char *)logReader.deCompImage->imageData,
                            (unsigned short *)&decompressionBuffer[0],
                            Rcurr,
                            tcurr,
                            logReader.timestamp);

            fout2<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<endl;
            int64_t matchTime;
            Eigen::Matrix4d transformation;
            Eigen::MatrixXd cov(6, 6);

            cout<<"map.addKeyframe is OK"<<endl;

//            #3
            if(placeRecognition.detectLoop((unsigned char *)logReader.deCompImage->imageData,
                                           (unsigned short *)&decompressionBuffer[0],
                                           logReader.timestamp,
                                           matchTime,
                                           transformation,
                                           cov))
            {
               isam.addLoopConstraint(logReader.timestamp, matchTime, transformation, cov);
              // cout<<matchTime<<endl;
            }

        }
    }

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > posesBefore;
    isam.getCameraPoses(posesBefore);

//    #4
    isam.optimise();

   // fout3<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<endl;
    map.applyPoses(isam);

    pcl::PointCloud<pcl::PointXYZRGB> * cloud = map.getMap();

    pcl::visualization::PCLVisualizer cloudViewer;

    cloudViewer.setBackgroundColor(1, 1, 1);
    cloudViewer.initCameraParameters();
    cloudViewer.addCoordinateSystem(0.1, 0, 0, 0);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> color(cloud->makeShared());
    cloudViewer.addPointCloud<pcl::PointXYZRGB>(cloud->makeShared(), color, "Cloud Viewer");

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > poses;
    isam.getCameraPoses(poses);

    drawPoses(poses, cloudViewer, 1.0, 0, 0);
    drawPoses(posesBefore, cloudViewer, 0, 0, 1.0);

    cloudViewer.spin();

    delete [] decompressionBuffer;

    return 0;
}

