#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <boost/thread.hpp>
#include <boost/thread/condition_variable.hpp>

using namespace std;
using namespace cv;

class readData{
struct int2{
        int x;
        int y;
};
public:
        class Surf3DImage
        {
            public:
                Surf3DImage(std::vector<float> & imageDescriptor,
                            std::vector<cv::KeyPoint> & imageKeyPoints)
                 : descriptor(imageDescriptor),
                   keyPoints(imageKeyPoints)
                {}

                class PointCorrespondence
                {
                    public:
                        PointCorrespondence(CvPoint3D32f point3d,
                                            CvPoint2D32f coordIm)
                         : point3d(point3d),
                           coordIm(coordIm)
                        {}

                        CvPoint3D32f point3d;
                        CvPoint2D32f coordIm;
                };

                std::vector<float> & descriptor;
                std::vector<cv::KeyPoint> & keyPoints;
                std::vector<PointCorrespondence> pointCorrespondences;
        };


       Surf3DImage * calculate3dPointsSURF(cv::Mat depth_1,
                                           std::vector<float> & imageDescriptor,
                                           std::vector<cv::KeyPoint> & imageKeyPoints)
      {
            Surf3DImage * newSurf3DImage = new Surf3DImage(imageDescriptor, imageKeyPoints);

           for(int i=0; i<(int)imageKeyPoints.size(); i++)
           {
                auto depthValue1 = depth_1.at<unsigned short>(imageKeyPoints[i].pt.y, imageKeyPoints[i].pt.x);

                double worldZ1=0;

                if(depthValue1 > min_dis && depthValue1 < max_dis )
                {
                    worldZ1=depthValue1/factor;
                }

                double worldX1=(imageKeyPoints[i].pt.x-cx)*worldZ1/fx;
                double worldY1=(imageKeyPoints[i].pt.y-cy)*worldZ1/fy;

                newSurf3DImage->pointCorrespondences.push_back(Surf3DImage::PointCorrespondence(cvPoint3D32f(worldX1,
                                                                                                             worldY1,
                                                                                                             worldZ1),
                                                        cvPoint2D32f(imageKeyPoints[i].pt.x, imageKeyPoints[i].pt.y)));
            }

            return newSurf3DImage;
     }

    void readRGBDFromFile(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
    {
         img_1 = imread(rgb_name1, CV_LOAD_IMAGE_GRAYSCALE);
         depth_1 = imread(depth_name1, CV_LOAD_IMAGE_ANYDEPTH); // CV_LOAD_IMAGE_ANYDEPTH

         img_2 = imread(rgb_name2, CV_LOAD_IMAGE_GRAYSCALE);
         depth_2 = imread(depth_name2, CV_LOAD_IMAGE_ANYDEPTH);
         assert(img_1.type()==CV_8U);
         assert(img_2.type()==CV_8U);
         assert(depth_1.type()==CV_16U);
         assert(depth_2.type()==CV_16U);

    }

    void featureMatching()
    {
        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );


        detector.detect( img_1, keypoints_1 );
        detector.detect( img_2, keypoints_2 );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;

        Mat descriptors_1, descriptors_2;

        extractor.compute(img_1, keypoints_1, descriptors_1 );
        extractor.compute(img_2, keypoints_2, descriptors_2 );


        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;

        matcher.match( descriptors_1, descriptors_2, matches);

        /*
        cout<<"matchesRaw.size()  "<<matchesRaw.size()<<endl;

          // compute homography using RANSAC
        cv::Mat mask;
        int ransacThreshold=9;

        vector<Point2d> imgpts1beforeRANSAC, imgpts2beforeRANSAC;

        for( int i = 0; i < (int)matchesRaw.size(); i++ )
        {
            imgpts1beforeRANSAC.push_back(keypoints_1[matchesRaw[i].queryIdx].pt);
            imgpts2beforeRANSAC.push_back(keypoints_2[matchesRaw[i].trainIdx].pt);
        }

        cv::Mat H12 = cv::findHomography(imgpts1beforeRANSAC, imgpts2beforeRANSAC, CV_RANSAC, ransacThreshold, mask);
        cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
        int numMatchesbeforeRANSAC=(int)matchesRaw.size();
        cout<<"The number of matches before RANSAC"<<numMatchesbeforeRANSAC<<endl;

        int numRANSACInlier=0;
        for(int i=0; i<(int)matchesRaw.size(); i++)
        {
            if((int)mask.at<uchar>(i, 0) == 1)
            {
                numRANSACInlier+=1;
            }
        }

        cout<<"The number of matches after RANSAC"<<numRANSACInlier<<endl;

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        {
            double dist = matchesRaw[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );

        */

        vector<float> descriptors1(descriptors_1.rows*descriptors_1.cols);

        cout<<"descriptors_1.rows "<<descriptors_1.rows<<"descriptors_1.cols "<<descriptors_1.cols<<endl;

        vector<float> descriptors2(descriptors_2.rows*descriptors_2.cols);

        cout<<"descriptors_2.rows "<<descriptors_2.rows<<"descriptors_2.cols "<<descriptors_2.cols<<endl;

        Surf3DImage * one=calculate3dPointsSURF(depth_1, descriptors1, keypoints_1);
        Surf3DImage * two=calculate3dPointsSURF(depth_2, descriptors2, keypoints_2);

        cout<<"descriptors1.size() "<<descriptors1.size()<<endl;
        cout<<"descriptors2.size() "<<descriptors2.size()<<endl;


        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
       /*
        std::vector< DMatch > matches;

        for( int i = 0; i < descriptors_1.rows; i++)
        {
          if( matchesRaw[i].distance <= max(2*min_dist, 0.02)&& (int)mask.at<uchar>(i, 0) == 1)  //consider RANSAC
            { matches.push_back(matchesRaw[i]); }
        }

        cout<<"good_matches.size() after RANSAC "<<matches.size()<<endl;
        */
        vector<vector<float> > matches1(matches.size()), matches2(matches.size());  //the matched keypoints in image1 and image2 separately.


        for(unsigned int i = 0; i < matches.size(); i++)
        {
            matches1[i].resize(5);
            matches1[i][0] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.x;
            cout<<"matches1[i][0] "<<matches1[i][0]<<endl;
            matches1[i][1] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.y;
            cout<<"matches1[i][1] "<<matches1[i][1]<<endl;
            matches1[i][2] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.z;
            cout<<"matches1[i][2] "<<matches1[i][2]<<endl;
            matches1[i][3] = one->keyPoints.at(matches[i].queryIdx).pt.x;
            cout<<"matches1[i][3] "<<matches1[i][3]<<endl;
            matches1[i][4] = one->keyPoints.at(matches[i].queryIdx).pt.y;
            cout<<"matches1[i][4] "<<matches1[i][4]<<endl;
            matches2[i].resize(5);
            matches2[i][0] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.x;
            cout<<"matches2[i][0] "<<matches2[i][0]<<endl;
            matches2[i][1] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.y;
            cout<<"matches2[i][1] "<<matches2[i][1]<<endl;
            matches2[i][2] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.z;
            cout<<"matches2[i][2] "<<matches2[i][2]<<endl;
            matches2[i][3] = two->keyPoints.at(matches[i].trainIdx).pt.x;
            cout<<"matches2[i][3] "<<matches2[i][3]<<endl;
            matches2[i][4] = two->keyPoints.at(matches[i].trainIdx).pt.y;
            cout<<"matches2[i][4] "<<matches2[i][4]<<endl;
        }


        Mat K=cv::Mat(3,3,CV_64F);
        K.at<double>(0,0)=fx;
        K.at<double>(1,1)=fy;
        K.at<double>(2,2)=1;
        K.at<double>(0,2)=cx;
        K.at<double>(1,2)=cy;
        K.at<double>(0,1)=0;
        K.at<double>(1,0)=0;
        K.at<double>(2,0)=0;
        K.at<double>(2,1)=0;

        std::vector<cv::Point3f> points3d;
        std::vector<cv::Point2f> points2d;

        for(size_t i = 0; i < matches.size(); i++)
        {
            points2d.push_back(cv::Point2f(matches2[i][3], matches2[i][4]));
            points3d.push_back(cv::Point3f(matches1[i][0], matches1[i][1], matches1[i][2]));
        }

        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1); //Zero distortion: Deal with it

        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);

        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

        std::vector<std::pair<int2, int2> > inliers;

        cv::Mat inliersCv;

        cv::solvePnPRansac(points3d,
                           points2d,
                           K,
                           distCoeffs,
                           rvec,
                           tvec,
                           false,
                           500,
                           10.0f,
                           0.85,
                           inliersCv);

        cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
        cv::Rodrigues(rvec, R_matrix);

        for(int i = 0; i < inliersCv.rows; i++)
        {
            int n = inliersCv.at<int>(i);
            int2 corresp1 = {int(matches1[i][3]), int(matches1[i][4])};
            int2 corresp2 = {int(matches2[i][3]), int(matches2[i][4])};
            inliers.push_back(std::pair<int2, int2>(corresp1, corresp2));
        }

        cout<<" inliers.size() "<<inliers.size()<<endl;
        cout<<" matches.size() "<<matches.size()<<endl;

        cout<< "inliers: " << (double)inliers.size()/matches.size() << "%... ";

     }

        void testing(string& rgb_name1, string& depth_name1, string& rgb_name2, string& depth_name2)
        {
            readRGBDFromFile(rgb_name1, depth_name1, rgb_name2, depth_name2);
            featureMatching();
        }

        Mat img_1, img_2;

        Mat depth_1, depth_2;

        vector<KeyPoint> keypoints_1, keypoints_2;

        //camera parameters
        double fx = 525.0; //focal length x
        double fy = 525.0; //focal le

        double cx = 319.5; //optical centre x
        double cy = 239.5; //optical centre y

        double min_dis = 500;
        double max_dis = 50000;

        double X1, Y1, Z1, X2, Y2, Z2;
        double factor = 5000;

};

int main()
{
    readData r;

    string rgb1="/home/lili/workspace/c++practice/SURF_Matching/image_00000.png";
    string depth1="/home/lili/workspace/c++practice/SURF_Matching/depth_00000.png";
    string rgb2="/home/lili/workspace/c++practice/SURF_Matching/image_00002.png";
    string depth2="/home/lili/workspace/c++practice/SURF_Matching/depth_00002.png";

    r.testing(rgb1,depth1,rgb2,depth2);


    return 0;
}
