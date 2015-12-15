#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

class readData{

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
        std::vector< DMatch > matchesRaw;

        matcher.match( descriptors_1, descriptors_2, matchesRaw);

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

        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.


        vector<float> descriptors1(descriptors_1.rows*descriptors_1.cols);

        vector<float> descriptors2(descriptors_2.rows*descriptors_2.cols);


        Surf3DImage * one=calculate3dPointsSURF(depth_1, descriptors1, keypoints_1);
        Surf3DImage * two=calculate3dPointsSURF(depth_2, descriptors2, keypoints_2);

        cout<<"descriptors1.size() "<<descriptors1.size()<<endl;
        cout<<"descriptors2.size() "<<descriptors2.size()<<endl;

        std::vector< DMatch > matches;

        for( int i = 0; i < descriptors_1.rows; i++)
        {
          if( matchesRaw[i].distance <= max(2*min_dist, 0.02)&& (int)mask.at<uchar>(i, 0) == 1)  //consider RANSAC
            { matches.push_back(matchesRaw[i]); }
        }

        cout<<"good_matches.size() after RANSAC "<<matches.size()<<endl;
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

        cv::Mat * im1=&img_1;
        cv::Mat * im2=&img_2;

        cv::namedWindow("Loop Closure", CV_WINDOW_AUTOSIZE);

        cv::Mat * full_image;

        cv::Size full_size(im1->cols * 2, im1->rows);

        full_image = new cv::Mat(full_size, CV_8UC3);

        cv::Mat left = full_image->colRange(0,im1->cols);
        im1->copyTo(left);
        cv::Mat right = full_image->colRange(im1->cols,im1->cols*2);
        im2->copyTo(right);

        for(unsigned int i = 0; i < matches1.size(); i++)
        {
                cv::line(*full_image, cv::Point(matches1[i][3], matches1[i][4]), cv::Point(matches2[i][3] + im1->cols, matches2[i][4]), cv::Scalar(0, 0, 255), 1);
        }

        cv::imshow("Loop Closure", *full_image);

        cvWaitKey(3);

     }

       void compare_descriptors(vector<float> & des1, vector<float> & des2, vector<int> & ind)
        {
            vector<int> ptpairs;
            int i;

            if((int)des1.size() > 4 * 64)
            {
                flannFindPairs(des1, des2, ptpairs);
            }

            cout<<"ptpairs.size() "<<ptpairs.size()<<endl;

            ind.resize(des1.size() /64);

            cout<<"ind.size() in the function of compare_descriptors "<<ind.size()<<endl;

            for(i = 0; i < (int)ptpairs.size(); i += 2)
            {
                ind[ptpairs[i]] = ptpairs[i + 1];
                cout<<"ptpairs[i] "<< ptpairs[i]<<"  ind[ptpairs[i]]  "<<ind[ptpairs[i]]<<endl;
            }
        }

        void flannFindPairs(vector<float> & des1, vector<float> & des2, vector<int> & ptpairs)
        {

            cout<<"des1.size() "<<des1.size()<<endl;
            cout<<"des2.size() "<<des2.size()<<endl;

            ptpairs.clear();

            if(des1.size() == 0 || des2.size() == 0)
            {
                return;
            }

            float A[(int)des1.size()];
            float B[(int)des2.size()];

            int k = 0;
            k = min((int)des1.size(), (int)des2.size());

            cout<<"k "<<k<<endl;

            for(int i = 0; i < k; i++ )
            {
                A[i] = des1[i];
                B[i] = des2[i];
            }

            if(k == (int)des1.size())
            {
                for(int i = k; i < (int)des2.size(); i++)
                {
                    B[i] = des2[i];
                }
            }
            else
            {
                for(int i = k; i < (int)des1.size(); i++)
                {
                    A[i] = des1[i];
                }
            }

            cv::Mat m_image((int)des1.size() / 64, 64, CV_32F, A);
            cv::Mat m_object((int)des2.size() /64, 64, CV_32F, B);

            // find nearest neighbors using FLANN
            cv::Mat m_indices((int)des2.size() / 64, 2, CV_32S);
            cv::Mat m_dists((int)des2.size() / 64, 2, CV_32F);

            cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(1));  // using 1 randomized kdtrees

            flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64)); // maximum number of leafs checked

            int * indices_ptr = m_indices.ptr<int>(0);
            float * dists_ptr = m_dists.ptr<float>(0);

            for(int i = 0; i < m_indices.rows; i++)
            {
                if (dists_ptr[2 * i] < 0.49 * dists_ptr[2 * i + 1])
                {
                    ptpairs.push_back(indices_ptr[2 * i]);
                    ptpairs.push_back(i);
                }
            }

            cout<<"ptpairs.size() in flannFindPairs "<<ptpairs.size()<<endl;
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
