/*
 * Surf3DTools.h
 *
 *  Created on: 12 May 2012
 *      Author: thomas
 */

#ifndef SURF3DTOOLS_H_
#define SURF3DTOOLS_H_

#include "DBowInterfaceSurf.h"
#include <opencv2/opencv.hpp>
#include <sstream>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

#include "opencv2/calib3d/calib3d.hpp"

#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif
#include "KinectCamera.h"

using namespace std;

class InterestPoint
{
    public:
        InterestPoint(float X, float Y, float Z, float u, float v)
             : X(X), Y(Y), Z(Z), u(u), v(v)
            {}
            float X, Y, Z, u, v;
};


class Surf3DTools
{
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

        static Surf3DImage * calculate3dPointsSURF(KinectCamera * kinectCamera,
                                                   cv::Mat * depthMap,
                                                   std::vector<float> & imageDescriptor,
                                                   std::vector<cv::KeyPoint> & imageKeyPoints)
        {
            Surf3DImage * newSurf3DImage = new Surf3DImage(imageDescriptor, imageKeyPoints);

            cv::Mat * image3D = new cv::Mat();

            kinectCamera->computeImage3D(*depthMap, *image3D);

            const double maxZ = 10.0;

            for(int y = 0; y < image3D->rows; y++)
            {
                for(int x = 0; x < image3D->cols; x++)
                {
                    cv::Vec3f point = image3D->at<cv::Vec3f> (y, x);

                    if(fabs(point[2] - maxZ) < FLT_EPSILON || fabs(point[2]) > maxZ)
                    {
                        continue;
                    }
                    else
                    {
                        newSurf3DImage->pointCorrespondences.push_back(Surf3DImage::PointCorrespondence(cvPoint3D32f(point[0],
                                                                                                                     point[1],
                                                                                                                     point[2]),
                                                                                                        cvPoint2D32f(x, y)));
                    }
                }
            }

            delete image3D;

            return newSurf3DImage;
        }

        static void surfMatch3D(Surf3DImage * one,
                                Surf3DImage * two,
                                std::vector<std::vector<float> > & matches1,
                                std::vector<std::vector<float> > & matches2)
        {
            cv::FlannBasedMatcher matcher;

            std::vector< cv::DMatch > matchesRaw;

            cv::Mat descriptors_1 = cv::Mat(one->descriptor.size()/64, 64, CV_32FC1);
            memcpy(descriptors_1.data, one->descriptor.data(), one->descriptor.size()*sizeof(float));

            cv::Mat descriptors_2 = cv::Mat(two->descriptor.size()/64, 64, CV_32FC1);
            memcpy(descriptors_2.data, two->descriptor.data(), two->descriptor.size()*sizeof(float));

            matcher.match(descriptors_1, descriptors_2, matchesRaw);

            cout<<"matchesRaw.size()  "<<matchesRaw.size()<<endl;

            // compute homography using RANSAC
            cv::Mat mask;
            int ransacThreshold=9;

            vector<cv::Point2d> imgpts1beforeRANSAC, imgpts2beforeRANSAC;

            for( int i = 0; i < (int)matchesRaw.size(); i++ )
            {
                imgpts1beforeRANSAC.push_back(one->keyPoints[matchesRaw[i].queryIdx].pt);
                imgpts2beforeRANSAC.push_back(two->keyPoints[matchesRaw[i].trainIdx].pt);
            }

            cv::Mat H12 = cv::findHomography(imgpts1beforeRANSAC, imgpts2beforeRANSAC, CV_RANSAC, ransacThreshold, mask);
            cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
            int numMatchesbeforeRANSAC=(int)matchesRaw.size();

            cout<<"The number of matches before RANSAC"<<numMatchesbeforeRANSAC<<endl;

            int numRANSACInlier=0;
            std::vector< cv::DMatch > matches;

            for(int i=0; i<(int)matchesRaw.size(); i++)
            {
                if((int)mask.at<uchar>(i, 0) == 1)
                {
                    numRANSACInlier+=1;
                    matches.push_back(matchesRaw[i]);
                }
            }

            cout<<"The number of matches after RANSAC"<<numRANSACInlier<<endl;


            matches1.resize(matches.size());
            matches2.resize(matches.size());

            for(unsigned int i = 0; i < matches.size(); i++)
            {
                matches1[i].resize(5);
                matches1[i][0] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.x;
                //cout<<"matches1[i][0] "<<matches1[i][0]<<endl;
                matches1[i][1] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.y;
               //cout<<"matches1[i][1] "<<matches1[i][1]<<endl;
                matches1[i][2] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.z;
                //cout<<"matches1[i][2] "<<matches1[i][2]<<endl;
                matches1[i][3] = one->keyPoints.at(matches[i].queryIdx).pt.x;
                //cout<<"matches1[i][3] "<<matches1[i][3]<<endl;
                matches1[i][4] = one->keyPoints.at(matches[i].queryIdx).pt.y;
               // cout<<"matches1[i][4] "<<matches1[i][4]<<endl;
                matches2[i].resize(5);
                matches2[i][0] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.x;
               // cout<<"matches2[i][0] "<<matches2[i][0]<<endl;
                matches2[i][1] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.y;
                //cout<<"matches2[i][1] "<<matches2[i][1]<<endl;
                matches2[i][2] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.z;
               // cout<<"matches2[i][2] "<<matches2[i][2]<<endl;
                matches2[i][3] = two->keyPoints.at(matches[i].trainIdx).pt.x;
              //  cout<<"matches2[i][3] "<<matches2[i][3]<<endl;
                matches2[i][4] = two->keyPoints.at(matches[i].trainIdx).pt.y;
               // cout<<"matches2[i][4] "<<matches2[i][4]<<endl;
            }
        }


        static void displayMatches(cv::Mat * im1,
                                   cv::Mat * im2,
                                   std::vector<std::vector<float> > & matches1,
                                   std::vector<std::vector<float> > & matches2, isam::Pose3d pose,
                                   bool save = false)
        {
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

//            im1->copyTo(left);
//            im2->copyTo(right);
//
//            for(unsigned int i = 0; i < matches1.size(); i++)
//            {
//                isam::StereoMeasurement m = kinectCamera->projectWithPose(pose, isam::Point3dh(matches2[i][2], matches2[i][0], matches2[i][1], 1.));
//                cv::line(*full_image, cv::Point(matches1[i][3], matches1[i][4]), cv::Point(m.u, m.v), cv::Scalar(0, 0, 255), 1);
//            }
//
//            for(unsigned int i = 0; i < matches2.size(); i++)
//            {
//                isam::StereoMeasurement m = kinectCamera->projectWithPose(isam::Pose3d(pose.oTw()), isam::Point3dh(matches1[i][2], matches1[i][0], matches1[i][1], 1.));
//                cv::line(*full_image, cv::Point(matches2[i][3] + left.cols, matches2[i][4]), cv::Point(m.u + left.cols, m.v), cv::Scalar(0, 0, 255), 1);
//            }

            cv::imshow("Loop Closure", *full_image);

            cvWaitKey(3);

            delete full_image;
        }



    private:
        Surf3DTools()
        {}
};

#endif /* SURF3DTOOLS_H_ */
