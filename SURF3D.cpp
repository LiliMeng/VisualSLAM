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
        //imshow( "Good Matches", img_matches );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;

        Mat descriptors_1, descriptors_2;

        extractor.compute(img_1, keypoints_1, descriptors_1 );
        extractor.compute(img_2, keypoints_2, descriptors_2 );

        vector<int> ind;

        Mat m(3,3,CV_32F);

        vector<float> descriptors1(descriptors_1.rows*descriptors_1.cols);

        vector<float> descriptors2(descriptors_2.rows*descriptors_2.cols);

        compare_descriptors(descriptors1, descriptors2, ind);

        std::vector<std::vector<float> > matches1, matches2;

        Surf3DImage * one=calculate3dPointsSURF(depth_1, descriptors1, keypoints_1);
        Surf3DImage * two=calculate3dPointsSURF(depth_2, descriptors2, keypoints_2);

        int inn1, inn2, is1, is2;

            float tol = 0.5;

            float mindist2, dist2;

            vector<int> j1, j2, i1, i2;

            for(int is = 0; is < (int) ind.size(); is++)
            {
                if(ind[is] > 0)
                {
                    inn1 = -1;
                    is1 = -1;
                    mindist2 = 1;

                    int tmp1=(int)one->pointCorrespondences.size();
                    cout<<"(int)one->PointCorrespondence.size() "<<(int)one->pointCorrespondences.size()<<endl;

                    cout<<"one->keyPoints.size() "<<(int)one->keyPoints.size()<<endl;

                     int tmp2=(int)one->keyPoints.size();
                    cout<<"one->keyPoints.at(66).pt.x "<<one->keyPoints.at(tmp2-1).pt.x<<endl;
                    cout<<"one->keyPoints.at(66).pt.y "<<one->keyPoints.at(tmp2-1).pt.y<<endl;

                    cout<<"one->pointCorrespondences.at(244466).coordIm.x "<<one->pointCorrespondences.at(tmp1-1).coordIm.x<<endl;
                    cout<<"one->pointCorrespondences.at(244466).coordIm.y "<<one->pointCorrespondences.at(tmp1-1).coordIm.y<<endl;

                    for(int ipc = 0; ipc < (int) one->pointCorrespondences.size(); ipc++)
                    {
                        if(one->pointCorrespondences.at(ipc).coordIm.y > (one->keyPoints.at(is).pt.y - tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.y < (one->keyPoints.at(is).pt.y + tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.x > (one->keyPoints.at(is).pt.x - tol) &&
                           one->pointCorrespondences.at(ipc).coordIm.x < (one->keyPoints.at(is).pt.x + tol))
                        {
                            dist2 = pow(one->pointCorrespondences.at(ipc).coordIm.x - one->keyPoints.at(is).pt.x, 2) +
                                    pow(one->pointCorrespondences.at(ipc).coordIm.y - one->keyPoints.at(is).pt.y, 2);

                            if(dist2 < mindist2)
                            {
                                mindist2 = dist2;
                                inn1 = ipc;
                                is1 = is;
                            }
                        }
                    }

                    inn2 = -1;
                    is2 = -1;
                    mindist2 = 1;

                     int tmp11=(int)two->pointCorrespondences.size();
                    cout<<"(int)two->PointCorrespondence.size() "<<(int)two->pointCorrespondences.size()<<endl;

                    cout<<"two->keyPoints.size() "<<(int)two->keyPoints.size()<<endl;

                    cout<<"two->keyPoints.at(is).pt.x "<<two->keyPoints.at(is).pt.x<<endl;
                    cout<<"two->keyPoints.at(ind[is]).pt.y "<<two->keyPoints.at(is).pt.y<<endl;

                    cout<<"two->pointCorrespondences.at(244466).coordIm.x "<<two->pointCorrespondences.at(tmp11-1).coordIm.x<<endl;
                    cout<<"two->pointCorrespondences.at(244466).coordIm.y "<<two->pointCorrespondences.at(tmp11-1).coordIm.y<<endl;


                    for(int ipc = 0; ipc < (int) two->pointCorrespondences.size(); ipc++)
                    {
                        if(two->pointCorrespondences.at(ipc).coordIm.y > (two->keyPoints.at(ind[is]).pt.y - tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.y < (two->keyPoints.at(ind[is]).pt.y + tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.x > (two->keyPoints.at(ind[is]).pt.x - tol) &&
                           two->pointCorrespondences.at(ipc).coordIm.x < (two->keyPoints.at(ind[is]).pt.x + tol))
                        {
                            dist2 = pow(two->pointCorrespondences.at(ipc).coordIm.x - two->keyPoints.at(ind[is]).pt.x, 2) +
                                    pow(two->pointCorrespondences.at(ipc).coordIm.y - two->keyPoints.at(ind[is]).pt.y, 2);
                            if(dist2 < mindist2)
                            {
                                mindist2 = dist2;
                                inn2 = ipc;
                                is2 = ind[is];
                            }
                        }
                    }

                    if(inn1 >= 0 and inn2 >= 0)
                    {
                        j1.push_back(inn1);
                        j2.push_back(inn2);
                        i1.push_back(is1);
                        i2.push_back(is2);
                    }

                    cout<<"j1.size() "<<j1.size()<<endl;
                    cout<<"j2.size() "<<j2.size()<<endl;
                }
            }

            matches1.resize(j1.size());
            matches2.resize(j2.size());

            cout<<"matches1.size() "<<matches1.size()<<endl;
            cout<<"matches2.size() "<<matches2.size()<<endl;

            for(unsigned int i = 0; i < j1.size(); i++)
            {
                matches1[i].resize(5);
                matches1[i][0] = one->pointCorrespondences.at(j1[i]).point3d.x;
                matches1[i][1] = one->pointCorrespondences.at(j1[i]).point3d.y;
                matches1[i][2] = one->pointCorrespondences.at(j1[i]).point3d.z;
                matches1[i][3] = one->keyPoints.at(i1[i]).pt.x;
                matches1[i][4] = one->keyPoints.at(i1[i]).pt.y;
                matches2[i].resize(5);
                matches2[i][0] = two->pointCorrespondences.at(j2[i]).point3d.x;
                matches2[i][1] = two->pointCorrespondences.at(j2[i]).point3d.y;
                matches2[i][2] = two->pointCorrespondences.at(j2[i]).point3d.z;
                matches2[i][3] = two->keyPoints.at(i2[i]).pt.x;
                matches2[i][4] = two->keyPoints.at(i2[i]).pt.y;
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
            ptpairs.clear();

            if(des1.size() == 0 || des2.size() == 0)
            {
                return;
            }

            float A[(int)des1.size()];
            float B[(int)des2.size()];

            int k = 0;
            k = min((int)des1.size(), (int)des2.size());

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
