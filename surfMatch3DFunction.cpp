static void surfMatch3D(Surf3DImage * one,
                                Surf3DImage * two,
                                std::vector<std::vector<float> > & matches1,
                                std::vector<std::vector<float> > & matches2)
        {
            Ptr<DescriptorMatcher> matcher;
            matcher = DescriptorMatcher::create("BruteForce"); //If using BruteForce, change the "FlannBased" to "BruteForce"

            std::vector< cv::DMatch > matchesRaw;

            cv::Mat descriptors_1 = Mat(one->descriptor.size()/64, 64, CV_32FC1);
            memcpy(descriptors_1.data, one->descriptor.data(), one->descriptor.size()*siezeof(float));

            cv::Mat descriptors_2 = Mat(two->descriptor.size()/64, 64, CV_32FC1);
            memcpy(descriptors_2.data, two->descriptor.data(), two->descriptor.size()*siezeof(float));

            matcher->match(descriptors_1, descriptors_2, matchesRaw);

            cout<<"matchesRaw.size()  "<<matchesRaw.size()<<endl;

            // compute homography using RANSAC
            cv::Mat mask;
            int ransacThreshold=9;

            vector<Point2d> imgpts1beforeRANSAC, imgpts2beforeRANSAC;

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


            vector<vector<float> > matches1(matches.size()), matches2(matches.size());

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
