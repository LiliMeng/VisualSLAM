vector<float> to Mat 

  cv::Mat descriptors_1 = cv::Mat(one->descriptor.size()/64, 64, CV_32FC1);
 memcpy(descriptors_1.data, one->descriptor.data(), one->descriptor.size()*sizeof(float));

 cv::Mat descriptors_2 = cv::Mat(two->descriptor.size()/64, 64, CV_32FC1);
 memcpy(descriptors_2.data, two->descriptor.data(), two->descriptor.size()*sizeof(float));


Mat to vector<float>

  vector<float> descriptors1(descriptors_1.rows*descriptors_1.cols);
  vector<float> descriptors2(descriptors_2.rows*descriptors_2.cols);
