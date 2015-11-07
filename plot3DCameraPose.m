function plot3DCameraPose(x,y,z)
figure
%test = xlsread('camera_pose.xlsx', 'A2:C1665');
test = xlsread('camera_pose_Keyframe_MotionMetric0.1.xlsx', 'A1:C195');
x=test(:,1);
y=test(:,2);
z=test(:,3);

%plot3(x,y,z);
title('Raw Camera Pose from Keyframe Motion Metric Threshold 0.1');
scatter3(x,y,z,'filled')
xlabel('x position (meter)'); % x-axis label
ylabel('y position (meter)');% y-axis label
zlabel('z position (meter)');% z-axis label

end
