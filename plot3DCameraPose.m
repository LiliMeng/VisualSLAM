function plot3DCameraPose(x,y,z)
test = xlsread('camera_pose.xlsx', 'A2:C1665');
x=test(:,1);
y=test(:,2);
z=test(:,3);
figure
plot3(x,y,z)
end
