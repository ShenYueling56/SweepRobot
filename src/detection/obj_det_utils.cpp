//
// Created by shenyl on 2020/11/1.
//

#include "detection/obj_det_utils.h"

cv::Mat Cal3D_2D(pcl::PointXYZ point3D, cv::Mat projectionMatrix, cv::Size imageSize) {
    cv::Mat point2D(2, 1, CV_16UC1, cv::Scalar(0));
    cv::Mat cur(4, 1, CV_64FC1, cv::Scalar(0));  //点云的三维齐次坐标向量4×1
    cur.at<double>(0) = point3D.x; // attention: double not float!!!!
    cur.at<double>(1) = point3D.y;
    cur.at<double>(2) = point3D.z;
    cur.at<double>(3) = 1.000000e+00;

//    cout<<"P1"<<projectionMatrix<<endl;
//    cout<<"cur"<<cur<<endl;

    cv::Mat Image_coord = projectionMatrix * cur;   //激光点云投影在图像上的齐次坐标向量3×1
    double Image_coord0 = Image_coord.at<float>(0);
    double Image_coord1 = Image_coord.at<float>(1);
    double Image_coord2 = Image_coord.at<float>(2);

    int pos_col = round(Image_coord.at<double>(0) / Image_coord.at<double>(2));  //齐次坐标转化为非齐次坐标
    int pos_row = round(Image_coord.at<double>(1) / Image_coord.at<double>(2));  //即除去一个尺度标量


//    cout<<"Image_coord"<<Image_coord<<endl;

    if (pos_col >= imageSize.width)
        pos_col = imageSize.width;
    if (pos_row >= imageSize.height)
        pos_row = imageSize.height;
    if (pos_col < 0)
        pos_col = 0;
    if (pos_row < 0)
        pos_row = 0;//去除超出图像范围的投影点云坐标
    point2D.at<int>(0) = pos_col;
    point2D.at<int>(1) = pos_row;
    return point2D;
}

//计算IoU---intersectionPercent
float intersectRect(const cv::Rect rectA, const cv::Rect rectB, cv::Rect &intersectRect) {
    if (rectA.x > rectB.x + rectB.width) { return 0.; }
    if (rectA.y > rectB.y + rectB.height) { return 0.; }
    if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
    if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
    float colInt = std::min(rectA.x + rectA.width, rectB.x + rectB.width) - std::max(rectA.x, rectB.x);
    float rowInt = std::min(rectA.y + rectA.height, rectB.y + rectB.height) - std::max(rectA.y, rectB.y);
    float intersection = colInt * rowInt;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float intersectionPercent = intersection / (areaA + areaB - intersection);

    intersectRect.x = std::max(rectA.x, rectB.x);
    intersectRect.y = std::max(rectA.y, rectB.y);
    intersectRect.width = std::min(rectA.x + rectA.width, rectB.x + rectB.width) - intersectRect.x;
    intersectRect.height = std::min(rectA.y + rectA.height, rectB.y + rectB.height) - intersectRect.y;
    return intersectionPercent;
}

bool transAABB2w(cv::Mat Tcw, AABB_box3d b3d, AABB_box3d &b3d_w) {
    cv::Mat cur(4, 1, CV_64FC1, cv::Scalar(0));
    cur.at<double>(0) = b3d._position_x; // attention: double not float!!!!
    cur.at<double>(1) = b3d._position_y;
    cur.at<double>(2) = b3d._position_z;
    cur.at<double>(3) = 1.000000e+00;
    cv::Mat Twc;
    invert(Tcw, Twc);
    cv::Mat world_coord = Twc * cur;
    b3d_w._position_x = world_coord.at<double>(0);
    b3d_w._position_y = world_coord.at<double>(1);
    b3d_w._position_z = world_coord.at<double>(2);
    b3d_w._width = b3d._width;
    b3d_w._length = b3d._length;
    b3d_w._height = b3d._height;
    b3d_w._score = b3d._score;
    b3d_w._c = b3d._c;
//    cout<<cur.at<double>(0)<<";"<<cur.at<double>(1)<<";"<<cur.at<double>(2)<<";"<<cur.at<double>(3)<<endl;
//    cout<<b3d_w._position_x<<";"<<b3d_w._position_y<<";"<<b3d_w._position_z<<" "<<world_coord.at<double>(3)<<endl;
    return true;
}