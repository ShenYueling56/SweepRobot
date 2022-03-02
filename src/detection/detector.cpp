//
// Created by shenyl on 2020/10/28.
//
#include "detection/detector.h"

#ifdef RKNN_MODEL
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <dlfcn.h>
#include "rknn_api.h"
#include "postprocess.h"
#include <fstream>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <typeinfo>
#endif


int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image

Detector::Detector(string config_file) {
    //从参数文件中读取相机参数
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    fs_read["cameraMatrixL"] >> cameraMatrix_L;
    fs_read["distCoeffsL"] >> distCoeffs_L;
    fs_read["cameraMatrixR"] >> cameraMatrix_R;
    fs_read["distCoeffsR"] >> distCoeffs_R;
    fs_read["R"] >> R;
    fs_read["T"] >> T;
    fs_read["imageSize"] >> imageSize;
    fs_read["Tcw"] >> Tcw;
    // 图片保存地址
    fs_read["save_path"] >> save_path;
    // 障碍物mask参数
//    fs_read["iou_offset"] >> iou_offset;
    fs_read["h0"] >> h0;
    //深度图mask参数
    fs_read["d0"] >> d0;
    //是否只进行障碍物检测
    fs_read["only_obs_det"] >> only_obs_det;
    //wls参数
    fs_read["wls_lamda"] >> wls_lamda;
    fs_read["wls_Sigma"] >> wls_Sigma;
    //障碍物mask参数
    bool use_iou;
    fs_read["use_iou"] >> use_iou;
    if (use_iou) {
        fs_read["v0"] >> v0;
//        v0 = (int)(cameraMatrix_L.at<double>(1, 2));
    } else
        v0 = 0;
    //障碍物轮廓提取参数
    fs_read["w_offset"] >> w_offset;
    fs_read["h_offset"] >> h_offset;
    fs_read["min_area"] >> min_area;

    //yolo网络参数
    string cfg_path, classes_path, weight_path;
    fs_read["cfg_path"] >> cfg_path;
    fs_read["classes_path"] >> classes_path;
    fs_read["weight_path"] >> weight_path;

    //距离尺度矫正参数
    fs_read["scale_offset"] >> scale_offset;
    fs_read["w1"] >> w1;
    fs_read["w2"] >> w2;
    fs_read["offset_x"] >> offset_x;
    fs_read["offset_y"] >> offset_y;
    fs_read["offset_z"] >> offset_z;

    // 物体检测范围参数
    fs_read["obj_min_d"] >> obj_min_d;
    fs_read["obj_max_d"] >> obj_max_d;
    //可通行障碍物最低高度
    fs_read["obs_min_h"] >> obs_min_h;
    //毛毯距离地面最低高度和最高高度
    fs_read["min_carpet_h"] >> min_carpet_h;
    fs_read["max_carpet_h"] >> max_carpet_h;
    fs_read.release();

    //计算双目立体矫正参数
    _stereoRectification();

    h = Tcw.at<double>(1, 3);
    b = T.at<double>(0);
    b = -b / 1000;

    // Load the network
    std::ifstream classNamesFile(classes_path.c_str());
    if (classNamesFile.is_open()) {
        std::string className = "";
        while (std::getline(classNamesFile, className)) {
            classes.push_back(className);
        }
    } else {
        std::cout << "can not open classNamesFile" << std::endl;
    }
    net = cv::dnn::readNetFromDarknet(cfg_path, weight_path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

#ifdef save_image
    //双目矫正结果存放目录
    left_rectified_path = save_path + "rectified/left/";
    right_rectified_path = save_path + "rectified/right/";
    pairs_rectified_path = save_path + "rectified/pairs/";
    //视差图结果存放目录
    disparities_path = save_path + "disparities/";
    filtered_disparities_path = save_path + "filtered_disparities/";
    //障碍物mask结果存放目录
    obstacle_mask_path = save_path + "obstacle_mask/";
    //深度图和轮廓图
    depth_image_path = save_path+"depth_image/";
    edge_image_path = save_path+"edge_image/";
    depth_mask_path = save_path+"depth_mask/";
    bbox_image_path = save_path + "bbox_image/";
    //yolo结果存放目录
    yolo_image_path = save_path + "yolo_image/";
    //carpet mask结果存放目录
    carpet_mask_path = save_path + "carpet_mask/";

    //新建结果文件夹
    string command = "mkdir -p " + left_rectified_path + " " + right_rectified_path + " " + pairs_rectified_path;
    system(command.c_str());
    command = "mkdir -p " + disparities_path + " " + filtered_disparities_path;
    system(command.c_str());
    command = "mkdir -p " + obstacle_mask_path;
    system(command.c_str());
    command = "mkdir -p " + depth_image_path + " " + edge_image_path + " " + depth_mask_path + " " + bbox_image_path;
    system(command.c_str());
    command = "mkdir -p " + yolo_image_path;
    system(command.c_str());
    command = "mkdir -p " + carpet_mask_path;
    system(command.c_str());


#endif
}

bool Detector::_stereoRectification() {
//    cout<<cameraMatrix1<<endl<<distCoeffs1<<endl<<cameraMatrix2<<endl<<distCoeffs2<<endl<<imageSize<<endl<<R <<endl<<T<<endl;
//    flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
//    alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
//    alpha 参数必须设置为0，否则图像可能为倒像？？？
    stereoRectify(cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R, imageSize,
                  R, T, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);
    cout << "R1:" << endl;
    cout << R1 << endl;
    cout << "R2:" << endl;
    cout << R2 << endl;
    cout << "P1:" << endl;
    cout << P1 << endl;
    cout << "P2:" << endl;
    cout << P2 << endl;
    cout << "Q:" << endl;
    cout << Q << endl;
    initUndistortRectifyMap(cameraMatrix_L, distCoeffs_L, R1, P1, imageSize, CV_32FC1, mapl1, mapl2);
    initUndistortRectifyMap(cameraMatrix_R, distCoeffs_R, R2, P2, imageSize, CV_32FC1, mapr1, mapr2);
    return true;
}

bool Detector::_rectification(const cv::Mat img_l, const cv::Mat img_r, const int frame) {
    if (img_l.empty() | img_r.empty()) {
        cout << img_l << " , " << img_r << " is not exist" << endl;
    }

    imageSize.width = img_l.cols;
    imageSize.height = img_l.rows;
    cv::Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC3);
    cv::Mat canLeft = canvas(cv::Rect(0, 0, imageSize.width, imageSize.height));
    cv::Mat canRight = canvas(cv::Rect(imageSize.width, 0, imageSize.width, imageSize.height));

    remap(img_l, rectified_l, mapl1, mapl2, cv::INTER_LINEAR);
    remap(img_r, rectified_r, mapr1, mapr2, cv::INTER_LINEAR);


    rectified_l.copyTo(canLeft);
    rectified_r.copyTo(canRight);
    rectangle(canLeft, validRoi[0], cv::Scalar(255, 255, 255), 5, 8);
    rectangle(canRight, validRoi[1], cv::Scalar(255, 255, 255), 5, 8);
    for (int j = 0; j <= canvas.rows; j += 16)
        line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);

#ifdef save_image
    char file_name[200];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(left_rectified_path + file_name, rectified_l);
    imwrite(right_rectified_path + file_name, rectified_r);
    imwrite(pairs_rectified_path + file_name, canvas);
#endif


#ifdef show_image
    imshow("rectified_l", rectified_l);
    imshow("rectified_r", rectified_r);
    imshow("canLeft", canvas);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif
    return true;

}

bool Detector::_filterDisparityImage(const int frame) {
    //将矫正后的图片转化为灰度图
    cv::Mat rectified_l_grey, rectified_r_grey;
    cvtColor(rectified_l, rectified_l_grey, cv::COLOR_BGR2GRAY);
    cvtColor(rectified_r, rectified_r_grey, cv::COLOR_BGR2GRAY);
    cv::Mat disp_roi;
    cv::Mat filtered_Disp;
    cv::Mat rectified_l_roi(rectified_l_grey, cv::Rect(0, v0, rectified_l_grey.cols, rectified_l_grey.rows - v0));
    cv::Mat rectified_r_roi(rectified_r_grey, cv::Rect(0, v0, rectified_r_grey.cols, rectified_r_grey.rows - v0));
//    cv::Mat rectified_l_roi(rectified_l, cv::Rect(0, v0-iou_offset, rectified_l.cols, rectified_l.rows-v0+iou_offset));
//    cv::Mat rectified_r_roi(rectified_r, cv::Rect(0, v0-iou_offset, rectified_r.cols, rectified_r.rows-v0+iou_offset));

//    利用sgbm计算左右图的视差图
//    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//    cv::Ptr<cv::StereoSGBM> sgbm;
//    sgbm = cv::StereoSGBM::create(
//            0, 160, 8, 8 * 8 * 8, 32 * 8 * 8, 1, 1, 10, 200, 200, cv::StereoSGBM::MODE_SGBM);
//    cv::Ptr<cv::StereoMatcher> right_sgbm = cv::ximgproc::createRightMatcher(sgbm);
//    sgbm->compute(rectified_l_roi, rectified_r_roi, disparity_l_roi); // 计算视差图
//    right_sgbm->compute(rectified_r_roi, rectified_l_roi, disparity_r_roi);
//    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//    cout<<"sgbm计算左右视差图时间: "<<t<<endl;


    //用BM计算视差图
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
    cv::Rect roi1, roi2;
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
    bm->compute(rectified_l_roi, rectified_r_roi, disparity_l_roi);
    cv::Ptr<cv::StereoMatcher> right_bm = cv::ximgproc::createRightMatcher(bm);
    right_bm->compute(rectified_r_roi, rectified_l_roi, disparity_r_roi);
    disp_roi = disparity_l_roi;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//    cout<<"bm计算左右视差图时间: "<<t<<endl;

    //对左视差图进行最小二乘滤波
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilter(bm);   //sgbm
    wls_filter->setLambda(wls_lamda);
    wls_filter->setSigmaColor(wls_Sigma);
    wls_filter->filter(disparity_l_roi, rectified_l_roi, disp_roi, disparity_r_roi);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
//    cout<<"最小二乘滤波时间: "<<t<<endl;

    disp_roi.convertTo(filtered_disparity, CV_32F, 1.0 / 16);
    disparity_l_roi.convertTo(disparity_l_roi, CV_32F, 1.0 / 16);
//    disparity_r_roi.convertTo(disparity_r_roi, CV_32F, 1.0 / 16);
    cv::Mat rectified_l_roi_norm, disparity_r_roi_norm, filtered_disparity_norm;
    normalize(disparity_l_roi, rectified_l_roi_norm, 0, 256, cv::NORM_MINMAX, CV_8U);
//    normalize(disparity_r_roi, disparity_r_roi_norm, 0, 256, cv::NORM_MINMAX, CV_8U);
    normalize(filtered_disparity, filtered_disparity_norm, 0, 256, cv::NORM_MINMAX, CV_8U);
#ifdef show_image
    imshow("disparity_l_roi", rectified_l_roi_norm);
//    imshow("disparity_r_roi", disparity_r_roi_norm);
    imshow("filtered_disparity", filtered_disparity_norm);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif

#ifdef save_image
    char file_name[30];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(disparities_path + file_name, rectified_l_roi_norm);
    imwrite(filtered_disparities_path + file_name, filtered_disparity_norm);
#endif
    return true;
}

bool Detector::_get_obstacle_mask(const int frame, double min_obs_h) {
    obstacle_mask = cv::Mat(rectified_l.rows, rectified_l.cols, CV_8UC1, cv::Scalar::all(255));
    carpet_mask = cv::Mat(rectified_l.rows, rectified_l.cols, CV_8UC1, cv::Scalar::all(255));

    for (int v = v0; v < v0 + filtered_disparity.rows; v++) {
        double gd_disparity = b * (v - (int) (cameraMatrix_L.at<double>(1, 2))) / (h - min_obs_h);//比地面高2cm作为障碍物
        //为了找到毛毯，或许距离地面min_carpet_h ~ max_carpet_h部分的障碍物像素，得到毛毯mask
        double min_carpet_disparity = b * (v - (int) (cameraMatrix_L.at<double>(1, 2))) / (h - min_carpet_h);
        double max_carpet_disparity = b * (v - (int) (cameraMatrix_L.at<double>(1, 2))) / (h - max_carpet_h);
        for (int u = 0; u < filtered_disparity.cols; u++) {
            float d = filtered_disparity.at<float>(v - v0, u);
            if (d < gd_disparity) {
                obstacle_mask.at<unsigned char>(v, u) = 0;
            }
//            if ((d>min_carpet_disparity)&&(d<max_carpet_disparity))
            if (d < min_carpet_disparity) {
                carpet_mask.at<unsigned char>(v, u) = 0;
            }
        }
    }
    //对障碍物mask进行腐蚀膨胀
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3), cv::Point(-1, -1));
    erode(obstacle_mask, obstacle_mask, element);
    dilate(obstacle_mask, obstacle_mask, element);
    erode(carpet_mask, carpet_mask, element);
    dilate(carpet_mask, carpet_mask, element);

#ifdef save_image
    char obstacle_mask_file[20];
    sprintf(obstacle_mask_file, "%06d.jpg", frame);
    imwrite(obstacle_mask_path+obstacle_mask_file, obstacle_mask);
    imwrite(carpet_mask_path+obstacle_mask_file, carpet_mask);
#endif
#ifdef show_image
    imshow("obstacle_mask", obstacle_mask);
    imshow("carpet_mask", carpet_mask);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif
    return true;
}

bool Detector::_get_obstacle_depth_image(int frame) {
    cv::Mat depth_image_iou = cv::Mat(image3D_iou.rows, image3D_iou.cols, CV_32FC1,
                                      cv::Scalar::all(0)); //对于不在roi内的点，给较大深度值
    //将3D图的第三维作为深度图
    for (int i = 0; i < image3D_iou.rows; i++) {
        for (int j = 0; j < image3D_iou.cols; j++) {
            depth_image_iou.at<float>(i, j) = image3D_iou.at<cv::Vec3f>(i, j)[2];
        }
    }
    //归一化深度图用于显示
    depth_image_iou.copyTo(depth_image(cv::Rect(0, v0, rectified_l.cols, rectified_l.rows - v0)));
    cv::Mat depth_image_normed;
    normalize(depth_image, depth_image_normed, 0, 256, cv::NORM_MINMAX, CV_8U);

    // 对深度图进行轮廓提取
    cv::Mat edge_image;
    cv::Canny(depth_image_normed, edge_image, 100, 200);

    float min_d = d0 * 1000;
    // 对深度图小于d0m的部分提取mask
    for (int i = 0; i < depth_image.rows; i++) {
        for (int j = 0; j < depth_image.cols; j++) {
            if (depth_image.at<float>(i, j) < min_d) {
                depth_mask.at<uchar>(i, j) = 255;
            }
        }
    }
#ifdef show_image
    imshow("depth_image", depth_image_normed);
    imshow("EdgeImage", edge_image);
    imshow("depth_mask", depth_mask);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif
#ifdef save_image
    char file_name[20];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(depth_image_path+file_name, depth_image_normed);
    imwrite(edge_image_path+file_name, edge_image);
    imwrite(depth_mask_path+file_name, depth_mask);
#endif
    return true;
}

bool Detector::_get_obstacle_points(const int frame) {
    int index = 0;
    points_obs_ptr->width = image3D_obs.cols * image3D_obs.rows;
    points_obs_ptr->height = 1;
    points_obs_ptr->is_dense = false;  //不是稠密型的
    points_obs_ptr->points.resize(points_obs_ptr->width * points_obs_ptr->height);  //点云总数大小
    for (int u = 0; u < image3D_obs.cols; u++) {
        for (int v = 0; v < image3D_obs.rows; v++) {
            cv::Point point;
            point.x = u;
            point.y = v;
            if (image3D_obs.at<cv::Vec3f>(point)[2] < 450 | image3D_obs.at<cv::Vec3f>(point)[2] > 2000)
                continue;
            points_obs_ptr->points[index].x = image3D_obs.at<cv::Vec3f>(point)[0] / 1000;
            points_obs_ptr->points[index].y = image3D_obs.at<cv::Vec3f>(point)[1] / 1000;
            points_obs_ptr->points[index++].z = image3D_obs.at<cv::Vec3f>(point)[2] / 1000;
        }
    }

    //voxel downsample //!!!!voxel之后会出现一些在零点的点，直接进行聚类会有错误
    if (points_obs_ptr->points.size() > 1000) {
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud(points_obs_ptr);
        vox.setLeafSize(0.002f, 0.002f, 0.005f);
        vox.filter(*points_obs_ptr);
    }

    if (points_obs_ptr->points.size() < 20) return false;

    // filtering cloud by passthrough
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(points_obs_ptr);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.45, 2.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter(*points_obs_ptr);
    if (points_obs_ptr->points.size() < 20) return false;

#ifdef ros_points_pub
    //    sensor_msgs::PointCloud2 Obj_Point_msg;
    //    pcl::toROSMsg(*pc_object_filtered_ptr, Obj_Point_msg);
    //    Obj_Point_msg.header.stamp = ros::Time::now();
    //    Obj_Point_msg.header.frame_id = "map";
    //    obj_points_pub_.publish(Obj_Point_msg);
#endif
    return true;
}

bool Detector::_get_obstacle_boxes(int frame) {
    //计算凸包
//    cout << "计算轮廓....." << endl;
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    findContours(depth_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    /// 对每个轮廓计算其凸包
    vector<vector<cv::Point> > hull(contours.size());
    vector<cv::Rect> bbox_obs;
    for (int i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) < min_area)//面积小于area的凸包，可忽略
            continue;
        cv::convexHull(cv::Mat(contours[i]), hull[i], false);
        cv::Rect r = cv::boundingRect(contours[i]);
        if ((r.x < 0) || ((r.x + r.width) > rectified_l.cols))
            continue;
        if ((r.y < 0) || ((r.y + r.height) > rectified_l.rows))
            continue;
        bbox_obs.push_back(r);
        //draw
        drawContours(filtered_disparity, hull, i, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
        rectangle(filtered_disparity, r, cv::Scalar(255, 0, 0), 2);
    }
#ifdef show_image
    imshow("hull", rectified_l);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif
#ifdef save_image
    char file_name[20];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(bbox_image_path+file_name, filtered_disparity);
#endif

    //统计轮廓内的所有点的3d坐标，求xy平均值和z轴最小值
    for (int i = 0; i < bbox_obs.size(); i++) {
        //障碍物位置
        float ave_x = 0;
        float ave_y = 0;
        float ave_z = 0;
        float min_z = 1000;
        //障碍物尺寸
        float min_h = 10000;
        float max_h = -10000;
        float min_w = 10000;
        float max_w = -10000;
        int count = 0;
        //图像包络框端点
        int minx = bbox_obs[i].x;
        int miny = bbox_obs[i].y;
        int maxx = bbox_obs[i].x + bbox_obs[i].width;
        int maxy = bbox_obs[i].y + bbox_obs[i].height;

        for (int v = miny; v < maxy; v++) {
            for (int u = minx; u < maxx; u++) {
                if (depth_image.at<float>(v, u) > d0 * 1000)
                    continue;
                ave_x = ave_x + image3D_obs.at<cv::Vec3f>(v, u)[0];
                ave_y = ave_y + image3D_obs.at<cv::Vec3f>(v, u)[1];
                ave_z = ave_z + image3D_obs.at<cv::Vec3f>(v, u)[2];
                if (image3D_obs.at<cv::Vec3f>(v, u)[2] < min_z) {
                    min_z = image3D_obs.at<cv::Vec3f>(v, u)[2];
                }

                //计算轮廓
                if (((u - minx) < w_offset) && (image3D_obs.at<cv::Vec3f>(v, u)[0] < min_w)) {
                    min_w = image3D_obs.at<cv::Vec3f>(v, u)[0];
                }
                if (((maxx - u) < w_offset) && (image3D_obs.at<cv::Vec3f>(v, u)[0] > max_w)) {
                    max_w = image3D_obs.at<cv::Vec3f>(v, u)[0];
                }
                if (((v - miny) < h_offset) && (image3D_obs.at<cv::Vec3f>(v, u)[1] < min_h)) {
                    min_h = image3D_obs.at<cv::Vec3f>(v, u)[1];
                }
                if (((maxy - v) < h_offset) && (image3D_obs.at<cv::Vec3f>(v, u)[1] > max_h)) {
                    max_h = image3D_obs.at<cv::Vec3f>(v, u)[1];
                }
                count = count + 1;
            }
        }
        if (count == 0)
            continue;
        if (max_h < min_h)
            continue;
        if (max_w < min_w)
            continue;
        //如果障碍物最低点高于阈值，忽略该障碍物
//        cout<<"max_h "<<max_h<<endl;
        if (max_h < obs_min_h)
            continue;
        ave_x = (ave_x / count) / 1000;
        ave_y = (ave_y / count) / 1000;
        ave_z = (ave_z / count) / 1000;
        ave_x = ave_x / scale_offset - offset_x;
        ave_y = ave_y / scale_offset - offset_y;
        ave_z = ave_z / scale_offset - offset_z;
        float height = (max_h - min_h) / 1000;
        float width = (max_w - min_w) / 1000;
        AABB_box3d aabb_box3d(ave_x, ave_y, ave_z, width, -1, height);
        AABB_box3d aabb_box3d_w;
        transAABB2w(Tcw, aabb_box3d, aabb_box3d_w);
        obs_boxes.push_back(aabb_box3d_w);
    }
    return true;
}

#ifdef RKNN_MODEL
static int GetElementByte(rknn_tensor_attr *in_attr)
{
    int byte = 0;
    switch (in_attr->type)
    {
        case RKNN_TENSOR_FLOAT32:
            byte = 4;
//            cout << "model input RKNN_TENSOR_FLOAT32" << endl;
            break;
        case RKNN_TENSOR_FLOAT16:
        case RKNN_TENSOR_INT16:
            byte = 2;
//            cout << "model input RKNN_TENSOR_FLOAT16 OR RKNN_TENSOR_INT16" << endl;
            break;
        case RKNN_TENSOR_INT8:
        case RKNN_TENSOR_UINT8:
            byte = 1;
//            cout << "model input RKNN_TENSOR_UINT8 OR RKNN_TENSOR_INT8" << endl;
            break;
        default:
            break;
    }
    return byte;
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    // return (int32_t)((f > 0.0) ? (f + 0.5) : (f - 0.5));    //四舍五入
    return f;
}

void qnt_f32_to_affine(uint8_t *qnt, uint8_t type, uint8_t zp, float scale,
                       float *f32, int num)
{
    float *src_ptr = f32;
    int i = 0;
    float dst_val = 0.0;

//    RKNN_TENSOR_UINT8
    for (; i < num; i++)
    {
        dst_val = ((*src_ptr) / scale) + zp;
        *qnt = (uint8_t)__clip(dst_val, 0, 255);
        src_ptr++;
        qnt++;
    }
}

void qnt_f32_to_dfp(uint8_t *qnt, uint8_t type, int8_t fl, float *f32,
                    int num)
{
    float *src_ptr = f32;
    int i = 0;
    float dst_val = 0.0;

    switch (type)
    {
        case RKNN_TENSOR_INT8:
            for (; i < num; i++)
            {
                dst_val = (fl > 0) ? ((*src_ptr) * ((float)(1 << fl)))
                                   : ((*src_ptr) / (float)(1 << -fl));
                *((int8_t *)qnt) = (int8_t)__clip(dst_val, -128, 127);
                src_ptr++;
                qnt++;
            }
            break;
        case RKNN_TENSOR_UINT8:
            for (; i < num; i++)
            {
                dst_val = (fl > 0) ? ((*src_ptr) * ((float)(1 << fl)))
                                   : ((*src_ptr) / (float)(1 << -fl));
                *qnt = (uint8_t)__clip(dst_val, 0, 255);
                src_ptr++;
                qnt++;
            }
            break;
        case RKNN_TENSOR_INT16:
            for (; i < num; i++)
            {
                dst_val = (fl > 0) ? ((*src_ptr) * ((float)(1 << fl)))
                                   : ((*src_ptr) / (float)(1 << -fl));
                *((int16_t *)qnt) = (int16_t)__clip(dst_val, -32768, 32767);
                src_ptr++;
                qnt += 2;
            }
            break;
        default:
            break;
    }
}

void qnt_f32_to_none(uint8_t *qnt, uint8_t type, float *f32, int num)
{
    float *src_ptr = f32;
    int i = 0;

    switch (type)
    {
        case RKNN_TENSOR_INT8:
            for (; i < num; i++)
            {
                *((int8_t *)qnt) = (int8_t)__clip(*src_ptr, -128, 127);
                src_ptr++;
                qnt++;
            }
            break;
        case RKNN_TENSOR_UINT8:
            for (; i < num; i++)
            {
                *qnt = (uint8_t)__clip(*src_ptr, 0, 255);
                src_ptr++;
                qnt++;
            }
            break;
        case RKNN_TENSOR_INT16:
            for (; i < num; i++)
            {
                *((int16_t *)qnt) = (int16_t)__clip(*src_ptr, -32768, 32767);
                src_ptr++;
                qnt += 2;
            }
            break;
        default:
            break;
    }
}

void f32_to_f16(uint16_t *f16, float *f32, int num)
{

    float *src = f32;
    uint16_t *dst = f16;
    int i = 0;

    for (; i < num; i++)
    {
        float in = *src;

        uint32_t fp32 = *((uint32_t *)&in);
        uint32_t t1 = (fp32 & 0x80000000u) >> 16; /* sign bit. */
        uint32_t t2 = (fp32 & 0x7F800000u) >> 13; /* Exponent bits */
        uint32_t t3 = (fp32 & 0x007FE000u) >> 13; /* Mantissa bits, no rounding */
        uint32_t fp16 = 0u;

        if (t2 >= 0x023c00u)
        {
            fp16 = t1 | 0x7BFF; /* Don't round to infinity. */
//            cout << "toosmalltoosmalltoosmalltoosmalltoosmall" << endl;
        }
        else if (t2 <= 0x01c000u)
        {
            fp16 = t1;
//            cout << "toobigtoobigtoobigtoobigtoobigtoobig" << endl;
        }
        else
        {
            t2 -= 0x01c000u;
            fp16 = t1 | t2 | t3;
        }
        //    cout << "norm img data: " << fp16 << endl;
        *dst = (uint16_t)fp16;

        src++;
        dst++;
    }

//    cout << "dst data: " << dst << endl;
//    cout << "*dst data: " << *dst << endl;


}

//uint16_t *input
static int saveInput(const char *file_name, float *array,  int grid_h, int grid_w)          //float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "a+");
    printf("start    saving     inputs\n");
    int tensor_size = grid_h * grid_w * 3;       //(5+4)*3
    printf("tensor size: %d\n", tensor_size);

    for (int i = 0;i < tensor_size; i++)
    {
        fprintf(fp, "%.6f ", array[i]);
        fprintf(fp, "\n");
    }
//    for (int i = 0; i < grid_h; i++)
//    {
//        for (int j = 0; j < grid_w; j++)
//        {
//            fprintf(fp, "%.6f ", (img.at<Vec3f>(i,j)[2]));
//            fprintf(fp, "\n");
//        }
//        fprintf(fp, "%.6f ", deqnt_affine_to_f32(input[i], zp, scale));
//         fprintf(fp, "%d ", unsigned(input[i]));
//
//    }

    printf("finish    saving     outputs\n");
    fclose(fp);
    return 0;
}

//int Detector::ProcessInput(cv::Mat &img, void **dst_buf, rknn_tensor_attr *in_attr,
//                 bool isReorder210, bool isNCHW)

int Detector::ProcessInput(cv::Mat img, void **dst_buf, rknn_tensor_attr *in_attr,
                           bool isReorder210, bool isNCHW)
{
    cv::Mat norm_out, norm_nchw;
    // RGB2BGR
    if (isReorder210 == true)
    {
//        printf("perform RGB2BGR\n");
        cv::cvtColor(img, norm_out, cv::COLOR_BGR2RGB);
    }
    norm_out.convertTo(norm_out, CV_32FC3);
//    img.copyTo(norm_out);

    int HW = img.cols * img.rows;
    int C = img.channels();
    int ele_count = HW * C;
    int ele_bytes = GetElementByte(in_attr);
//    printf("total element count = %d, bytes per element = %d\n", ele_count, ele_bytes);

    // normalize
//    printf("perform normalize\n");

//    img.convertTo(norm_out, CV_32F);
//    img.copyTo(norm_out);

    float norm_num = 255.0f;
    norm_out = norm_out / norm_num;

//    for (int i=0;i<32;i++)
//    {
//        int a = img.at<Vec3b>(80, i)[0];
//        cout << "norm_out00:  " << i << "    " << a << endl;
//    }

    bool transform_by_myself = false;
    if (isNCHW && transform_by_myself)
    {

        norm_out.copyTo(norm_nchw);

//        int count=0;
//        int H = img.rows;
//        int W = img.cols;

        //toNCHW
//        for (int i = 0; i < H; i++) //H
//        {
//            for (int j = 0; j < W; j++) //W
//            {
//                for (int k = 0; k < C; k++) //C
//                {
//                    int ii,jj,kk;
//                    kk = count / (H*W);
//                    jj = (count - kk*H*W)/W;
//                    ii = (count - kk*H*W - jj*W);
//                    float value = norm_out.at<Vec3f>(j, i)[k];
//                    norm_nchw.at<Vec3f>(ii, jj)[kk] = value;
//                    count++;
//
//                }
//
//            }
//        }
//        norm_out = norm_nchw.clone();

        int ii = 0;
        int jj = 0;
        int kk = 0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 640; j++)
            {
                for (int k = 0; k < 640; k++)
                {
                    float value = norm_out.at<Vec3f>(j, k)[i];
                    norm_nchw.at<Vec3f>(ii, jj)[kk] = value;
                    kk += 1;
                    if (kk == 3)
                    {
                        kk = 0;
                        jj += 1;
                        if (jj == 640)
                        {
                            jj = 0;
                            ii += 1;
                        }
                    }
                }

            }
        }

        norm_out = norm_nchw.clone();
        ele_bytes = 4;
    }


//    saveInput(INPUTS_PATH, norm_out, 640, 640);


    // quantize
//    printf("perform quantize\n");
//    uint8_t
    qnt_buf = (uint8_t *)malloc(ele_count * ele_bytes);

//    float *qnt_buf;
//    qnt_buf = (float *)malloc(ele_count * ele_bytes);
//    printf("float pointer\n");

    if (in_attr->type == RKNN_TENSOR_FLOAT16 && transform_by_myself)
    {
        memcpy(qnt_buf, norm_out.data, ele_count * ele_bytes);
        cout << "quantize RKNN_TENSOR_FLOAT32" << endl;
    }
    else if(in_attr->type == RKNN_TENSOR_FLOAT16 && !transform_by_myself)
    {
//        cout << "quantize RKNN_TENSOR_FLOAT16" << endl;
        if (transform_by_myself){
            memcpy(qnt_buf, norm_out.data, ele_count * ele_bytes);
//        cout << "norm_out.data: " << norm_out.data << endl;
//            cout << "(float *)norm_out.data:  " << (float *)norm_out.data << endl;
//        uint16_t
//            cout << "qnt_buf after:   " << qnt_buf << endl;
//            cout << "*qnt_buf after:  " << *qnt_buf << endl;
            cout << endl;
        }
        else{
            f32_to_f16((uint16_t *)(qnt_buf), (float *)(norm_out.data), ele_count);
        }

    }
    else if(in_attr->type == RKNN_TENSOR_UINT8 || in_attr->type == RKNN_TENSOR_INT8 || in_attr->type == RKNN_TENSOR_INT16)
    {
        if(in_attr->qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
        {
            cout << "quantize RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC" << endl;
//            qnt_f32_to_affine(qnt_buf, in_attr->type, in_attr->zp, in_attr->scale,
//                              (float *)(norm_out.data), ele_count);
        }
        else if(in_attr->qnt_type == RKNN_TENSOR_QNT_DFP)
        {
            cout << "quantize RKNN_TENSOR_QNT_DFP" << endl;
//            qnt_f32_to_dfp(qnt_buf, in_attr->type, in_attr->fl,
//                           (float *)(norm_out.data), ele_count);
        }
        else if(in_attr->qnt_type == RKNN_TENSOR_QNT_NONE)
        {
            cout << "quantize RKNN_TENSOR_QNT_NONE" << endl;
//            qnt_f32_to_none(qnt_buf, in_attr->type, (float *)(norm_out.data),
//                            ele_count);
        }
    }

    // NHWC ==> NCHW
//    uint8_t
//    saveInput("/data/ttt/model_input_data.txt", qnt_buf, 640, 640);

    nchw_buf = (uint8_t *)malloc(ele_count * ele_bytes);
    if (transform_by_myself){
        *dst_buf = qnt_buf;
    }


    if (isNCHW && (!transform_by_myself))
    {
//        printf("perform NHWC to NCHW\n");
//        nchw_buf = (uint8_t *)malloc(ele_count * ele_bytes);
        uint8_t *dst_ptr = nchw_buf;
        uint8_t *src_ptr = qnt_buf;
        for (int i = 0; i < C; ++i)
        {
            src_ptr = qnt_buf + i * ele_bytes;
            dst_ptr = nchw_buf + i * HW * ele_bytes;
            for (int j = 0; j < HW; ++j)
            {
                // dst_ptr[i*HW+j] = src_ptr[j*C+i];
                memcpy(dst_ptr, src_ptr, ele_bytes);
                src_ptr += C * ele_bytes;
                dst_ptr += ele_bytes;
            }
        }
        *dst_buf = nchw_buf;
//        printf("Done NHWC to NCHW\n");

//        free(qnt_buf);
//        printf("free\n");
    }
    else
    {
        printf("perform No Changing Channels\n");
        *dst_buf = qnt_buf;
    }

    return 0;
}
#endif

bool Detector::obstacle_det(const cv::Mat img_l, const cv::Mat img_r, const int frame) {
    int img_height = int(imageSize.height);
    int img_width = int(imageSize.width);
    int img_roi_height = img_height - v0;
    int img_roi_width = img_width;
    obstacle_mask = cv::Mat(img_height, img_width, CV_8UC1, cv::Scalar::all(255));
    masked_disparity_obs = cv::Mat(img_height, img_width, CV_32FC1, cv::Scalar::all(255));
    masked_disparity_roi_obs = cv::Mat(img_roi_height, img_roi_width, CV_32FC1, cv::Scalar::all(0));
//    image3D = cv::Mat(img_height, img_width, CV_32FC3, cv::Scalar::all(0));
//    image3D_iou = cv::Mat(img_roi_height, img_roi_width, CV_32FC3, cv::Scalar::all(0));
    depth_image = cv::Mat(img_height, img_width, CV_32FC1, cv::Scalar::all(8918.82));//对于视差图空洞部分的深度默认较大
    depth_mask = cv::Mat(img_height, img_width, CV_8UC1, cv::Scalar::all(0));
    //初始化障碍物点云
    points_obs_ptr.reset(new pcl::PointCloud<pcl::PointXYZ>());

    //Step1 双目矫正
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    _rectification(img_l, img_r, frame);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//    cout<<"双目矫正时间: "<<t<<endl;
//    cout<<"完成双目矫正"<<endl;

    //Step2 利用双目矫正后的图像计算视差图，并进行最小二乘滤波
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    _filterDisparityImage(frame);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    t = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
//    cout<<"视差图提取时间: "<<t<<endl;
//    cout<<"完成视差图提取"<<endl;

    //Step3 得到障碍物mask
    double min_obs_h = h0;
    _get_obstacle_mask(frame, min_obs_h);

    //Step4 将障碍物mask和视差图相与，得到障碍物部分的视差图
    cv::Mat obstacle_mask_roi(obstacle_mask, cv::Rect(0, v0, obstacle_mask.cols, obstacle_mask.rows - v0));
    bitwise_and(filtered_disparity, filtered_disparity, masked_disparity_roi_obs, obstacle_mask_roi);
    masked_disparity_roi_obs.copyTo(masked_disparity_obs(cv::Rect(0, v0, rectified_l.cols, rectified_l.rows - v0)));
//    cout<<"得到障碍物mask和mask作用下的视差图"<<endl;

    //Step5 利用视差图和投影参数Q得到三维图像
    reprojectImageTo3D(masked_disparity_obs, image3D_obs, Q);
    image3D_iou = cv::Mat(image3D_obs, cv::Rect(0, v0, rectified_l.cols, rectified_l.rows - v0));

    //Step6 计算深度图,得到障碍物深度小于某个阈值的深度图
    _get_obstacle_depth_image(frame);
//    cout<<"计算3D图和深度图，并提取距离小于d0的深度图mask"<<endl;

    //Step7 计算障碍物轮廓, 计算各障碍物位置和尺寸
    obs_boxes.clear();
    _get_obstacle_boxes(frame);
//    cout<<"提取障碍物轮廓"<<endl;

    //Step8 通过3d img 得到障碍物点云
//    return _get_obstacle_points(frame);
//    std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
//    t = std::chrono::duration_cast<std::chrono::duration<double> >(t_last - t1).count();
//    cout<<"障碍物检测总体时间: "<<t<<endl;
    return true;
}

Detector::~Detector() {}

#ifdef RKNN_MODEL
int last_name_arr[5] = {0};
int current_name_arr[5] = {0};
int dog_shit_score;

//  void *drmBuf, void *resizeBuf,
bool Detector::_yolo_det_on_rv1126(cv::Mat img_l, int frame, rknn_context rknn_ctx,
                         int input_num, int output_num, int model_input_width, int model_input_height,
                         rknn_input *input_struct, rknn_tensor_attr *output_struct) {

//    cout << "Start yolo detection" << endl;
    auto start1 = std::chrono::steady_clock::now();
    bbox_list.clear();
//    obj_boxes.clear();
    /*
    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(img_l, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));
    */
    const float vis_threshold = 0.1;
    const float nms_threshold = 0.1;
    const float conf_threshold = 0.7;
    int ret, img_w, img_h;
    struct timeval start_time, stop_time;
    string class_name;
    int r,g,b;

    img_w = img_l.cols;
    img_h = img_l.rows;

    //loop
//    img_resize_slow(&rga, drmBuf, img_w, img_h, resizeBuf, model_input_width, model_input_height);

    gettimeofday(&start_time, NULL);
//    printf("Input set before\n");
    rknn_inputs_set(rknn_ctx, input_num, input_struct);
//    printf("Input set\n");

    rknn_output outputs[output_num];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < output_num; i++)
    {
        outputs[i].want_float = 1;          //obtain float directly (before is 0, which needs to dequantitation by myself)
    }

//    printf("Run yolo\n");
    ret = rknn_run(rknn_ctx, NULL);
//    printf("Obtain outputs\n");
    ret = rknn_outputs_get(rknn_ctx, output_num, outputs, NULL);
    gettimeofday(&stop_time, NULL);
//    printf("once run use %f ms\n",
//           (__get_us(stop_time) - __get_us(start_time)) / 1000);

    //post process
    float scale_w = (float)model_input_width / img_w;
    float scale_h = (float)model_input_height / img_h;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;
    for (int i = 0; i < output_num; ++i)
    {
        out_scales.push_back(output_struct[i].scale);
        out_zps.push_back(output_struct[i].zp);
    }

//    printf("Post processing\n");
    post_process((float *)outputs[0].buf, (float *)outputs[1].buf, (float *)outputs[2].buf,
                 model_input_height, model_input_width,conf_threshold, nms_threshold, vis_threshold, scale_w, scale_h,
                 out_zps, out_scales, &detect_result_group);

    // Draw Objects
    cv:: Mat img_clone = img_l.clone();

    int carpet_times = 0;

    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
//        printf("%d @ (%d %d %d %d) %f\n",
//               det_result->obj_name,
//               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
//               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

//        int continuous_frames = 2;
//
//        if (current_name_arr[det_result->obj_name] == 0)
//        {
//            current_name_arr[det_result->obj_name] += 1;
//        }
//
//        cout << "obj name:  " << det_result->obj_name << endl;
//        cout << "current: " << current_name_arr[det_result->obj_name] << endl;
//        cout << "last: " << last_name_arr[det_result->obj_name] << endl;
//        if ((last_name_arr[det_result->obj_name] + current_name_arr[det_result->obj_name]) < continuous_frames)
//        {
//            cout << "remove: " << det_result->obj_name << endl;
//            continue;
//        }

        if (det_result->obj_name == 0){
            class_name = "badminton";
            b = 255;
            g = 0;
            r = 0;
        }
        if (det_result->obj_name == 1){
            class_name = "wire";
            b = 255;
            g = 255;
            r = 0;
        }
        if (det_result->obj_name == 2){
            class_name = "mahjong";
            b = 255;
            g = 0;
            r = 255;
        }
        if (det_result->obj_name == 3){
            class_name = "dogshit";
            b = 0;
            g = 255;
            r = 255;
        }
        if (det_result->obj_name == 4){
            class_name = "carpet";
            b = 0;
            g = 255;
            r = 0;

//            carpet_times += 1;
        }

        rectangle(img_clone, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(b, g, r, 255), 3);
        putText(img_clone, class_name, cv::Point(x1, y1 - 10), 1, 2, cv::Scalar(255, 255, 255, 255),2);

//        int center_x = (x1+x2) / 2;
//        int center_y = (y1+y2) / 2;
//
//        char cor[20];
//        sprintf(cor, "cx:%d, cy:%d", center_x, center_y);
//        cv::putText(img_clone, cor, cv::Point(center_x, center_y), 1, 2, cv::Scalar(255, 255, 255), 2);


//        sprintf(text_y, "current_y: %0.6f", current_y);
//        putText(img_clone, text_y, cv::Point(point.x - 10, point.y + 10), 1, 2, cv::Scalar(0, 0, 255, 255),2);

        bbox box2d(det_result->obj_name, det_result->prop, x1, y1, x2, y2);
        bbox_list.push_back(box2d);

    }

//    AABB_box3d aabb_box3d(0, 0, 0, -1, -1, -1, carpet_times);
//    obj_boxes.push_back(aabb_box3d);

    cout << "Total number of objects detected per frame: " << detect_result_group.count << endl;
    if (detect_result_group.count > 0){
//        std::string save_img_path = "/data/ttt/img_result_yolo_det_new/out_" + std::to_string(frame) + ".jpg";
//        std::string save_img_path = save_path + "yolo_detection_result/out_" + std::to_string(frame) + ".jpg";
//        imwrite(save_img_path, img_clone);
    }
    else{
        cout << "No object detected!" << endl;
    }
    ret = rknn_outputs_release(rknn_ctx, output_num, outputs);

//#ifdef show_image
//    imshow("yolo_result", img_clone);
//    if (cv::waitKey(0) == 27) {
//        cv::destroyAllWindows();
//    }
//#endif
#ifdef save_image
    char file_name[20];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(yolo_image_path+file_name, img_clone);
#endif

    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed1 = end1 - start1; // std::micro 表示以微秒为时间单位
//    cout << "time for yolo detection: " << elapsed1.count() / 1000000 << " s" << endl;

    return true;
}

#else
bool Detector::_yolo_det(cv::Mat img_l, int frame) {
    auto start1 = std::chrono::steady_clock::now();
    bbox_list.clear();
    // Create a 4D blob from a frame.
    cv::Mat blob;
    cv::dnn::blobFromImage(img_l, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

    //Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(img_l, outs, bbox_list, classes);
#ifdef show_image
    imshow("yolo_result", img_l);
    if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif
#ifdef save_image
    char file_name[20];
    sprintf(file_name, "%06d.jpg", frame);
    imwrite(yolo_image_path+file_name, img_l);
#endif

    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed1 = end1 - start1; // std::micro 表示以微秒为时间单位
//    cout << "time for yolo detection: " << elapsed1.count() / 1000000 << endl;

    return true;
}
#endif

static double obj_height = 0;
bool Detector::_pick_cluster(bbox b, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr &object_cluster, double object_point_num, box3d &b3d) {
    vector<double> scores;
    double max_score = 0;
    double p_x = 0;
    double p_y = 0;
    double p_z = 0;

    for (int i = 0; i < pc_clusters.size(); i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster = pc_clusters[i];
        cv::Mat point2D_this(2, 1, CV_16UC1, cv::Scalar(0));
        cv::Mat point2D_min(2, 1, CV_16UC1, cv::Scalar(0));
        point2D_min.at<int>(0) = 10000;
        point2D_min.at<int>(1) = 10000;
        cv::Mat point2D_max(2, 1, CV_16UC1, cv::Scalar(0));
        point2D_max.at<int>(0) = 0;
        point2D_max.at<int>(1) = 0;
        cv::Rect rect_bbox;
        cv::Rect rect_cluster;
        cv::Rect rect_intersect;
        double pointnum = 0;
        double ave_dist = 0;
        double projection_iou = 0;
        double score = 0;
        double ave_x = 0;
        double ave_y = 0;
        double ave_z = 0;
        double min_z = 10000;
        for (int j = 0; j < cluster->points.size(); j++) {
            if ((cluster->points[j].x == 0) && (cluster->points[j].y == 0) && (cluster->points[j].z == 0))
                continue; //have many (0,0,0)
            pointnum = pointnum + 1;
            ave_dist = ave_dist +
                       sqrt(cluster->points[j].x * cluster->points[j].x + cluster->points[j].z * cluster->points[j].z);
            ave_x = ave_x + cluster->points[j].x;
            ave_y = ave_y + cluster->points[j].y;
            ave_z = ave_z + cluster->points[j].z;
//            cout<<"ave_dist"<<typeid(ave_dist).name()<<endl;
//            cout<<"ave_dist"<<ave_dist<<endl;
            // find point2D_min and point2D_max
//            cout<<"point 3d in cluster"<<endl;
//            cout << cluster.points[j].x << "," <<cluster.points[j].y<<","<<cluster.points[j].z<<endl;
            point2D_this = Cal3D_2D(cluster->points[j], P1, imageSize);
//            cout<<"Point2d: "<<point2D_this.at<int>(0)<<","<<point2D_this.at<int>(1)<<endl;
            if (point2D_this.at<int>(0) <= point2D_min.at<int>(0))
                point2D_min.at<int>(0) = point2D_this.at<int>(0);
            if (point2D_this.at<int>(1) <= point2D_min.at<int>(1))
                point2D_min.at<int>(1) = point2D_this.at<int>(1);
            if (point2D_this.at<int>(0) >= point2D_max.at<int>(0))
                point2D_max.at<int>(0) = point2D_this.at<int>(0);
            if (point2D_this.at<int>(1) >= point2D_max.at<int>(1))
                point2D_max.at<int>(1) = point2D_this.at<int>(1);
            if (cluster->points[j].z < min_z)
                min_z = cluster->points[j].z;
        }
        // ave_dist
        ave_dist = ave_dist / pointnum;
        ave_x = (ave_x / pointnum) / scale_offset;
        ave_y = (ave_y / pointnum) / scale_offset;
        ave_z = (ave_z / pointnum) / scale_offset;
//        cout<<"point num"<<pointnum<<endl;
        //normalize point num
        pointnum = pointnum / object_point_num;

        // calculate IoU
        rect_cluster.x = point2D_min.at<int>(0);
        rect_cluster.y = point2D_min.at<int>(1);
        rect_cluster.width = point2D_max.at<int>(0) - point2D_min.at<int>(0);
        rect_cluster.height = point2D_max.at<int>(1) - point2D_min.at<int>(1);
        rect_bbox.x = b._xmin;
        rect_bbox.y = b._ymin;
        rect_bbox.width = b._xmax - b._xmin;
        rect_bbox.height = b._ymax - b._ymin;
        projection_iou = intersectRect(rect_bbox, rect_cluster, rect_intersect);
#ifdef show_image
        cv::rectangle(rectified_l, cv::Point(rect_bbox.x, rect_bbox.y), cv::Point(rect_bbox.x+rect_bbox.width, rect_bbox.y+rect_bbox.height), cv::Scalar(255, 0, 0));
        cv::rectangle(rectified_l, cv::Point(rect_cluster.x, rect_cluster.y), cv::Point(rect_cluster.x+rect_cluster.width, rect_cluster.y+rect_cluster.height), cv::Scalar(255, 0, 0));
        imshow("iou", rectified_l);
        if (cv::waitKey(0) == 27) {
        cv::destroyAllWindows();
    }
#endif

        score = pointnum + w1 / ave_dist + w2 * projection_iou;
        if (pointnum == 0) score = 0;
        if (ave_z < 0) {
            score = 0;
            continue;
        }

//        cout << "p_x: " << ave_x << ", p_y: " << ave_y << ", p_z: " << min_z << " ,socre: " << score << " ,pointnum: "
//             << pointnum << ", ave_dist: " << ave_dist << ", IoU: " << projection_iou << endl;
        scores.push_back(score);
        if (score > max_score) {
            max_score = score;
            object_cluster = cluster;
            p_x = ave_x / scale_offset - offset_x;
            p_y = ave_y / scale_offset - offset_y;
            p_z = min_z / scale_offset - offset_z;
        }
    }
    if (max_score == 0) {
        cout << "can't find the best cluster" << endl;
        return false;
    }
    if (p_z > 2.0) {
        cout << "too remote!!!" << endl;
        return false;
    }
    b3d._position_x = p_x;
    b3d._position_y = p_y;
    b3d._position_z = p_z;
    b3d._c = b._c;
    b3d._score = b._score;
//    cout << "find the best cluster with score " << max_score << endl;
//    cout << "p_x: " << p_x << ", p_y: " << p_y << ", p_z: " << p_z << ", class: " << b._c << ", score: " << b._score
//         << endl;

    return true;
}

#ifdef RKNN_MODEL
bool Detector::_obj_det_3d_on_rv1126(cv::Mat img_l, const int frame) {

    bool has_obj = false;
    obj_boxes.clear();
//    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> object_clusters;
//    vector<box3d> b3d_list;
    cv::Mat img = img_l.clone();
    auto start4 = std::chrono::steady_clock::now();
    for (int i = 0; i < bbox_list.size(); i++) {
//        cout << "/////////////object" << i << "/////////////////" << endl;
        bbox b = bbox_list[i];

//        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr(new pcl::PointCloud<pcl::PointXYZ>);
//        pc_object_ptr->width = (b._xmax - b._xmin) * (b._ymax - b._ymin);
//        pc_object_ptr->height = 1;
//        pc_object_ptr->is_dense = false;  //不是稠密型的
//        pc_object_ptr->points.resize(pc_object_ptr->width * pc_object_ptr->height);  //点云总数大小

        int p_index_object = 0;
        double average_x = 0;
        double average_y = 0;
        double average_z = 0;
        int total_times = 0;
        int area;

        if (b._ymax <= 80 || b._ymin >= 560){
            continue;
        }
        else if (b._ymin < 80){
            b._ymin = 80;
        }
        else if (b._ymax > 560){
            b._ymax = 560;
        }

        for (int x = b._xmin; x < b._xmax; x++) {
            for (int y = b._ymin; y < b._ymax; y++) {
                Point point;
                point.x = x;
                point.y = (y - 80);
//                if (image3D_obj.empty()) cout << "empty" << endl;
//                if (!image3D_obj.data) cout << "data empty" << endl;

//                obj_min_d * 1000   obj_max_d * 1000
                // 250, 10000
                //450, 1600
                if (image3D_obj.at<Vec3f>(point)[2] < obj_min_d * 1000 ||
                    image3D_obj.at<Vec3f>(point)[2] > obj_max_d * 1000) {
                    continue;
                }

                average_x += image3D_obj.at<Vec3f>(point)[0];
                average_y += image3D_obj.at<Vec3f>(point)[1];
                average_z += image3D_obj.at<Vec3f>(point)[2];
                total_times += 1;

//                cout << "/////////////process1" << "/////////////////" << endl;
//                pc_object_ptr->points[p_index_object].x = image3D_obj.at<Vec3f>(point)[0] / 1000;
//                pc_object_ptr->points[p_index_object].y = image3D_obj.at<Vec3f>(point)[1] / 1000;
//                pc_object_ptr->points[p_index_object++].z = image3D_obj.at<Vec3f>(point)[2] / 1000;
            }
        }

        if (total_times != 0)
        {
            average_x /= (total_times * 1000);
            average_y /= (total_times * 1000);
            average_z /= (total_times * 1000);
        }

        //高度检测+连续帧检测
//        if (average_y > 0.06)
//        if (average_y > 0 && average_z <= 4 && average_z > 0)
        if (average_y > 0)
        {
//            current_name_arr[b._c] += 1;

//            if ((last_name_arr[b._c] < 1) || (current_name_arr[b._c] < 1))
//            {
//                continue;
//            }
//            cout << "zzzzz   :" << average_z << endl;
//            area = (b._xmax - b._xmin) * (b._ymax - b._ymin);
            drawPred(b._c, b._score, average_y, average_z, b._xmin, b._ymin, b._xmax, b._ymax, img, classes);

            //draw contours
            /*
            Mat obj_win, gray_img, binary_img;

            cvtColor(obj_win, gray_img, COLOR_BGR2GRAY);  //转化成灰度图
            threshold(gray_img, binary_img, 170, 255, THRESH_BINARY | THRESH_OTSU);  //自适应二值化
            obj_win = binary_img(Rect(b._xmin, b._ymin, (b._xmax - b._xmin), (b._ymax - b._ymin)));

            // 轮廓检测
            vector<vector<Point>> contours_obj;  //轮廓
            vector<vector<cv::Point> > hull_obj(contours_obj.size());
            vector<Vec4i> hierarchy;  //存放轮廓结构变量
            findContours(obj_win, contours_obj, hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE, Point());
            for (int t = 0; t < contours_obj.size(); t++)
            {
                if (contourArea(contours_obj[t]) < min_area || contourArea(contours_obj[t]) > 0.95 * obj_win.rows * obj_win.cols)//面积小于area的凸包，可忽略
                    continue;
                cv::convexHull(cv::Mat(contours_obj[t]), hull_obj[t], false);
                cv::Rect r = cv::boundingRect(contours_obj[t]);
                if ((r.x < b._xmin) || ((r.x + r.width) > b._xmax))
                    continue;
                if ((r.y < b._ymin) || ((r.y + r.height) > b._ymax))
                    continue;

//                for (int t = 0; t < contours_obj.size(); t++)hull_obj[t];
                //draw
                drawContours(img, hull_obj, t, cv::Scalar(0, 0, 255), 1, 8, vector<cv::Vec4i>(), 0, cv::Point());
            }
             */

            AABB_box3d aabb_box3d(average_x, average_y, average_z, -1, -1, -1, b._c, b._xmin, b._xmax, b._ymin, b._ymax);
            AABB_box3d aabb_box3d_w;
            transAABB2w(Tcw, aabb_box3d, aabb_box3d_w);
            obj_boxes.push_back(aabb_box3d_w);

        }

    }
//        char height[20];
//        sprintf(height, "h:%06f", average_y);
//        int center_x = (b._xmin+b._xmax) / 2;
//        int center_y = (b._ymin+b._ymax) / 2;
//        cv::putText(img, height, cv::Point(center_x, center_y), 1, 2, cv::Scalar(0, 0, 255), 2);
//

//        std::string save_img_path = "/data/ttt/img_result_obj_det_4000/out_" + std::to_string(frame) + ".jpg";
        std::string save_img_path = save_path + "final_detection_result/out_" + std::to_string(frame) + ".jpg";
        imwrite(save_img_path, img);

#ifdef save_image
        char file_name[20];
        sprintf(file_name, "%06d.jpg", frame);
        imwrite(det_image_path+file_name, img);
#endif
        return has_obj;
}


int Detector::rknn_initialization(unsigned char *model_data_char, int model_total_data_size, int &width, int &height ,rknn_context &model_para, rknn_input_output_num &in_out_num,
                        rknn_input *data_input, rknn_tensor_attr *model_input, rknn_tensor_attr *model_output, bool &model_type){

    int ret;

    ret = rknn_init(&model_para, model_data_char, model_total_data_size, 0);      //rknn模型初始化
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    ret = rknn_query(model_para, RKNN_QUERY_IN_OUT_NUM, &in_out_num, sizeof(in_out_num));  //??

    memset(model_input, 0, sizeof(model_input));
    for (int i = 0; i < in_out_num.n_input; i++)
    {
        model_input[i].index = i;
        ret = rknn_query(model_para, RKNN_QUERY_INPUT_ATTR, &(model_input[i]),      //??
                         sizeof(rknn_tensor_attr));

    }


    memset(model_output, 0, sizeof(model_output));
    for (int i = 0; i < in_out_num.n_output; i++)
    {
        model_output[i].index = i;
        ret = rknn_query(model_para, RKNN_QUERY_OUTPUT_ATTR, &(model_output[i]),       //??
                         sizeof(rknn_tensor_attr));
//        printRKNNTensor(&(model_output[i]));
    }

    int channel = 3;

    if (model_input[0].fmt == RKNN_TENSOR_NCHW)      //rknn模型的输入通道顺序: channel, height, width?   height, width, channel?
    {
//        printf("Model is NCHW input fmt\n");
        width = model_input[0].dims[0];
        height = model_input[0].dims[1];
        model_type = true;
    }
    else
    {
//        printf("Model is NHWC input fmt\n");
        width = model_input[0].dims[1];
        height = model_input[0].dims[2];
    }



    memset(data_input, 0, sizeof(data_input));
    data_input[0].index = 0;

    if (model_input[0].type == RKNN_TENSOR_FLOAT32){
        model_input[0].type = RKNN_TENSOR_FLOAT32;
//        cout << "data inputs RKNN_TENSOR_FLOAT32" << endl;
    }
    else if (model_input[0].type == RKNN_TENSOR_FLOAT16) {
        data_input[0].type = RKNN_TENSOR_FLOAT16;
//        cout << "data inputs RKNN_TENSOR_FLOAT16" << endl;
    } else if (model_input[0].type == RKNN_TENSOR_UINT8) {
        data_input[0].type = RKNN_TENSOR_UINT8;
//        cout << "data inputs RKNN_TENSOR_UINT8" << endl;
    }
    else if (model_input[0].type == RKNN_TENSOR_INT16) {
        data_input[0].type = RKNN_TENSOR_INT16;
//        cout << "data inputs RKNN_TENSOR_INT16" << endl;
    }

    data_input[0].size = width * height * channel;
    data_input[0].fmt = RKNN_TENSOR_NCHW;       //NHWC     NCHW

    data_input[0].pass_through = 1;             //1: 内部不处理，需要手动处理； 0为自动处理

}

//  void *drmBuf, void *resizeBuf,
bool Detector::run(const cv::Mat img_l, const cv::Mat img_r, const int frame, rknn_context rknn_ctx,
                   int input_num, int output_num, int model_input_width, int model_input_height,
                   rknn_input *rknn_inputs, rknn_tensor_attr *input_struct, rknn_tensor_attr *output_struct,
                   bool channel_type) {
    if (!my_mutex.try_lock()) return false;
    if (only_obs_det) {
//        std::cout << "障碍物检测 frame: " << frame << endl;
        obstacle_det(img_l, img_r, frame);
    }
    else{

        cout << "Start obstacle detection" << endl;
        std::chrono::steady_clock::time_point tt1 = std::chrono::steady_clock::now();
        obstacle_det(img_l, img_r, frame);
        //得到物体检测的3d image, 不经过障碍物mask，直接利用滤波之后的视差图
//        cv::Mat disparity_obj(img_l.rows, img_l.cols, CV_32FC1, cv::Scalar::all(0));
//        filtered_disparity.copyTo(disparity_obj(cv::Rect(0, v0, rectified_l.cols, rectified_l.rows - v0)));
        std::chrono::steady_clock::time_point tt2 = std::chrono::steady_clock::now();
        double t_obstacledet = std::chrono::duration_cast<std::chrono::duration<double> >(tt2 - tt1).count();
//        cout << "t_obstacledet: " << t_obstacledet << endl << endl;

        cout << "Pre-processing" << endl;
        bool isReorder210 = true;
        cv::Mat resize_M;
//        img_l.copyTo(resize_M);
        cv::copyMakeBorder(rectified_l, resize_M, 80, 80, 0, 0, BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        //        cv::imwrite("/data/ttt/rectify.jpg",rectified_l);
        ProcessInput(resize_M, &in_data, &(input_struct[0]), isReorder210, channel_type);
        rknn_inputs[0].buf = in_data;
        std::chrono::steady_clock::time_point tt3 = std::chrono::steady_clock::now();
        double t_processinput = std::chrono::duration_cast<std::chrono::duration<double> >(tt3 - tt2).count();
//        cout << "t_processinput: " << t_processinput << endl << endl;

        cout << "Perform depth map calculation" << endl;
        reprojectImageTo3D(filtered_disparity, image3D_obj, Q);
        std::chrono::steady_clock::time_point tt4 = std::chrono::steady_clock::now();
        double t_reprojectImageTo3D = std::chrono::duration_cast<std::chrono::duration<double> >(tt4 - tt3).count();
//        cout << "t_reprojectImageTo3D: " << t_reprojectImageTo3D << endl << endl;

        cout << "Start object detection" << endl;
//        &drmBuf, &resizeBuf,
        _yolo_det_on_rv1126(resize_M, frame, rknn_ctx, input_num, output_num,
                  model_input_width, model_input_height,
                  rknn_inputs, output_struct);

        cout << "Perform height verification" << endl;
        std::chrono::steady_clock::time_point tt5 = std::chrono::steady_clock::now();
        _obj_det_3d_on_rv1126(resize_M, frame);
        std::chrono::steady_clock::time_point tt6 = std::chrono::steady_clock::now();
        double t_obj_det_3d = std::chrono::duration_cast<std::chrono::duration<double> >(tt6 - tt5).count();
//        cout << "t_obj_det_3d: " << t_obj_det_3d << endl << endl;

    }
    my_mutex.unlock();
    return true;
}

#else
bool Detector::_obj_det_3d(const int frame) {
    bool has_obj = false;
    obj_boxes.clear();
    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> object_clusters;
    vector<box3d> b3d_list;
//    cout << "bbox_list.size()" << bbox_list.size() << endl;
    for (int i = 0; i < bbox_list.size(); i++) {
//        cout << "/////////////object" << i << "/////////////////" << endl;
        bbox b = bbox_list[i];
//        cout << "comes here 1" << endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pc_object_ptr->width = (b._xmax - b._xmin) * (b._ymax - b._ymin);
        pc_object_ptr->height = 1;
        pc_object_ptr->is_dense = false;  //不是稠密型的
        pc_object_ptr->points.resize(pc_object_ptr->width * pc_object_ptr->height);  //点云总数大小
//        cout << "comes here 2" << endl;
        int p_index_object = 0;
        for (int x = b._xmin; x < b._xmax; x++) {
            for (int y = b._ymin; y < b._ymax; y++) {
                Point point;
                point.x = x;
                point.y = y;
                if (image3D_obj.at<Vec3f>(point)[2] < obj_min_d * 1000 |
                    image3D_obj.at<Vec3f>(point)[2] > obj_max_d * 1000)
                    continue;
                pc_object_ptr->points[p_index_object].x = image3D_obj.at<Vec3f>(point)[0] / 1000;
                pc_object_ptr->points[p_index_object].y = image3D_obj.at<Vec3f>(point)[1] / 1000;
                pc_object_ptr->points[p_index_object++].z = image3D_obj.at<Vec3f>(point)[2] / 1000;
            }
        }

//        cout << "!!!!!points num1 " << pc_object_ptr->points.size() << endl;
        //voxel downsample //!!!!voxel之后会出现一些在零点的点，直接进行聚类会有错误
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_object_filtered_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        if (pc_object_ptr->points.size() > 5000) {
            pcl::VoxelGrid<pcl::PointXYZ> vox;
            vox.setInputCloud(pc_object_ptr);
            vox.setLeafSize(0.001f, 0.001f, 0.001f);
            vox.filter(*pc_object_filtered_ptr);
        } else {
            pc_object_filtered_ptr = pc_object_ptr;
        }
//        cout << "!!!!!points num2 " << pc_object_filtered_ptr->points.size() << endl;
        if (pc_object_filtered_ptr->points.size() < 20){
//            cout << "comes here 6" << endl;
            continue;
        }
//        cout << "comes here 3" << endl;
//        cout << "!!!!!points num3 " << pc_object_filtered_ptr->points.size() << endl;

        // filtering cloud by passthrough
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(pc_object_filtered_ptr);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(obj_min_d, obj_max_d);
        //pass.setFilterLimitsNegative (true);
        pass.filter(*pc_object_filtered_ptr);


        if (pc_object_filtered_ptr->points.size() < 20)
            continue;
        // cluster the pc_object
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters;
        auto start = std::chrono::steady_clock::now();
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(pc_object_filtered_ptr);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.025); // 2.5cm
        ec.setMinClusterSize(20);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(pc_object_filtered_ptr);
        ec.extract(cluster_indices);
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
             it != cluster_indices.end(); ++it) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr pc_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            pc_cluster->width = 20000;
            pc_cluster->height = 1;
            pc_cluster->is_dense = false;  //不是稠密型的
            pc_cluster->points.resize(pc_cluster->width * pc_cluster->height);  //点云总数大小
            int i = 0;
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
//            cout<<pc_object_ptr->points[*pit].x<<" "<<pc_object_ptr->points[*pit].y<<" "<<pc_object_ptr->points[*pit].z<<endl;
                pc_cluster->points[i].x = pc_object_ptr->points[*pit].x;
                pc_cluster->points[i].y = pc_object_ptr->points[*pit].y;
                pc_cluster->points[i++].z = pc_object_ptr->points[*pit].z;
            }
            pc_clusters.push_back(pc_cluster);
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start; // std::micro 表示以微秒为时间单位

        // find the best cluster for the object
        int object_point_num = pc_object_ptr->points.size();
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        box3d b3d;
        if (_pick_cluster(b, pc_clusters, object_cluster, object_point_num, b3d)) {
            object_clusters.push_back(object_cluster);
            b3d_list.push_back(b3d);
            has_obj = true;
        } else {
        }
    }
//    cout << "comes here 4" << endl;
//    cout << "clusters num: " << object_clusters.size() << endl;

#ifdef pcl_show
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
#endif
    for (int i = 0; i < object_clusters.size(); i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster = object_clusters[i];
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cluster_filtered_ptr(new pcl::PointCloud<pcl::PointXYZ>());
        box3d b3d = b3d_list[i];

        //原始点云存在很多零点，通过滤波器去除
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cluster);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(obj_min_d, obj_max_d);
        //pass.setFilterLimitsNegative (true);
        //ect_cluster_filtered_ptr
        pass.filter(*object_cluster_filtered_ptr);


        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud(object_cluster_filtered_ptr);
        feature_extractor.compute();

        std::vector<float> moment_of_inertia;
        std::vector<float> eccentricity;
        pcl::PointXYZ min_point_AABB;
        pcl::PointXYZ max_point_AABB;

        feature_extractor.getAABB(min_point_AABB, max_point_AABB);

#ifdef pcl_show
        //  show by pcl visualization
            char cloud_id[20];
            char AABB_id[20];
            sprintf(cloud_id, "cloud%i", i);
            sprintf(AABB_id, "AABB%i", i);
            viewer->addPointCloud<pcl::PointXYZ> (object_cluster_filtered_ptr, cloud_id);
            viewer->addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, AABB_id);
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, AABB_id);
#endif

        double p_x = (min_point_AABB.x + max_point_AABB.x) / 2 / scale_offset;
        double p_y = (min_point_AABB.y + max_point_AABB.y) / 2 / scale_offset;
        double p_z = (min_point_AABB.z + max_point_AABB.z) / 2 / scale_offset;
        double width = (max_point_AABB.x - min_point_AABB.x) / scale_offset;
        double height = (max_point_AABB.y - min_point_AABB.y) / scale_offset;
        double length = (max_point_AABB.z - min_point_AABB.z) / scale_offset;
//        AABB_box3d aabb_box3d(p_x, p_y, p_z, width, length, height, b3d._c);
        AABB_box3d aabb_box3d(b3d._position_x, b3d._position_y, b3d._position_z, width, length, height, b3d._c);
        AABB_box3d aabb_box3d_w;
        transAABB2w(Tcw, aabb_box3d, aabb_box3d_w);
        obj_boxes.push_back(aabb_box3d_w);
    }

#ifdef pcl_show
    while(!viewer->wasStopped())
        {
            viewer->spinOnce (100);
        }
#endif
//    cout << "return" << endl;
    return has_obj;
}

bool Detector::run(const cv::Mat img_l, const cv::Mat img_r, const int frame) {
    if (!my_mutex.try_lock()) return false;
    if (only_obs_det) {
//        cout<<"障碍物检测 frame: "<<frame<<endl;
        obstacle_det(img_l, img_r, frame);
    } else {
//        cout << "进行障碍物检测" << endl;
        obstacle_det(img_l, img_r, frame);
        //得到物体检测的3d image, 不经过障碍物mask，直接利用滤波之后的视差图
//        cout<<"计算物体检测3d image"<<endl;
        cv::Mat disparity_obj(img_l.rows, img_l.cols, CV_32FC1, cv::Scalar::all(0));
        filtered_disparity.copyTo(disparity_obj(cv::Rect(0, v0, rectified_l.cols, rectified_l.rows - v0)));
        reprojectImageTo3D(filtered_disparity, image3D_obj, Q);
//        cout << "进行物体识别" << endl;
        _yolo_det(rectified_l, frame);
//         cout << "comes here 5" << endl;
        _obj_det_3d(frame);
    }
    my_mutex.unlock();
    return true;
}

#endif


