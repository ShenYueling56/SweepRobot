//
// Created by shenyl on 2020/10/28.
//

#ifndef SRC_DETECTOR_H
#define SRC_DETECTOR_H

#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include <typeinfo>
#include <mutex>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/ximgproc.hpp"

#include "box3d.h"
#include "yolo_opencv.h"
#include "obj_det_utils.h"

#ifdef RKNN_MODEL
#include "rknn_api.h"
#else
#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/cloud_viewer.h>
#endif

//
//#define save_image  //将过程图片保存到save_path目录，用于调试
//#define show_image  //显示过程图片，用于调试
//#define ros       //使用ros
//#define pcl_show//pcl显示点云和包络框

using namespace std;

class Detector {
private:
    // 锁，用于防止被同时调用多次
    std::mutex my_mutex;
    //结果保存地址
    string save_path;
    //相机参数
    cv::Mat cameraMatrix_L; // 相机的内参数
    cv::Mat cameraMatrix_R; // 初始化相机的内参数
    cv::Mat distCoeffs_L; // 相机的畸变系数
    cv::Mat distCoeffs_R; // 初始化相机的畸变系数
    cv::Mat R, T;
    cv::Size imageSize;
    // 立体校正参数
    cv::Rect validRoi[2];//双目矫正有效区域
    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Mat mapl1, mapl2, mapr1, mapr2; // 图像重投影映射表
    // 相机和世界坐标系外参
    cv::Mat Tcw;
    // 视差图感兴趣区域参数
    int v0;//视差图感兴趣区域纵坐标初始值
    double h0;//比地面高h0的部分作为障碍物
    double b;//双目相机基线长度
    double h;//相机距离地面高度
    // 障碍物深度图mask
    double d0;
    //wls参数
    double wls_lamda;
    double wls_Sigma;
    //障碍物轮廓提取参数
    int w_offset;
    int h_offset;
    double min_area;

    //物体检测网络参数
    vector<string> classes;
    cv::dnn::Net net;
    double scale_offset;
    double w1; //weight for dist
    double w2; //weight for IoU
    double offset_x;
    double offset_y;
    double offset_z;

    // 物体检测范围
    double obj_min_d;
    double obj_max_d;
    // 可通行的障碍物最低高度
    double obs_min_h;
    //毛毯距离地面最低高度和最高高度
    double min_carpet_h;
    double max_carpet_h;

#ifdef save_image
    //双目矫正结果存放目录
    string left_rectified_path;
    string right_rectified_path;
    string pairs_rectified_path;
    //视差图结果存放目录
    string disparities_path;
    string filtered_disparities_path;
    //障碍物mask结果存放目录
    string obstacle_mask_path;
    //障碍物mask作用于视差图后的结果存放目录
    string masked_disparity_path;
    //深度图存放目录
    string depth_image_path;
    string edge_image_path;
    string depth_mask_path;
    //障碍物轮廓图存放目录
    string bbox_image_path;
    // yolo结果图片存放目录
    string yolo_image_path;
    // carpet mask结果存放目录
    string carpet_mask_path;
#endif

public:
    bool only_obs_det; //true: 只进行障碍物检测不进行物体识别； false: 同时进行障碍物检测和物体识别
    cv::Mat rectified_l, rectified_r;//左右双目矫正后的图片
    cv::Mat disparity_l_roi, disparity_r_roi;//roi区域视差图
    cv::Mat filtered_disparity;//滤波后视差图
    cv::Mat obstacle_mask;//障碍物mask
    cv::Mat masked_disparity_obs;//作用障碍物mask后的视差图
    cv::Mat masked_disparity_roi_obs;//作用障碍物mask后的感兴趣区域的视差图
    cv::Mat image3D_obs;//用于障碍物检测的3d图
    cv::Mat image3D_obj;//用于物体识别的3d图
    cv::Mat image3D_iou;//iou部分3D图
    cv::Mat depth_image;//深度图
    cv::Mat depth_mask;//深度小于d0m的障碍物mask
    vector<AABB_box3d> obs_boxes;//障碍物三维框
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_obs_ptr;//障碍物点云
    vector<bbox> bbox_list;//二维物体检测结果框
    vector<AABB_box3d> obj_boxes;//识别物体三维框
    cv::Mat carpet_mask;//毛毯mask

    void *in_data = NULL;

//    uint8_t
    uint8_t *qnt_buf;
    uint8_t *nchw_buf;

private:
    //进行双目立体矫正
    bool _stereoRectification();

    bool _rectification(cv::Mat img1, cv::Mat img2, int frame);

    //计算视差图并进行加权最小二乘滤波
    bool _filterDisparityImage(int frame);

    bool _get_obstacle_mask(int frame, double min_obs_h);

    bool _get_obstacle_depth_image(int frame);

    bool _get_obstacle_boxes(int frame);

    bool _get_obstacle_points(int frame);

    // 物体识别相关
    bool _yolo_det(cv::Mat img_l, int frame);

    bool _obj_det_3d(const int frame);

    bool _pick_cluster(bbox b, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pc_clusters,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr &object_cluster, double object_point_num, box3d &b3d);
#ifdef RKNN_MODEL
    int ProcessInput(cv::Mat img, void **dst_buf, rknn_tensor_attr *in_attr,
                     bool isReorder210, bool isNCHW);

    bool _yolo_det_on_rv1126(cv::Mat img_l, int frame, rknn_context rknn_ctx,
                   int input_num, int output_num, int model_input_width, int model_input_height,
                   rknn_input *input_struct, rknn_tensor_attr *output_struct);

    bool _obj_det_3d_on_rv1126(cv::Mat img_l, const int frame);
#endif

public:
    //构造函数
    Detector(string config_file);

    //析构函数
    ~Detector();

    //障碍物检测
    bool obstacle_det(cv::Mat img_l, cv::Mat img_r, int frame);
    //二维物体识别

#ifdef RKNN_MODEL
    int rknn_initialization(unsigned char *model_data_char, int model_total_data_size, int &width, int &height ,rknn_context &model_para, rknn_input_output_num &in_out_num,
                                      rknn_input *data_input, rknn_tensor_attr *model_input, rknn_tensor_attr *model_output, bool &model_type);

    // rga_context rga, void *drmBuf, void *resizeBuf,
    bool run(cv::Mat img_l, cv::Mat img_r, int frame,  rknn_context rknn_ctx,
             int input_num, int output_num, int model_input_width, int model_input_height,
             rknn_input *rknn_inputs, rknn_tensor_attr *input_struct, rknn_tensor_attr *output_struct,
             bool channel_type);
#else
    bool run(cv::Mat img_l, cv::Mat img_r, int frame);
#endif
};

#endif //SRC_DETECTOR_H
