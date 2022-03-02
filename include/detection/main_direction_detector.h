//
// Created by shenyl on 2020/11/17.
//

#ifndef MAIN_DIRECTION_DETECTION_MAIN_DIRECTION_DETECTOR_H
#define MAIN_DIRECTION_DETECTION_MAIN_DIRECTION_DETECTOR_H

#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include <typeinfo>
#include <mutex>

#include "VPDetection.h"

using namespace std;

struct Result {
    double t;
    double yaw;
    double score;
};

class Main_Direction_Detector {
private:
    // 结果保存地址
    string save_path;
    // 相机参数
    cv::Mat cameraMatrix_L; // 相机的内参数
    cv::Mat cameraMatrix_R; // 初始化相机的内参数
    cv::Point2d pp_l, pp_r;
    double f_l, f_r;
    int index_num;
    double thLength;
    // ransac参数
    int max_iters; // 20
    float ave_radio;//0.5
    float min_det_radio;//0.5
    // 双目矫正参数
    float max_stereo_dif;

    // 标记第一帧
    bool global_begin;
    // 记录全局第一帧的时间和第一帧odo_yaw
    double global_first_frame_odo_yaw;
    double global_first_frame_time;
    // 记录当前循环第一帧的时间和odo_yaw
    double first_frame_odo_yaw;
    double first_frame_time;
    // 存储当前循环下所有yaw_est
    vector<double> yaw_est_first_frame_list;
    double max_diff;

public:
    // 帧序号
    int frame;
    // 标记主方向识别是否完成
    bool main_direction_finish;
    // 主方向结果
    Result result;

public:
    // 构造函数
    Main_Direction_Detector(string config_file);

    // 运行函数
    bool run(cv::Mat img_l, cv::Mat img_r, double odo_yaw, double frame_t);

private:
    void LineDetect_opencv(cv::Mat image, std::vector<std::vector<double> > &lines);

    bool covert_vps_to_imgs(std::vector<cv::Point3d> vps, double f, cv::Point2d pp, vector<cv::Point2d> &vps_img);

    bool yaw_est(vector<cv::Point2d> vps_img_l, vector<vector<double> > lines_l, vector<vector<int> > clusters_l,
                 cv::Mat image2_l,
                 vector<cv::Point2d> vps_img_r, vector<vector<double> > lines_r, vector<vector<int> > clusters_r,
                 cv::Mat image2_r,
                 double &measure_yaw);

    bool sort_vps(vector<cv::Point2d> vps_img, vector<cv::Point2d> &vps_sorted, cv::Mat image2,
                  vector<std::vector<int> > clusters, vector<std::vector<int> > &clusters_sorted);

    bool remove_abnormal_vps(vector<cv::Point2d> &vps_sorted, cv::Mat img);

    bool estimate_R(vector<cv::Point2d> vps_img, cv::Mat K, double &yaw);

    bool isRotationMatrix(cv::Mat &R);

    cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);
};


#endif //MAIN_DIRECTION_DETECTION_MAIN_DIRECTION_DETECTOR_H


