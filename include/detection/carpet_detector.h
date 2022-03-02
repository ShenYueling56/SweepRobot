#include <stdio.h>
#include <sstream>
#include <iostream>
#include "detector.h"
#include <mutex>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/ximgproc.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

class Carpet_detector {
private:
    // 锁，用于防止被同时调用多次
    std::mutex my_mutex;
    //可调节参数
    int roi_h;
    int canny_threshold;                       //边缘检测阈值
    int close_operation_size;                 //闭操作核大小
    int open_operation_size;                   //开操作核大小
    int carpet_area_threshold;               //视为地毯面积的阈值
    int carpet_width_threshold;               //视为地毯宽度的阈值
    int detected_times_threshold;               //连续帧检测的阈值

    //固定参数（一般情况下无需调节）
    cv::Size imageSize;
    string text = "Carpet";
    int font_face = FONT_HERSHEY_COMPLEX;
    double font_scale = 1;
    int thickness = 1;
    int baseline;
    Size text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);

    bool display_rec_region_flag = false;       //框选出毛毯所在区域
    int roi_height, roi_width, ori_height;      //感兴趣区域高度，宽度
    int carpet_detected_times = 0;              //连续帧检测次数
    int no_carpet_times = 0;                    //无毛毯次数

    vector<vector<cv::Point>> contours;
    vector<vector<cv::Point>> rec_region_contours;

    bool show_process();

    bool _get_possible_region(const Mat img);

    bool _remove_noise();

    bool _calculate_area();

    bool _carpet_in_roi(bool visualize_process_flag);

public:
    int start_y;
    int detected_times = 0;                //检测到地毯的帧数
    //bool visualize_process_flag = false;   //是否可视化
    bool quit_flag = false;   //是否退出

    Mat rectified_left_image;
    Mat rectified_left_roi;
    Mat gray_roi, edge_roi;
    Mat close_operation_mask;
    Mat merged_mask;

    Mat open_operation_merged_mask;
    Mat current_disturbance_mask;
    Mat carpet_region_mask;
    Mat carpet_region_after_mask;
    Mat display_carpet_region_in_rec;

    //构造函数
    Carpet_detector(string config_file);

    //毛毯检测
    bool run(const Mat img_l, const Mat obstacle_mask, bool visualize_process_flag);
};