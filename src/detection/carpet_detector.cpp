#include "detection/carpet_detector.h"

Carpet_detector::Carpet_detector(string config_file) {
    //创建实例
    //从参数文件中读取相机参数
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    fs_read["roi_h"] >> roi_h;
    fs_read["canny_threshold"] >> canny_threshold;
    fs_read["close_operation_size"] >> close_operation_size;
    fs_read["open_operation_size"] >> open_operation_size;
    fs_read["carpet_area_threshold"] >> carpet_area_threshold;
    fs_read["carpet_width_threshold"] >> carpet_width_threshold;
    fs_read["detected_times_threshold"] >> detected_times_threshold;
    fs_read["imageSize"] >> imageSize;
    fs_read.release();

    ori_height = imageSize.height;  //rectified_left_image.rows;
    roi_width = imageSize.width;     //rectified_left_image.cols;
    roi_height = roi_h;
    start_y = ori_height - roi_height;
}

bool Carpet_detector::_get_possible_region(const Mat img) {
    //Step2: 利用闭操作获取毛毯所在区域的mask
    img.copyTo(rectified_left_image);

    Mat img_roi(rectified_left_image, Rect(0, start_y, roi_width, roi_height));
    img_roi.copyTo(rectified_left_roi);

    cvtColor(rectified_left_roi, gray_roi, COLOR_BGR2GRAY);
    blur(gray_roi, gray_roi, Size(5, 5));
    Canny(gray_roi, edge_roi, canny_threshold, canny_threshold * 2);      //边缘检测

    Mat element_close = getStructuringElement(MORPH_CROSS, Size(close_operation_size, close_operation_size),
                                              Point(-1, -1));    //闭操作核
    dilate(edge_roi, close_operation_mask, element_close);
    erode(close_operation_mask, close_operation_mask, element_close);

    close_operation_mask.copyTo(merged_mask);
    //merged_mask.setTo(0, object_detection_mask);

    return true;
}

bool Carpet_detector::_remove_noise() {
    //Step3: 进一步去除干扰物体
    Mat element_open = getStructuringElement(MORPH_CROSS, Size(open_operation_size, open_operation_size),
                                             Point(-1, -1));
    erode(merged_mask, open_operation_merged_mask, element_open);
    dilate(open_operation_merged_mask, open_operation_merged_mask, element_open);
    return true;
}

bool Carpet_detector::_calculate_area() {
    //Step4： 查找轮廓，对应连通域 
    carpet_region_mask = Mat::zeros(rectified_left_roi.size(), CV_8UC1);
    findContours(open_operation_merged_mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++)    //寻找连通域
    {
        Rect rec = boundingRect(contours[i]);
        double area = contourArea(contours[i]);
        //cout << "area:" << area << endl;
        if ((area >= carpet_area_threshold) && (rec.width >= carpet_width_threshold)) {
            carpet_region_mask(rec).setTo(255);
            display_rec_region_flag = true;
            if (rec.y >= 80) {
                Rect rec_fill = Rect(rec.x, 30, rec.width, rec.y - 30);
                carpet_region_mask(rec_fill).setTo(255);
            }
        } else {
            current_disturbance_mask = Mat::zeros(rectified_left_roi.size(), CV_8UC1);
            current_disturbance_mask(rec).setTo(255);
            open_operation_merged_mask.setTo(0, current_disturbance_mask);
            no_carpet_times++;
        }
    }
    if (no_carpet_times == contours.size()) display_rec_region_flag = false;

    if (display_rec_region_flag == true) {
        carpet_detected_times++;
    } else {
        carpet_detected_times = 0;
    }

    no_carpet_times = 0;

    return true;
}

bool Carpet_detector::_carpet_in_roi(bool visualize_process_flag) {
    //Step5: 获取毛毯在roi区域中的图像
    if (carpet_detected_times >= detected_times_threshold) {
        bitwise_and(rectified_left_roi, rectified_left_roi, carpet_region_after_mask, open_operation_merged_mask);

        bitwise_and(rectified_left_roi, rectified_left_roi, display_carpet_region_in_rec, carpet_region_mask);

        //Step6: 框出毛毯区域
        findContours(carpet_region_mask, rec_region_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        for (size_t i = 0; i < rec_region_contours.size(); i++)    //寻找连通域
        {
            Rect carpet_rec = boundingRect(rec_region_contours[i]);
            rectangle(carpet_region_after_mask, carpet_rec, Scalar(0, 255, 0));

            Point origin;
            origin.x = carpet_rec.x;
            origin.y = carpet_rec.y + text_size.height;
            putText(carpet_region_after_mask, text, origin, font_face, font_scale, Scalar(0, 255, 255), thickness, 8,
                    0);
            if (i == 0) {
                //cout << "Detected!" << endl;
                detected_times += 1;
            }
        }
    } else {
        carpet_region_after_mask = Mat::zeros(rectified_left_roi.size(), CV_8UC3);
        display_carpet_region_in_rec = Mat::zeros(rectified_left_roi.size(), CV_8UC3);
    }

    if (visualize_process_flag == true) {
        show_process();
    }
    return true;
}

bool Carpet_detector::show_process() {

//    namedWindow("edge_roi", 1);                                  //边缘检测图像
//    namedWindow("close_operation_mask", 1);                      //边缘图像经过闭操作处理后的掩膜
//    namedWindow("open_operation_merged_mask", 1);                //对mask进行开操作后处理的图像

//    namedWindow("origin_left_roi", 1);                           //畸变矫正后的感兴趣区域
//    namedWindow("Carpet Region", 1);                             //地毯区域及其包络框
//    namedWindow("display_carpet_region_in_rec", 1);              //地毯区域对应的外接矩形区域图像

//    moveWindow("edge_roi", 0, 360);
//    moveWindow("close_operation_mask", 700, 360);
//    moveWindow("open_operation_merged_mask", 1340, 360);

//    moveWindow("origin_left_roi", 0, 750);
//    moveWindow("Carpet Region", 700, 750);
//    moveWindow("display_carpet_region_in_rec", 1340, 750);

//    imshow("edge_roi", edge_roi);                                           //边缘检测图像
//    imshow("close_operation_mask", close_operation_mask);                   //边缘图像经过闭操作处理后的掩膜
//    imshow("open_operation_merged_mask",
//           open_operation_merged_mask);       //对mask进行开操作后处理的图像                                   //上述所有掩膜合并后的mask

//    imshow("origin_left_roi", rectified_left_roi);                          //畸变矫正后的感兴趣区域
//    imshow("Carpet Region", carpet_region_after_mask);                      //地毯区域及其包络框
//    imshow("display_carpet_region_in_rec", display_carpet_region_in_rec);   //地毯区域对应的外接矩形区域图像

//    if (waitKey(0) == 27) {
//        cout << "Terminated!" << endl;
//        quit_flag = true;
//        destroyAllWindows();
//    }
    return true;
}

bool Carpet_detector::run(const Mat img_l, const Mat obstacle_mask, bool visualize_process_flag) {
    if (my_mutex.try_lock() == false) return false;
//    cout<<"毛毯识别"<<endl;
    carpet_region_after_mask = Mat::zeros(rectified_left_roi.size(), CV_8UC1);
    display_carpet_region_in_rec = Mat::zeros(rectified_left_roi.size(), CV_8UC1);

    _get_possible_region(img_l);
    merged_mask.setTo(0, obstacle_mask); // 将障碍物mask和用闭操作得到的毛毯所在区域mask进行相与
    _remove_noise();
    _calculate_area();
    _carpet_in_roi(visualize_process_flag);

    if (visualize_process_flag == true) {
        destroyAllWindows();
    }
    my_mutex.unlock();
    return true;
}
