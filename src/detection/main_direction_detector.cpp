//
// Created by shenyl on 2020/11/17.
//
#include "detection/main_direction_detector.h"

Main_Direction_Detector::Main_Direction_Detector(string config_file) {
    // 从参数文件中读取参数
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    fs_read["cameraMatrixL"] >> cameraMatrix_L;
    fs_read["cameraMatrixR"] >> cameraMatrix_R;
    fs_read["main_direction_save_path"] >> save_path;
    fs_read["index_num"] >> index_num;
    fs_read["thLength"] >> thLength;
    fs_read["max_iters"] >> max_iters;
    fs_read["ave_radio"] >> ave_radio;
    fs_read["min_det_radio"] >> min_det_radio;
    fs_read["max_stereo_dif"] >> max_stereo_dif;
    fs_read["max_diff"] >> max_diff;
    fs_read.release();
    // 初始化帧序号为0
    frame = 0;
    // 初始化全局第一帧标识符
    global_begin = true;
    // 主方向识别是否完成标识符
    main_direction_finish = false;
    // 初始化相机参数
    f_l = cameraMatrix_L.at<double>(0, 0);          // Focal length (in pixel)
    f_r = cameraMatrix_R.at<double>(0, 0);         // Focal length (in pixel)
    pp_l.x = cameraMatrix_L.at<double>(0, 2);
    pp_l.y = cameraMatrix_L.at<double>(1, 2);
    pp_r.x = cameraMatrix_R.at<double>(0, 2);
    pp_r.y = cameraMatrix_R.at<double>(1, 2);
    cout << "Initialize main direction" << endl;
}

// LSD line segment detection with opencv
void Main_Direction_Detector::LineDetect_opencv(cv::Mat image, std::vector<std::vector<double> > &lines) {
    bool useRefine = true;
    bool useCanny = false;
    cv::Mat grayImage;
    if (image.channels() == 1)
        grayImage = image;
    else
        cv::cvtColor(image, grayImage, CV_BGR2GRAY);
    if (useCanny)
        Canny(image, image, 50, 200, 3);

    cv::Ptr<cv::LineSegmentDetector> ls = useRefine ? createLineSegmentDetector(cv::LSD_REFINE_STD)
                                                    : createLineSegmentDetector(
                    cv::LSD_REFINE_NONE);

    vector<cv::Vec4f> lines_std;

    ls->detect(grayImage, lines_std);
    std::vector<double> lineTemp(4);
    for (int i = 0; i < lines_std.size(); i++) {
        double x1 = lines_std[i][0];
        double y1 = lines_std[i][1];
        double x2 = lines_std[i][2];
        double y2 = lines_std[i][3];

        double l = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        if (l > thLength) {
            lineTemp[0] = x1;
            lineTemp[1] = y1;
            lineTemp[2] = x2;
            lineTemp[3] = y2;
            lines.push_back(lineTemp);
        }
        // Show found lines
        if (useCanny)
            image = cv::Scalar(0, 0, 255);
        ls->drawSegments(grayImage, lines_std);
    }
}

bool Main_Direction_Detector::covert_vps_to_imgs(std::vector<cv::Point3d> vps, double f, cv::Point2d pp,
                                                 vector<cv::Point2d> &vps_img) {
    for (int i = 0; i < 3; i++) {
        cv::Point3d vp = vps[i];
        cv::Point2d vp_img;
        vp_img.x = floor(vp.x * f / vp.z + pp.x);
        vp_img.y = floor(vp.y * f / vp.z + pp.y);
        vps_img.push_back(vp_img);
//        cout<<vp_img.x<<" "<<vp_img.y<<endl;
    }
    return true;
}

bool Main_Direction_Detector::sort_vps(vector<cv::Point2d> vps_img, vector<cv::Point2d> &vps_sorted, cv::Mat image2,
                                       vector<std::vector<int> > clusters, vector<std::vector<int> > &clusters_sorted) {
    double img_center_x = image2.cols / 2;
    double img_center_y = image2.rows / 2;
    double max_x_dis = abs(vps_img[0].x - img_center_x);
    double max_y_dis = abs(vps_img[0].y - img_center_y);

    int vx_index = 0;
    int vy_index = 0;
    int vz_index = 0;
    if (vps_img.size() != 3) {
//        cout<<"do not have enough vps!"<<endl;
        return false;
    }

    // get the x vps and y vps
    for (int i = 1; i < 3; i++) {
        if (abs(vps_img[i].x - img_center_x) > max_x_dis) {
            vx_index = i;
            max_x_dis = abs(vps_img[i].x - img_center_x);
        }
        if (abs(vps_img[i].y - img_center_y) > max_y_dis) {
            vy_index = i;
            max_y_dis = abs(vps_img[i].y - img_center_y);
        }
    }
    if (vx_index == vy_index) {
//        cout<<"vx = vy"<<endl;
//        return true;
        return false;
    }

    for (int i = 1; i < 3; i++) {
        if (i == vx_index || i == vy_index) continue;
        vz_index = i;
    }

    //display
//    cout<<vx_index<<" "<<vy_index<<" "<<vz_index<<endl;
    vps_sorted.push_back(vps_img[vx_index]);
    vps_sorted.push_back(vps_img[vy_index]);
    vps_sorted.push_back(vps_img[vz_index]);
    clusters_sorted.push_back(clusters[vx_index]);
    clusters_sorted.push_back(clusters[vy_index]);
    clusters_sorted.push_back(clusters[vz_index]);

    return true;
}

bool Main_Direction_Detector::remove_abnormal_vps(vector<cv::Point2d> &vps_sorted, cv::Mat img) {
    cv::Point2d vps_x = vps_sorted[0];
    cv::Point2d vps_y = vps_sorted[1];
    cv::Point2d vps_z = vps_sorted[2];
    if (abs(vps_y.y) < 1500) {
        return false;
    }
//    if(vps_y.x<0 || vps_y.x>img.cols)
//    {
//        return false;
//    }
    return true;
}

bool Main_Direction_Detector::isRotationMatrix(cv::Mat &R) {
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
//    cout<<"shouldBeIdentity"<<shouldBeIdentity<<endl;
//    return  norm(I, shouldBeIdentity) < 1e-6;
}

cv::Vec3f Main_Direction_Detector::rotationMatrixToEulerAngles(cv::Mat &R) {

//    assert(isRotationMatrix(R));
    isRotationMatrix(R);

    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}

bool Main_Direction_Detector::estimate_R(vector<cv::Point2d> vps_img, cv::Mat K, double &yaw) {
    cv::Mat r1(3, 1, CV_64FC1, cv::Scalar::all(0));
    cv::Mat r2(3, 1, CV_64FC1, cv::Scalar::all(0));
    cv::Mat r3(3, 1, CV_64FC1, cv::Scalar::all(0));
    cv::Mat K_inv(3, 3, CV_64FC1, cv::Scalar::all(0));

    invert(K, K_inv);
    cv::Mat V(3, 1, CV_64FC1, cv::Scalar::all(0));

    for (int i = 0; i < 3; i++) {
        V.at<double>(0) = vps_img[i].x;
        V.at<double>(1) = vps_img[i].y;
        V.at<double>(2) = 1;
        cv::Mat K_inv_V_inv(1, 3, CV_64FC1, cv::Scalar::all(0));
//        cout<<K<<endl;
//        cout<<"K_inv"<<K_inv<<endl;
//        cout<<V<<endl;
        cv::Mat K_inv_V = K_inv * V;
//        cout<<K_inv_V<<endl;

        if (i == 0) {
            r1 = K_inv_V / norm(K_inv_V);

        }

        if (i == 1) {
            r3 = K_inv_V / norm(K_inv_V);
            if (r3.at<double>(1) < 0) {
                r3 = -r3;
            }
        }

    }

    // get r2
    r2 = r3.cross(r1);
//    cout<<"r1"<<r1<<endl;
//    cout<<"r3"<<r3<<endl;
//    cout<<"r2"<<r2<<endl;

    // Rcw世界坐标系在相机坐标系下旋转角
    cv::Mat Rcw(3, 3, CV_64FC1, cv::Scalar::all(0));
    r1.copyTo(Rcw(cv::Rect(0, 0, 1, 3)));
    r2.copyTo(Rcw(cv::Rect(1, 0, 1, 3)));
    r3.copyTo(Rcw(cv::Rect(2, 0, 1, 3)));

    cv::Vec3f r_e = rotationMatrixToEulerAngles(Rcw) / CV_PI * 180;

    yaw = atan(r1.at<double>(0) / r1.at<double>(2)) / CV_PI * 180;
    if (yaw > 45) yaw = yaw - 90;
    if (yaw < -45) yaw = yaw + 90;
//    cout<<"yaw: "<<yaw<<endl;
    return true;
}


bool Main_Direction_Detector::yaw_est(vector<cv::Point2d> vps_img_l, vector<vector<double> > lines_l,
                                      vector<vector<int> > clusters_l, cv::Mat image2_l,
                                      vector<cv::Point2d> vps_img_r, vector<vector<double> > lines_r,
                                      vector<vector<int> > clusters_r, cv::Mat image2_r,
                                      double &measure_yaw) {
    //sort vps as vpx, vpy, vpz
    vector<cv::Point2d> vps_sorted_l, vps_sorted_r;
    vector<std::vector<int> > cluster_sorted_l, cluster_sorted_r;
    cv::Mat R_l(3, 3, CV_64FC1, cv::Scalar::all(0));
    cv::Mat R_r(3, 3, CV_64FC1, cv::Scalar::all(0));


    if (sort_vps(vps_img_l, vps_sorted_l, image2_l, clusters_l, cluster_sorted_l)) {
        if (remove_abnormal_vps(vps_sorted_l, image2_l)) {
            double measure_yaw_l;
            double evlt_yaw_l;
            double evlt_delta_yaw_l;
            estimate_R(vps_sorted_l, cameraMatrix_L, measure_yaw_l);
//            cout<<"estimate_R"<<endl;
            if (sort_vps(vps_img_r, vps_sorted_r, image2_r, clusters_r, cluster_sorted_r)) {
                if (remove_abnormal_vps(vps_sorted_r, image2_r)) {
                    double measure_yaw_r;
                    double evlt_yaw_r;
                    double evlt_delta_yaw_r;
                    estimate_R(vps_sorted_r, cameraMatrix_R, measure_yaw_r);

                    if (abs(measure_yaw_l - measure_yaw_r) < max_stereo_dif) {
                        measure_yaw = (measure_yaw_l + measure_yaw_r) / 2;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool Main_Direction_Detector::run(cv::Mat img_l, cv::Mat img_r, double odo_yaw, double frame_t) {
    //从弧度制转化为角度制
    odo_yaw = odo_yaw / CV_PI * 180;
    // 达到index_num, 结合各帧结果利用ransac计算初始帧的yaw
//    cout << "frame: " << frame << "index num: " << index_num << endl;
    if (frame == (index_num - 1)) {
        double interior_num = 0.0;
        int iter = 0;
        double ave = 0;
        double ave_final = 0;

        srand(time(0));

        while (((interior_num / yaw_est_first_frame_list.size()) < ave_radio) && iter < max_iters) {
            ave = 0;
            ave_final = 0;
            interior_num = 0;
            for (int i = 0; i < int(yaw_est_first_frame_list.size() * ave_radio); i++) {
                ave = ave + yaw_est_first_frame_list[rand() % (yaw_est_first_frame_list.size())];
            }
            ave = ave / int(yaw_est_first_frame_list.size() * ave_radio);
            for (int i = 0; i < yaw_est_first_frame_list.size(); i++) {
//                cout<<ave<<"\t"<<yaw_est_first_frame_list[i]<<"\t"<<abs(ave-yaw_est_first_frame_list[i])<<endl;
                if (abs(ave - yaw_est_first_frame_list[i]) < max_diff) {
                    interior_num = interior_num + 1;
                    ave_final = ave_final + yaw_est_first_frame_list[i];
                } else if (abs(abs(ave - yaw_est_first_frame_list[i]) - 90) < 2) {
                    interior_num = interior_num + 1;
                    if (ave - yaw_est_first_frame_list[i] < 0) {
                        ave_final = ave_final + yaw_est_first_frame_list[i] + 90;
                    } else {
                        ave_final = ave_final + yaw_est_first_frame_list[i] - 90;
                    }
                }
            }
//            cout<<"iter: "<<iter<<" "<<ave<<" "<<(interior_num/yaw_est_first_frame_list.size())<<endl;
            iter = iter + 1;
        }
        if (interior_num > 0) {
            ave_final = ave_final / interior_num;
        }
        if (ave_final > 45) ave_final = ave_final - 90;
        if (ave_final < -45) ave_final = ave_final + 90;
//        cout << ave << endl;
//        cout << ave_final << endl;

        if ((iter == max_iters) || (yaw_est_first_frame_list.size() < index_num * min_det_radio)) {
//            ffirst_frame<<fixed<<first_frame_time<<"\t"<<ave_final<<"\t"<<0<<endl;
/////////////////////////////////////////////////
//            double dif_yaw = first_frame_odo_yaw - global_first_frame_odo_yaw;
//            if (dif_yaw<-180) dif_yaw = dif_yaw+360;
//            if (dif_yaw> 180) dif_yaw = dif_yaw-360;
//            double global_ave_final = ave_final - dif_yaw;
//            if(global_ave_final>45) global_ave_final=global_ave_final-90;
//            if(global_ave_final<-45) global_ave_final=global_ave_final+90;
//            result.t = global_first_frame_time;
//            result.yaw = global_ave_final;
//            result.score = (interior_num/yaw_est_first_frame_list.size());
//            // 标志主方向识别结束
//            main_direction_finish = true;
////////////////////////////////////////////////
        } else {
            // change to global first frame
            double dif_yaw = first_frame_odo_yaw - global_first_frame_odo_yaw;
            if (dif_yaw < -180) dif_yaw = dif_yaw + 360;
            if (dif_yaw > 180) dif_yaw = dif_yaw - 360;
            double global_ave_final = ave_final - dif_yaw;
            if (global_ave_final > 45) global_ave_final = global_ave_final - 90;
            if (global_ave_final < -45) global_ave_final = global_ave_final + 90;

//            ffirst_frame<<fixed<<first_frame_time<<"\t"<<ave_final<<"\t"<<(interior_num/yaw_est_first_frame_list.size())<<"\t"<<global_first_time<<"\t"<<global_ave_final<<endl;
//            ffirst_frame.close();

            result.t = global_first_frame_time;
            result.yaw = global_ave_final;
            result.score = (interior_num / yaw_est_first_frame_list.size());
            // 标志主方向识别结束
            main_direction_finish = true;
        }
        frame = 0;
    }
    // 分别进行直线检测
    std::vector<std::vector<double> > lines_l, lines_r;
    LineDetect_opencv(img_l, lines_l);
    LineDetect_opencv(img_r, lines_r);
    if ((lines_l.size() < 5) || (lines_r.size() < 5)) {
//        cout << "Detect no lines!" << endl;
        global_begin = true;
        return false;
    }
    if (global_begin) {
        global_first_frame_odo_yaw = odo_yaw;
        global_first_frame_time = frame_t;
        global_begin = false;
    }
    if (frame == 0) {
        first_frame_odo_yaw = odo_yaw;
        first_frame_time = frame_t;
        yaw_est_first_frame_list.clear();
    }
//    cout<<"进行灭点检测"<<endl;
    // 进行灭点检测
    std::vector<cv::Point3d> vps_l, vps_r;              // Detected vanishing points
    std::vector<std::vector<int> > clusters_l, clusters_r;   // Line segment clustering results of each vanishing point
    VPDetection detector;

    detector.run(lines_l, pp_l, f_l, vps_l, clusters_l);
    detector.run(lines_r, pp_r, f_r, vps_r, clusters_r);

//    cout<<"将灭点投影到图像平面"<<endl;
    // covert vps from 3d to 2d img
    vector<cv::Point2d> vps_img_l, vps_img_r;
    covert_vps_to_imgs(vps_l, f_l, pp_l, vps_img_l);
    covert_vps_to_imgs(vps_r, f_r, pp_r, vps_img_r);

//    cout<<"yaw检测"<<endl;
    double measure_yaw;
    // estmate yaw by camera
    if (yaw_est(vps_img_l, lines_l, clusters_l, img_l,
                vps_img_r, lines_r, clusters_r, img_r,
                measure_yaw)) {
        // change yaw to the first frame according to odo yaw
        double delta_yaw = odo_yaw - first_frame_odo_yaw;
        if (delta_yaw < -180) delta_yaw = delta_yaw + 360;
        if (delta_yaw > 180) delta_yaw = delta_yaw - 360;
        double yaw_est_first_frame = measure_yaw - delta_yaw;
        if (yaw_est_first_frame > 45) yaw_est_first_frame = yaw_est_first_frame - 90;
        if (yaw_est_first_frame < -45) yaw_est_first_frame = yaw_est_first_frame + 90;
        yaw_est_first_frame_list.push_back(yaw_est_first_frame);
//        cout<<"main_dir "<<fixed<<frame_t<<"\t"<<measure_yaw<<"\t"<<yaw_est_first_frame<<endl;
    }
//    cout<<"finish frame "<<frame<<endl;
    frame = frame + 1;
    return true;
}


