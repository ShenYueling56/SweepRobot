//
// Created by shenyl on 2020/11/24.
//
#include "detection/detector.h"
#include "detection/main_direction_detector.h"
#include "detection/carpet_detector.h"
#include <fstream>
#include <unistd.h>
#include <thread>

#define realtime

using namespace std;

void
LoadImages(const string &strSequence, vector<string> &vstrImageFilenamesLeft, vector<string> &vstrImageFilenamesRight,
           vector<double> &vTimestamps);

void
LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro,
        vector<cv::Point3f> &vOdo);


int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl << "Usage: ./sweeper_detection path_to_config_file" << endl;
        return 1;
    }
    string config_file = argv[1];
    //image path
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    string data_path;
    fs_read["data_path"] >> data_path;
    // result save path
    string save_path;
    fs_read["save_path"] >> save_path;
    ofstream main_diretion_result_file;//创建文件
    main_diretion_result_file.open(save_path + "main_direction_result.txt");
    ofstream obs_boxes_file;//创建文件
    obs_boxes_file.open(save_path + "obs_boxes.txt");
    fs_read.release();

    bool write_main_direction_result = false;

    // Retrieve paths to images and IMU
    vector<string> vstrImageFilenamesLeft, vstrImageFilenamesRight;
    vector<double> vTimestampsCam;
    vector<cv::Point3f> vAcc, vGyro, vOdo;
    vector<double> vTimestampsImu;
    // load image
    cout << "Loading Image " << endl;
    LoadImages(data_path, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestampsCam);
    int nImages = vstrImageFilenamesLeft.size();
    // load odometry
    string pathImu = data_path + "/robot.txt";
    cout << "Loading IMU " << endl;
    LoadIMU(pathImu, vTimestampsImu, vAcc, vGyro, vOdo);
    cout << "LOADED!" << endl;

    // Create Detector.
    Main_Direction_Detector *mp_main_direction_detector = new Main_Direction_Detector(config_file);
    Detector *mp_object_detector = new Detector(config_file);
    Carpet_detector *mp_carpet_detector = new Carpet_detector(config_file);

    // Vector for tracking time statistics
    vector<float> vTimesDetect;
    vTimesDetect.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    int imu_index = 0;

    // Main loop
    cv::Mat im_l;
    cv::Mat im_r;
    int ni = 0;
//    for(ni=0; ni<nImages; ni++)
    while (ni < nImages) {
//        cout << "/////////////////frame" << ni << "///////////////////" << endl;
        // Read image from file
        im_l = cv::imread(vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        im_r = cv::imread(vstrImageFilenamesRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestampsCam[ni];

        if (im_l.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenamesLeft[ni] << endl;
            return 1;
        }
        if (im_r.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenamesRight[ni] << endl;
            return 1;
        }

        // 找和相机时间戳最近的IMU时间戳// todo：使用预积分或者slam的结果
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        double odo_yaw;
        while (vTimestampsImu[imu_index] <= vTimestampsCam[ni]) {
            odo_yaw = vOdo[imu_index].z;
            imu_index = imu_index + 1;
        }

        // 进行物体检测线程
        mp_object_detector->run(im_l, im_r, ni);
        cv::Mat mask_iou = cv::Mat(mp_object_detector->obstacle_mask,
                                   cv::Rect(0, mp_carpet_detector->start_y, mp_object_detector->obstacle_mask.cols,
                                            mp_object_detector->obstacle_mask.rows - mp_carpet_detector->start_y));

        //　利用障碍物检测的结果进行毛毯识别
        bool show_process_flag = false;
        mp_carpet_detector->run(im_l, mask_iou, show_process_flag);

        // 将各帧的障碍物检测结果存入文件
        vector<AABB_box3d> obs_boxes = mp_object_detector->obs_boxes;
        for (int i = 0; i < obs_boxes.size(); i++) {
            obs_boxes_file << ni << " " << obs_boxes[i]._position_x << " " << obs_boxes[i]._position_y << " "
                           << obs_boxes[i]._position_z << " " << obs_boxes[i]._width << " " << obs_boxes[i]._height
                           << endl;
        }

        // 主方向识别结果存入文件中
        if (!mp_main_direction_detector->main_direction_finish) {
            std::thread main_direction_detector_thread(&Main_Direction_Detector::run, mp_main_direction_detector, im_l,
                                                       im_r, odo_yaw, vTimestampsCam[ni]);
            main_direction_detector_thread.detach();
        } else if (!write_main_direction_result) {
            Result main_direction_result = mp_main_direction_detector->result;
            main_diretion_result_file << fixed << setprecision(5) << main_direction_result.t << " "
                                      << main_direction_result.yaw << " " << main_direction_result.score << endl;
            write_main_direction_result = true;
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double tdetect = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesDetect[ni] = tdetect;

        // 以图片形式展示检测结果
        cv::Mat obstacle_result_image;
        bitwise_and(mp_object_detector->rectified_l, mp_object_detector->rectified_l, obstacle_result_image,
                    mp_object_detector->obstacle_mask);
//        cv::imshow("origin image", im_l);
//        cv::imshow("obstacle detection result", obstacle_result_image);
//        cv::imshow("carpet detection result", mp_carpet_detector->carpet_region_after_mask);
//        cv::waitKey(5);

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];


        if (tdetect < T) {
            ni = ni + 1;
            usleep((T - tdetect) * 1e6);
        } else {
            int k = 0;
            while ((vTimestampsCam[ni] + tdetect) > vTimestampsCam[ni + k]) {
                k = k + 1;
            }
#ifdef realtime
            ni = ni + k;
#else
            ni = ni + 1;
#endif
        }

    }

    // Tracking time statistics
    vTimesDetect.resize(ni);
    sort(vTimesDetect.begin(), vTimesDetect.end());
    float totaltime = 0;
    for (int i = 0; i < ni; i++) {
        totaltime += vTimesDetect[i];
    }
    cout << "Finish processing" << endl;

    obs_boxes_file.close();
    main_diretion_result_file.close();

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenamesLeft,
                vector<string> &vstrImageFilenamesRight, vector<double> &vTimestamps) {
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "cameraStamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof()) {
        string s;
        getline(fTimes, s);
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "left/";
    string strPrefixRight = strPathToSequence + "right/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenamesLeft.resize(nTimes);
    vstrImageFilenamesRight.resize(nTimes);

    for (int i = 0; i < nTimes; i++) {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenamesLeft[i] = strPrefixLeft + ss.str() + ".jpg";
        vstrImageFilenamesRight[i] = strPrefixRight + ss.str() + ".jpg";
//        cout << vstrImageFilenamesLeft[i] << endl;
    }
    cout << "Finish Load Image: " << vstrImageFilenamesLeft.size() << endl;
}

void
LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro,
        vector<cv::Point3f> &vOdo) {
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    vOdo.reserve(5000);

    while (!fImu.eof()) {
        string s;
        getline(fImu, s);
        if (s[0] == '#')
            continue;

        if (!s.empty()) {
            string item;
            size_t pos = 0;
            double data[10];
            int count = 0;
            while ((pos = s.find(' ')) != string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            //字符串转浮点数
            data[9] = stod(item);

            vTimeStamps.push_back(data[0]);
            vGyro.push_back(cv::Point3f(data[1], data[2], data[3]));
            vAcc.push_back(cv::Point3f(data[4], data[5], data[6]));
            vOdo.push_back(cv::Point3f(data[7], data[8], data[9]));
            //cout<<"data: "<<vTimeStamps.back()<<" "<<vGyro.back().x<<" "<<vGyro.back().y<<" "<<vGyro.back().z<<" "<<vAcc.back().x<<" "<<vAcc.back().y<<" "<<vAcc.back().z<<" "<<vOdo.back().x<<" "<<vOdo.back().y<<" "<<vOdo.back().z<<endl;
        }
    }
//    cout << "Finish LoadIMU: " << vTimeStamps.size() << endl;
}

