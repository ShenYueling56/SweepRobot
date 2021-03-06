/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "Util/parameters.h"

double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double TD;
int NUM_OF_CAM;
int USE_IMU;
int FREQ_IMU;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int SPEED_UP;
std::vector<std::string> CAM_NAMES;

double CAM_SCALE;

bool USE_ODO_PREDICT = true;

using namespace std;

void readParameters(std::string config_file) {
    int tmp;
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL) {
        cout << "config_file dosen't exist; wrong config_file path" << endl;
        //ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if (USE_IMU) {
        FREQ_IMU = fsSettings["IMU.Frequency"];
        double sf = sqrt(FREQ_IMU);
        ACC_N = fsSettings["IMU.NoiseAcc"].real() * sf;
        ACC_W = fsSettings["IMU.AccWalk"].real() / sf;
        GYR_N = fsSettings["IMU.NoiseGyro"].real() * sf;
        GYR_W = fsSettings["IMU.GyroWalk"].real() / sf;
        G.z() = fsSettings["g_norm"];

        cout << endl;
        cout << "IMU frequency: " << FREQ_IMU << " Hz" << endl;
        cout << "IMU discrete gyro noise: " << GYR_N << " rad/s" << endl;
        cout << "IMU discrete gyro walk: " << GYR_W << " rad/s" << endl;
        cout << "IMU discrete accelerometer noise: " << ACC_N << " m/s^2" << endl;
        cout << "IMU discrete accelerometer walk: " << ACC_W << " m/s^2" << endl;
    }


    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    CAM_SCALE = fsSettings["Camera.scale"];
    tmp = fsSettings["use_odo_predict"];
    if(tmp==0)
        USE_ODO_PREDICT = false;
    else
        USE_ODO_PREDICT = true;
    SPEED_UP = fsSettings["speed_up"];

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2) {
        cout << "have no prior about extrinsic param, calibrate extrinsic param" << endl;
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
    } else {
        if (ESTIMATE_EXTRINSIC == 1) {
            cout << " Optimize extrinsic param around initial guess!" << endl;
        }
        if (ESTIMATE_EXTRINSIC == 0)
            cout << " fix extrinsic param " << endl;

        cv::Mat cv_T;
//        cerr<<"1"<<endl;
        fsSettings["Tbc"] >> cv_T;
//        cerr<<"2"<<endl;
        Eigen::Matrix4d T;
//        cerr<<"2.5"<<endl;
        for (int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                T(i, j) = cv_T.at<float>(i,j);
            }
        }
        //?????????????????????cv::cv2eigen(cv_T, T);???"T=cv_T"

//        cerr<<"3"<<endl;
        RIC.push_back(T.block<3, 3>(0, 0));
//        cerr<<"4"<<endl;
        TIC.push_back(T.block<3, 1>(0, 3));
//        cerr<<"5"<<endl;
    }

    if (NUM_OF_CAM != 1 && NUM_OF_CAM != 2) {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        cout << "Unsynchronized sensors, online estimate time offset, initial td: " << TD << endl;
    else
        cout << "Synchronized sensors, fix time offset: " << TD << endl;

    if (!USE_IMU) {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    fsSettings.release();
}
