/*
sweep robot
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include "Util/FileUtil.h"
#include<opencv2/core/core.hpp>
#include "include/Util/selfDefine.h"
#include<include/orb_slam3/System.h>
#include "Util/parameters.h"

using namespace std;

#define realtime // 是否实时运行
#define MAX_NUM 7800
//#define MAX_NUM 1000000

#define SIZE_EVERY_MSG 14

class SensorLoader {
public:

    SensorLoader(string datasetPath) : datasetPath(datasetPath) {
        Td = -0.008;

        unsigned int iSize = datasetPath.size();
        if (datasetPath.at(iSize - 1) != '/')
            datasetPath.push_back('/');
    }

    vector<cv::Point3f> vAcc;
    vector<cv::Point3f> vGyro;
    vector<cv::Point3f> vOdo;
    vector<cv::Point3f> vRPY;
    vector<cv::Point2f> vEncoder;
    vector<double> vTimestampIMU;

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampCam;

    string datasetPath;

// initial value of time offset. unit: s. readed image clock + Td = real image clock (IMU clock)
    double Td;

    void LoadIMUOdo();

    void LoadTimeStamp();

    void LoadImages();

    void LoadData() {
        LoadIMUOdo();
        LoadTimeStamp();
        LoadImages();
        printf("Load vTimestampCam %d Image %d\n", vTimestampCam.size(), vstrImageRight.size());
        assert(vTimestampCam.size() == vstrImageRight.size() && vstrImageRight.size() == vstrImageLeft.size());
        assert(vTimestampIMU.size() == vEncoder.size() && vRPY.size() == vOdo.size());
        assert(vGyro.size() == vAcc.size() && vAcc.size() == vEncoder.size());
        printf("Load Image %d\n", vTimestampCam.size());
        printf("Load IMU %d\n", vTimestampIMU.size());
    }
};

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl << "Usage: ./sweep_robot config_file" << endl;
        return 1;
    }

    cout << fixed << endl << "-------" << endl;

    //// Retrieve paths to images
    int first_imu = 0;

    // 读取参数文件中的参数
    string config_file = string(argv[1]);
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    if (!fs_read.isOpened()) {
        cerr << "ERROR: Wrong path to  settings file" << endl;
        return -1;
    }

    string voc_path;
    fs_read["voc_path"] >> voc_path;
    string data_path;
    fs_read["data_path"] >> data_path;
    // result save pathmk
    string save_path;
    fs_read["save_path"] >> save_path;

    SensorLoader sensorLoader(data_path);
    sensorLoader.LoadData();

    fs_read["td"] >> sensorLoader.Td;
    fs_read.release();

    std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor;
    imuProcessor.reset(new ORB_SLAM3::IMUProcess());
    readParameters(config_file);
    imuProcessor->setParameter();
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(voc_path, config_file, ORB_SLAM3::System::STEREO, true);
    SLAM.SetIMUProcessor(imuProcessor);

    cv::Mat M1l, M2l, M1r, M2r;
#ifdef ALTER_STEREO_MATCHING
    bool do_rectify = false;
#else
    bool do_rectify = true;
#endif
    if (do_rectify) {
        // Read rectification parameters
        cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["cameraMatrixL"] >> K_l;
        fsSettings["cameraMatrixR"] >> K_r;

        fsSettings["P1"] >> P_l;
        fsSettings["P2"] >> P_r;

        fsSettings["R1"] >> R_l;
        fsSettings["R2"] >> R_r;

        fsSettings["distCoeffsL"] >> D_l;
        fsSettings["distCoeffsR"] >> D_r;

        cv::Size imagesize;
        fsSettings["imageSize"] >> imagesize;
        int rows_l = imagesize.height;
        int cols_l = imagesize.width;
        int rows_r = imagesize.height;
        int cols_r = imagesize.width;
        fsSettings.release();

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
            D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F,
                                    M1l,
                                    M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F,
                                    M1r,
                                    M2r);
    }


    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    int nImages = sensorLoader.vstrImageLeft.size();
//    nImages = 1000;
    vTimesTrack.resize(nImages);

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    // Seq loop
    for (int ni = 0; ni < nImages; ) {
        // Read left and right images from file
        imLeft = cv::imread(sensorLoader.vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(sensorLoader.vstrImageRight[ni], cv::IMREAD_UNCHANGED);

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(sensorLoader.vstrImageLeft[ni]) << endl;
            return 1;
        }

        if (imRight.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(sensorLoader.vstrImageRight[ni]) << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t_Start_Rect = std::chrono::steady_clock::now();
        if (do_rectify) {
            cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
            cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
        }
        std::chrono::steady_clock::time_point t_End_Rect = std::chrono::steady_clock::now();

        // Load imu measurements from previous frame
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vector<ORB_SLAM3::ODO::Point> vOdoMeas;
        if (!sensorLoader.vTimestampIMU.empty()) {
            vImuMeas.clear();
            vOdoMeas.clear();
            while (1) {
                if (first_imu > sensorLoader.vTimestampIMU.size() - 1)
                    break;
                double t = sensorLoader.vTimestampIMU[first_imu];
                cv::Point3f acc(sensorLoader.vAcc[first_imu].x,
                                sensorLoader.vAcc[first_imu].y,
                                sensorLoader.vAcc[first_imu].z);
                cv::Point3f gyr(sensorLoader.vGyro[first_imu].x,
                                sensorLoader.vGyro[first_imu].y,
                                sensorLoader.vGyro[first_imu].z);
                cv::Point3f odometer(sensorLoader.vOdo[first_imu].x,
                                     sensorLoader.vOdo[first_imu].y,
                                     sensorLoader.vOdo[first_imu].z);
                cv::Point2f encoder(sensorLoader.vEncoder[first_imu].x,
                                    sensorLoader.vEncoder[first_imu].y);
                cv::Point3f rpy(sensorLoader.vRPY[first_imu].x,
                                sensorLoader.vRPY[first_imu].y,
                                sensorLoader.vRPY[first_imu].z);

                imuProcessor->inputIMU(t, Eigen::Vector3d(acc.x, acc.y, acc.z), Eigen::Vector3d(gyr.x, gyr.y, gyr.z),
                                       encoder);
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                vOdoMeas.push_back(ORB_SLAM3::ODO::Point(odometer, encoder, rpy, t));

                if (sensorLoader.vTimestampIMU[first_imu] > sensorLoader.vTimestampCam[ni]) {
                    first_imu++;
                    break;
                }

                first_imu++;
            }
        }
        bool imu_ready = false;
        // Pass the images to the SLAM system
        double tframe = sensorLoader.vTimestampCam[ni];
        if (!vImuMeas.empty() && vImuMeas.back().t > tframe)
            imu_ready = true;
        if (first_imu < 5)
            continue;
        ORB_SLAM3::Verbose::PrintMess("TrackStereo 1", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (!vOdoMeas.empty())
            imuProcessor->mOdoReceive << vOdoMeas[vOdoMeas.size() - 1].odometer.x, vOdoMeas[vOdoMeas.size() -
                                                                                            1].odometer.y, vOdoMeas[
                    vOdoMeas.size() - 1].rpy.z;
        ORB_SLAM3::Verbose::PrintMess("tracking start", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (imu_ready)
            imuProcessor->preIntegrateIMU(tframe);
        ORB_SLAM3::Verbose::PrintMess("tracking preIntegrateIMU", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        SLAM.TrackStereo(imLeft, imRight, tframe, vImuMeas, vOdoMeas);
        ORB_SLAM3::Verbose::PrintMess("tracking TrackStereo", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (imu_ready)
            imuProcessor->updateIMUBias();
        ORB_SLAM3::Verbose::PrintMess("tracking over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = sensorLoader.vTimestampCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - sensorLoader.vTimestampCam[ni - 1];

#ifdef WAIT_FOR_LOOP
        while(!ORB_SLAM3::LoopClosingFinished){
            usleep(1000);
        }
#endif
#ifdef WAIT_FOR_LOCAL_MAPPING
        while(!ORB_SLAM3::LocalMappingFinished){
            usleep(1000);
        }
#endif

        if (ttrack < T) {
            ni = ni + 1;
#ifdef realtime
            usleep((T - ttrack) * 1e6);
#endif
        }
        else {
            int k = 0;
            while ((sensorLoader.vTimestampCam[ni] + ttrack) > sensorLoader.vTimestampCam[ni + k]) {
                k = k + 1;
            }
#ifdef realtime
            ni = ni + k;
#else
            ni = ni + 1;
#endif
        }
    }

    cout << "Finish SLAM" << endl;
// Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
//    cout << std::fixed << std::setprecision(5);
    cout << "Finish processing" << endl;
    cout << "-------" << endl << endl;

    createDirectory(save_path);
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM(save_path + string("orb3_stereo_slam.txt"));
    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

void SensorLoader::LoadImages() {

    cout << "Start LoadImages." << endl;
//    vTimestampCam.reserve(10000);
//    vstrImageLeft.reserve(10000);
//    vstrImageRight.reserve(10000);

    ifstream fTimes;
    string strPathTimeFile = datasetPath + "cameraStamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    uint8_t cnt = 0;
    int img_i = -1;
    while (!fTimes.eof()) {
        if (img_i > MAX_NUM)
            break;
        img_i = img_i + 1;
        cnt++;
        string s;
        getline(fTimes, s);
        if (cnt < SPEED_UP)
            continue;
        else
            cnt = 0;
        if (!s.empty()) {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestampCam.push_back(t + Td);
        }
    }
    string strPathLeft = datasetPath + "left";
    string strPathRight = datasetPath + "right";

    //load image 法一：
    //getSortedImages(strPathLeft, vstrImageLeft);
    //getSortedImages(strPathRight, vstrImageRight);

    //load image 法二：
    img_i = -1;
    cnt = 0;
    do {
        if (img_i > MAX_NUM)
            break;
        img_i = img_i + 1;
        cnt++;
        if (cnt < SPEED_UP)
            continue;
        else
            cnt = 0;
        stringstream ss;
        ss << setfill('0') << setw(6) << img_i;
        std::string file = strPathLeft + "/" + ss.str() + ".jpg";
        if (exists_file(file)) {
            double t = img_i / 10.0;
            ss.clear();
            ss.str("");
            ss << setfill('0') << setw(6) << img_i;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".jpg");
            ss.clear();
            ss.str("");
            ss << setfill('0') << setw(6) << img_i;
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".jpg");
        } else
            break;
    } while (1);

}


void SensorLoader::LoadIMUOdo() {

    string imu_odoFile = datasetPath + "imu_odo.bin";
    cout << "load " << imu_odoFile << endl;

    std::ifstream cntF(imu_odoFile, std::ios::binary);
    cntF.seekg(0, ios::end);
    uint64_t size = cntF.tellg() / sizeof(float);
    cntF.close();

    float rData[size];
    std::ifstream inF(imu_odoFile, std::ios::binary);
    inF.read((char *) rData, sizeof(float) * size);
    inF.close();

    int sizeMsg = size / SIZE_EVERY_MSG;
    vGyro.clear();
    vAcc.clear();
    vOdo.clear();
    vRPY.clear();
    vEncoder.clear();
    for (int i = 0; i < sizeMsg; i++) {
        vGyro.push_back(cv::Point3f(rData[i * SIZE_EVERY_MSG + 0], rData[i * SIZE_EVERY_MSG + 1],
                                    rData[i * SIZE_EVERY_MSG + 2]));
        vAcc.push_back(cv::Point3f(rData[i * SIZE_EVERY_MSG + 3], rData[i * SIZE_EVERY_MSG + 4],
                                   rData[i * SIZE_EVERY_MSG + 5]));
        vOdo.push_back(cv::Point3f(rData[i * SIZE_EVERY_MSG + 6], rData[i * SIZE_EVERY_MSG + 7],
                                   rData[i * SIZE_EVERY_MSG + 8]));
        vEncoder.push_back(cv::Point2f(rData[i * SIZE_EVERY_MSG + 9], rData[i * SIZE_EVERY_MSG + 10]));
        vRPY.push_back(cv::Point3f(rData[i * SIZE_EVERY_MSG + 11], rData[i * SIZE_EVERY_MSG + 12],
                                   rData[i * SIZE_EVERY_MSG + 13]));
    }

    return;
}


void SensorLoader::LoadTimeStamp() {

    string imuTimeFile = datasetPath + "timeStamp.bin";
    cout << "load " << imuTimeFile << endl;
    std::ifstream cntF(imuTimeFile, std::ios::binary);
    cntF.seekg(0, ios::end);
    uint64_t size = cntF.tellg() / sizeof(uint64_t);
    cntF.close();

    uint64_t rData[size];
    std::ifstream inF(imuTimeFile, std::ios::binary);
    inF.read((char *) rData, sizeof(uint64_t) * size);
    inF.close();

    vTimestampIMU.clear();
//    cout << std::fixed;
    for (int i = 0; i < size; i++) {
        vTimestampIMU.push_back(rData[i] / 1e9);
//        cout << rData[i] / 1e9 << endl;
    }

    return;
}