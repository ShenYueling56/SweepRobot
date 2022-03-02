/*
sweep robot
*/
# include "Util/parameters.h"
//#define MAX_NUM 7800
#define MAX_NUM 1000000
#ifdef RKNN_MODEL

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include "Util/FileUtil.h"
#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void
LoadImages(string &strPath, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

int main(int argc, char **argv) {
    if (argc != 4) {
        cerr << endl << "Usage: ./stereo_euroc path_to_vocabulary path_to_settings dataset_path" << endl;
        return 1;
    }

    //// Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestampsCam;
    string dataset_path = string(argv[3]);
    LoadImages(dataset_path, vstrImageLeft, vstrImageRight, vTimestampsCam);

    int nImages = vstrImageLeft.size();

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
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

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, M1l,
                                M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F, M1r,
                                M2r);


    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, true);

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    int proccIm = 0;
    for (int ni = 0; ni < nImages; ni++, proccIm++) {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        if (imRight.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageRight[ni]) << endl;
            return 1;
        }


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t_Start_Rect = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t_Start_Rect = std::chrono::monotonic_clock::now();
#endif
        cv::remap(imLeft, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightRect, M1r, M2r, cv::INTER_LINEAR);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t_End_Rect = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t_End_Rect = std::chrono::monotonic_clock::now();
#endif

        double tframe = vTimestampsCam[ni];


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeftRect, imRightRect, tframe, vector<ORB_SLAM3::IMU::Point>(),
                         vector<ORB_SLAM3::ODO::Point>(), vstrImageLeft[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestampsCam[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestampsCam[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6); // 1e6
    }

    // Stop all threads
    SLAM.Shutdown();

// Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }

    // Save camera trajectory
    string file_prefix = dataset_path + "robot" + getDirEnd(dataset_path) + "_";
    if (CLOSE_LOOP)
        SLAM.SaveTrajectoryTUM(file_prefix + string("orb3_stereo_slam.txt"));
    else
        SLAM.SaveTrajectoryTUM(file_prefix + string("orb3_stereo_vo.txt"));

    return 0;
}

void LoadImages(string &strPath, vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                vector<double> &vTimeStamps) {
    cerr << "Start LoadImages." << endl;
    vTimeStamps.reserve(10000);
    vstrImageLeft.reserve(10000);
    vstrImageRight.reserve(10000);


    unsigned int iSize = strPath.size();
    if (strPath.at(iSize - 1) != '/')
        strPath.push_back('/');

    ifstream fTimes;
    string strPathTimeFile = strPath + "cameraStamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    uint8_t cnt = 0;
    while (!fTimes.eof()) {
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
            vTimeStamps.push_back(t);
        }
    }
    string strPathLeft = strPath + "left";
    string strPathRight = strPath + "right";

    //load image 法一：
    //getSortedImages(strPathLeft, vstrImageLeft);
    //getSortedImages(strPathRight, vstrImageRight);

    //load image 法二：
    int img_i = -1;
    do {
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

    assert(vTimeStamps.size() == vstrImageLeft.size() && vTimeStamps.size() == vstrImageRight.size());

    cout << "Finish LoadImages: " << vstrImageLeft.size() << endl;
}
#else

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <include/orb_slam3/System.h>
#include <unistd.h>
#include <thread>


#include "Util/FileUtil.h"
#include "include/Util/selfDefine.h"
#include "detection/detector.h"

#include "detection/main_direction_detector.h"
#include "detection/carpet_detector.h"

//ofstream obs_boxes_file;//创建文件
//ofstream obj_boxes_file;//创建文件

#define realtime // 是否实时运行

using namespace std;

vector<float> vTimesDet; // 用来记录detection所需的时间
int det_frame = 0;
int tracking_start_frame = 0;

void
object_detection_with_carpet(Detector *mp_object_detector, Carpet_detector *mp_carpet_detector, const cv::Mat img_l,
                             const cv::Mat img_r, const int frame);

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
    cout << config_file <<endl;
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    cout << config_file <<endl;
    if (!fs_read.isOpened()) {
        cerr << "ERROR: Wrong path to  settings file" << endl;
        return -1;
    }

    string voc_path;
    fs_read["voc_path"] >> voc_path;
    string data_path;
    fs_read["data_path"] >> data_path;
    // result save path
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
    if (do_rectify)  // 获得双目矫正参数
    {
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

    //  检测部分结果保存文件
    ofstream main_diretion_result_file;//创建文件
    main_diretion_result_file.open(save_path + "main_direction_result.txt");
//    obs_boxes_file.open(save_path + "obs_boxes.txt");
//    obj_boxes_file.open(save_path + "obj_boxes.txt");

    // 主方向识别结果是否已经写入文件标识符
    bool write_main_direction_result = false;

    // Create Detector.
    Main_Direction_Detector *mp_main_direction_detector = new Main_Direction_Detector(config_file);
    Detector *mp_object_detector = new Detector(config_file);
    Carpet_detector *mp_carpet_detector = new Carpet_detector(config_file);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    int nImages = sensorLoader.vstrImageLeft.size();
    // Vector for main direction detection statistics
    vector<float> vTimesMainDirect;

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;

    // Seq loop
    int ni = 0;
    int process_frame_num = 0;
    int maindir_frame_num = 0;
    std::chrono::steady_clock::time_point initial_time = std::chrono::steady_clock::now();
    while (ni < nImages) { // 添加realtime仿真功能
//    for (int ni = 0; ni < nImages; ni++) {
        // Read left and right images from file
//        cout<<"////////////////////frame "<<ni<<"///////////////////////"<<endl;
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
//        double tremap = std::chrono::duration_cast<std::chrono::duration<double> >(t_End_Rect - t_Start_Rect).count();
//        cout<< "!!!!!!!!"<<tremap<<"!!!!!"<<endl;

        // Load imu measurements from previous frame
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vector<ORB_SLAM3::ODO::Point> vOdoMeas;
        double odo_yaw;  // imu提供的偏航角，用于主方向识别
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
                odo_yaw = odometer.z;
                // 找到两帧相机之间的所有IMU和里程计测量值，放入vImuMeas和vOdoMeas中
                if (sensorLoader.vTimestampIMU[first_imu] > sensorLoader.vTimestampCam[ni]) {
                    first_imu++;
                    break;
                }
                first_imu++;
            }
        }


        // 进行主方向识别
        if ((!mp_main_direction_detector->main_direction_finish)  && (ni < 200))
        {
//            cout << "进行主方向识别" << endl;
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double tmaindir = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//            double time_process = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - initial_time).count();
//            if(time_process > 10)
//            {
//                cerr << "主方向识别 timeout " << time_process << endl;
//                mp_main_direction_detector->main_direction_finish = true;
//                break;
//            }

            mp_main_direction_detector->run(imLeft, imRight, odo_yaw, sensorLoader.vTimestampCam[ni]);
            vTimesMainDirect.resize(maindir_frame_num + 1);
            vTimesMainDirect[maindir_frame_num++] = tmaindir;
//            cout << "主方向识别一帧时间: " << tmaindir << endl;

            // Wait to load the next frame
            double T = 0;
            if (ni < nImages - 1)
                T = sensorLoader.vTimestampCam[ni + 1] - sensorLoader.vTimestampCam[ni];
            else if (ni > 0)
                T = sensorLoader.vTimestampCam[ni] - sensorLoader.vTimestampCam[ni - 1];
//
            if (tmaindir < T) {
                ni = ni + 1;
#ifdef realtime
            usleep((T - tmaindir) * 1e6);
#endif
            }
//
            else {
                int k = 0;
                while ((sensorLoader.vTimestampCam[ni] + tmaindir) > sensorLoader.vTimestampCam[ni + k]) {
                    k = k + 1;
                }
#ifdef realtime
                ni = ni + k;
#else
                ni = ni + 1;
#endif
            }

            sort(vTimesMainDirect.begin(), vTimesMainDirect.end());
            float totaltime = 0;
            for (int k = 0; k < maindir_frame_num; k++) {
                totaltime += vTimesMainDirect[k];
            }
            continue;
        } else if (!write_main_direction_result)  // 将主方向识别结果存入文件
        {
            Result main_direction_result = mp_main_direction_detector->result;
            main_diretion_result_file << fixed << setprecision(5) << main_direction_result.t << " "
                                      << main_direction_result.yaw << " " << main_direction_result.score << endl;
            write_main_direction_result = true;
            tracking_start_frame = ni; // 记录slam+detection开始帧序号
        }

        // 开启物体检测和毛毯识别线程
        std::thread object_detection_thread(object_detection_with_carpet, mp_object_detector, mp_carpet_detector,
                                            imLeft, imRight, ni);
        object_detection_thread.detach();

        // Pass the images to the SLAM system
        bool imu_ready = false;
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

//        object_detection_thread.join(); //等待目标检测结束

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack.resize(process_frame_num + 1);
        vTimesTrack[process_frame_num++] = ttrack;

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
//            usleep((T-ttrack)*1e6);
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
    for (int ni = 0; ni < process_frame_num; ni++) {
        totaltime += vTimesTrack[ni];
    }
    sort(vTimesDet.begin(), vTimesDet.end());
    float totaltime_det = 0;
    for (int ni = 0; ni < det_frame; ni++) {
        totaltime_det += vTimesDet[ni];
    }
    cout << "Finish processing" << endl;

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


#define SIZE_EVERY_MSG 14

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


void object_detection_with_carpet(Detector *mp_object_detector, Carpet_detector *mp_carpet_detector, const cv::Mat im_l,
                                  const cv::Mat im_r, const int ni) {
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    // 进行物体检测线程
    if (mp_object_detector->run(im_l, im_r, ni)) {
        // 将各帧的障碍物检测结果存入文件
        vector<AABB_box3d> obs_boxes = mp_object_detector->obs_boxes;
//        vector<AABB_box3d> obj_boxes = mp_object_detector->obj_boxes;
//        for (int i = 0; i < obs_boxes.size(); i++) {
//            obs_boxes_file << ni << " " << obs_boxes[i]._position_x << " " << obs_boxes[i]._position_y << " "
//                           << obs_boxes[i]._position_z << " " << obs_boxes[i]._width << " " << obs_boxes[i]._height
//                           << endl;
//        }
//        for (int i = 0; i < obj_boxes.size(); i++) {
////            cout << "位置(x,y,z)： " << obj_boxes[i]._position_x << ", " << obj_boxes[i]._position_y << ", "
////                 << obj_boxes[i]._position_z;
////            cout << " 尺寸(w,h)： " << obj_boxes[i]._width << ", " << obj_boxes[i]._height << endl;
//            obj_boxes_file << ni << " " << 1 << " " << obj_boxes[i]._c << " " << obj_boxes[i]._position_x << " "
//                           << obj_boxes[i]._position_y << " " << obj_boxes[i]._position_z << " " << obj_boxes[i]._width
//                           << " " << obj_boxes[i]._height << endl;
//        }

        cv::Mat mask_iou = cv::Mat(mp_object_detector->obstacle_mask,
                                   cv::Rect(0, mp_carpet_detector->start_y, mp_object_detector->obstacle_mask.cols,
                                            mp_object_detector->obstacle_mask.rows - mp_carpet_detector->start_y));
        // 利用障碍物检测结果进行毛毯识别
        if (mp_carpet_detector->run(im_l, mask_iou, false)) {
            // 以图片形式展示检测结果
//            cv::namedWindow("obstacle detection result");
//            cv::namedWindow("carpet detection result");

            cv::Mat obstacle_result_image;
            bitwise_and(mp_object_detector->rectified_l, mp_object_detector->rectified_l, obstacle_result_image,
                        mp_object_detector->obstacle_mask);
//            cv::imshow("obstacle detection result", obstacle_result_image);
//            cv::imshow("carpet detection result", mp_carpet_detector->carpet_region_after_mask);

            // 归一化深度图并显示深度图
            double minv = 0.0, maxv = 0.0;
            double *minp = &minv;
            double *maxp = &maxv;
            minMaxIdx(mp_object_detector->depth_image, minp, maxp);
            cv::Mat depth_img_show;
            mp_object_detector->depth_image.convertTo(depth_img_show, CV_8U, 255.0 / (maxv - minv));//映射到0-255
//            cv::imshow("depth image", depth_img_show);

            startWindowThread();//开始不断的更新图片
            cv::waitKey(20);
        }
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        double tdet = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        vTimesDet.resize(det_frame + 1);
        vTimesDet[det_frame++] = tdet;
//        cout<<"detection time: "<<tdet<<endl;
    }

    sort(vTimesDet.begin(), vTimesDet.end());
    float totaltime = 0;
    for (int k = 0; k < det_frame; k++) {
        totaltime += vTimesDet[k];
    }
//    cout << "mean detection time: " << totaltime / det_frame << endl;
}
#endif
