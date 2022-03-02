/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>
#include "Util/record.h"
#include "include/Util/dataloader.h"
#include<opencv2/core/core.hpp>
#include"include/orb_slam3/System.h"
#include"include/orb_slam3/Frame.h"
#include "include/Util/selfDefine.h"
#include "Util/parameters.h"
#include <thread>

std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor;

std::mutex m_buf;
static uint32_t imuCnt=0;
static double timeInit=0.0;

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << endl << "Usage: ./sweep_robot config_file" << endl;
        return 1;
    }

    string config_file = string(argv[1]);
    // Load settings related to stereo calibration
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    string voc_path;
    fsSettings["voc_path"] >> voc_path;
    string data_path;
    fsSettings["data_path"] >> data_path;
    // result save pathmk
    string save_path;
    fsSettings["save_path"] >> save_path;
    fsSettings.release();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr<ORB_SLAM3::System> mpSLAM;
    mpSLAM.reset( new ORB_SLAM3::System(argv[1],argv[2], ORB_SLAM3::System::STEREO, false));

    imuProcessor.reset(new ORB_SLAM3::IMUProcess());
    readParameters(config_file);
    imuProcessor->setParameter();
    mpSLAM->SetIMUProcessor(imuProcessor);

    std::shared_ptr<RealTime> pRealTime;
    pRealTime.reset(new RealTime(config_file));
    pRealTime->SetIMUProcessor(imuProcessor);
    pRealTime->SetVSLAM(mpSLAM);

    ORB_SLAM3::ImuGrabber imugb(imuProcessor);
    ORB_SLAM3::ImageGrabber igb(mpSLAM,&imugb, imuProcessor, false, false, config_file);

    igb.setRealTime(pRealTime);
    imugb.setRealTime(pRealTime);

    // Maximum delay, 5 seconds
    std::thread grabImageThread(&ORB_SLAM3::ImageGrabber::GrabImage, &igb);
    std::thread grabImuThread(&ORB_SLAM3::ImuGrabber::GrabImu, &imugb);
    std::thread realTimeThread(&ORB_SLAM3::ImuGrabber::UpdateLatestPose, &imugb);

    std::thread sync_thread(&ORB_SLAM3::ImageGrabber::SyncWithImu,&igb);

    grabImageThread.join();
    grabImuThread.join();
    sync_thread.join();
    realTimeThread.join();

    int size = igb.vTimesTrack.size();
    ofstream f;
    f.open("Track_time.txt", std::ios_base::out);
    f << fixed;
    f<<"#(timeCur) tracking_time"<<endl;
    for(int i=0;i<size;i++)
        f<<igb.vTimesNow[i]<<" "<<igb.vTimesTrack[i]<<endl;
    f.close();

    sort(igb.vTimesTrack.begin(), igb.vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < igb.vTimesTrack.size(); ni++) {
        totaltime += igb.vTimesTrack[ni];
    }
    cout <<fixed <<setprecision(4);

    mpSLAM->Shutdown();

    // Save camera trajectory
    //mpSLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory_TUM_Format.txt");
//    mpSLAM->SaveTime(save_path);
    mpSLAM->SaveTrajectoryTUM("FrameTrajectory_TUM_Format.txt");
    mpSLAM->CreateVoc();

#ifdef SAVE_MAP_PCD
    mpSLAM->SaveMapPCD("slam_map.pcd");
    cout<<"Save Map to pcd "<<endl;
#endif
    //mpSLAM->SaveTrajectoryKITTI("FrameTrajectory_KITTI_Format.txt");

    return 0;
}


