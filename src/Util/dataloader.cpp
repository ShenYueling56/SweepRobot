//
// Created by qzj on 2021/4/25.
//

#include "include/Util/dataloader.h"

static bool IMUReady = false;
double Td = -0.008;

namespace ORB_SLAM3{

    void ImageGrabber::GrabImage()
    {
        cv::Mat frame, frame_L, frame_R;
        double time = 0.0;
        while(1){

            if(!IMUReady)
                return;

            if (cap.read(frame)) {
                double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - imuInit).count();
                frame_L = frame(cv::Rect(0, 0, 1280 / 2, 480));  //获取缩放后左Camera的图像
                frame_R = frame(cv::Rect(1280 / 2, 0, 1280 / 2, 480)); //获取缩放后右Camera的图像

                mBufMutex.lock();

                if (!imgBuf.empty())
                    imgBuf.pop();

                ImgMsg imgLeftMsg;
                imgLeftMsg.time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - imuInit).count();
                imgLeftMsg.image = frame_L;

                ImgMsg imgRightMsg;
                imgRightMsg.time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - imuInit).count();
                imgRightMsg.image = frame_R;

                imgBuf.push(std::make_pair(imgLeftMsg, imgRightMsg));

                mBufMutex.unlock();

            }
            usleep(100);
        }
    }

    void ImageGrabber::SyncWithImu()
    {
        const double maxTimeDiff = 0.01;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        while(1)
        {
            cv::Mat imLeft, imRight;
            double tImLeft = 0, tImRight = 0;
            if (!imgBuf.empty() && !mpImuGb->imuBuf.empty())
            {
                tImLeft = imgBuf.back().first.time + Td;
                tImRight = imgBuf.back().second.time + Td;

                // IMU值太少了,需要在相机之后依然有
                if(tImLeft <= mpImuGb->imuBuf.back().time)
                {
                    this->mBufMutex.lock();
                    imLeft = imgBuf.back().first.image;
                    imRight = imgBuf.back().second.image;
//                    imgBuf.pop();
                    while(!imgBuf.empty()){
                        imgBuf.pop();
                    }
                    this->mBufMutex.unlock();

                    t1 = std::chrono::steady_clock::now();
                    //载入IMU数据
                    vector<ORB_SLAM3::IMU::Point> vImuMeas;
                    vector<ORB_SLAM3::ODO::Point> vOdoMeas;
                    mpImuGb->mBufMutex.lock();
                    if(!mpImuGb->imuBuf.empty())
                    {
                        // Load imu measurements from buffer
                        vImuMeas.clear();
                        vOdoMeas.clear();
                        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front().time<=(tImLeft+0.01))
                        {
                            double t = mpImuGb->imuBuf.front().time;
                            cv::Point3f acc(mpImuGb->imuBuf.front().imu.a.x, mpImuGb->imuBuf.front().imu.a.y, mpImuGb->imuBuf.front().imu.a.z);
                            cv::Point3f gyr(mpImuGb->imuBuf.front().imu.w.x, mpImuGb->imuBuf.front().imu.w.y, mpImuGb->imuBuf.front().imu.w.z);

                            cv::Point3f odometer(mpImuGb->imuBuf.front().odo.odometer.x, mpImuGb->imuBuf.front().odo.odometer.y,
                                                 mpImuGb->imuBuf.front().odo.odometer.z);
                            cv::Point2f encoder(mpImuGb->imuBuf.front().odo.encoder.x,
                                                mpImuGb->imuBuf.front().odo.encoder.y);
                            cv::Point3f rpy(mpImuGb->imuBuf.front().odo.rpy.x, mpImuGb->imuBuf.front().odo.rpy.y,
                                            mpImuGb->imuBuf.front().odo.rpy.z);

                            vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
                            vOdoMeas.push_back(ORB_SLAM3::ODO::Point(odometer,encoder,rpy,t));
                            mpImuGb->imuBuf.pop();
                        }
                    }
                    mpImuGb->mBufMutex.unlock();
                    if(mbClahe)
                    {
                        mClahe->apply(imLeft,imLeft);
                        mClahe->apply(imRight,imRight);
                    }

                    if(do_rectify)
                    {
                        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
                        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
                    }

                    ORB_SLAM3::Verbose::PrintMess("TrackStereo 1", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    if(!vOdoMeas.empty())
                        mpImuProcessor->mOdoReceive<<vOdoMeas[vOdoMeas.size()-1].odometer.x,vOdoMeas[vOdoMeas.size()-1].odometer.y,vOdoMeas[vOdoMeas.size()-1].rpy.z;
                    ORB_SLAM3::Verbose::PrintMess("tracking start", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    t2 = std::chrono::steady_clock::now();
                    mpImuProcessor->preIntegrateIMU(tImLeft);
                    ORB_SLAM3::Verbose::PrintMess("tracking preIntegrateIMU", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    mpSLAM->TrackStereo(imLeft, imRight, tImLeft, vImuMeas, vOdoMeas);
                    ORB_SLAM3::Verbose::PrintMess("tracking TrackStereo", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    //预积分+偏置估计一共需要1.7ms左右
                    mpImuProcessor->updateIMUBias();
                    ORB_SLAM3::Verbose::PrintMess("tracking over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    t3 = std::chrono::steady_clock::now();
                    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
                    vTimesTrack.push_back(ttrack);
                    vTimesNow.push_back(GetTimeNow());
                    std::chrono::milliseconds tSleep(1);
                    std::this_thread::sleep_for(tSleep);
                }
            }
            usleep(100);
        }
    }

    void ImuGrabber::UpdateLatestPose(){
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        while(1)
        {
            cv::Mat imLeft, imRight;
            double tImLeft = 0, tImRight = 0;
            if (!imuBuf.empty())
            {
                if(mpRealTime->getLatestIMUPose()) {
                    ORB_SLAM3::Pose_t pose_realtime = mpRealTime->saveLatestPose(std::string("orb3_stereo_slam_realtime.txt"));
                }
            }
            usleep(1000);
        }
    }

    void ImuGrabber::GrabImu()
    {
        uint64_t last_time = 0;
        uint64_t imuCnt = 0;
        const double period = 0.01; //10 ms
        while(1){
            double cur_imu_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - imuInit).count();
            uint64_t cur_time = cur_imu_time / period;
            if(cur_time>last_time){

                IMUmsg imu_msg;
                imu_msg.time = cur_imu_time;
                imu_msg.imu.a.x = 0.0;
                imu_msg.imu.a.y = 0.0;
                imu_msg.imu.a.z = 0.0;
                imu_msg.imu.w.x = 0.0;
                imu_msg.imu.w.y = 0.0;
                imu_msg.imu.w.z = 0.0;

                double t = imu_msg.time;
                double dx = imu_msg.imu.a.x;
                double dy = imu_msg.imu.a.y;
                double dz = imu_msg.imu.a.z;
                double rx = imu_msg.imu.w.x;
                double ry = imu_msg.imu.w.y;
                double rz = imu_msg.imu.w.z;
                Eigen::Vector3d acc(dx, dy, dz);
                Eigen::Vector3d gyr(rx, ry, rz);
                cv::Point2f encoder(imu_msg.odo.encoder.x,
                                    imu_msg.odo.encoder.y);
                mpImuProcessor->inputIMU(t, acc, gyr, encoder);
                mpRealTime->inputIMU(t, acc, gyr, encoder);

                mBufMutex.lock();
                imuBuf.push(imu_msg);
                mBufMutex.unlock();

                if(imuCnt++>5)
                    IMUReady = true;

                last_time = cur_time;
            }
            usleep(100);
        }
    }
}