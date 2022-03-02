//
// Created by qzj on 2020/7/29.
//

#ifndef ORB_SLAM3_ODOMETER_H
#define ORB_SLAM3_ODOMETER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <mutex>
#include "include/orb_slam3/Converter.h"

namespace ORB_SLAM3 {

    namespace ODO {

        class Point {
        public:
            //Point(const float &acc_x, const float &acc_y, const float &acc_z,
            //      const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
            //      const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
            Point(const cv::Point3f odometer, const cv::Point2f encoder, const cv::Point3f rpy, const double &timestamp)
                    :
                    odometer(odometer.x, odometer.y, odometer.z), encoder(encoder.x, encoder.y),
                    rpy(rpy.x, rpy.y, rpy.z), t(timestamp) {}
            Point(){}

        public:
            cv::Point3f odometer;
            cv::Point2f encoder;
            cv::Point3f rpy;
            double t;
        };
    }
}

#endif //ORB_SLAM3_ODOMETER_H
