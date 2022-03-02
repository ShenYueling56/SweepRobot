//
// Created by shenyl on 2020/11/1.
//

#ifndef SRC_OBJ_DET_UTIL_H
#define SRC_OBJ_DET_UTIL_H

#include "box3d.h"
#include "bbox.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <pcl/point_types.h>

cv::Mat Cal3D_2D(pcl::PointXYZ point3D, cv::Mat projectionMatrix, cv::Size imageSize);

float intersectRect(const cv::Rect rectA, const cv::Rect rectB, cv::Rect &intersectRect);

bool transAABB2w(cv::Mat Tcw, AABB_box3d b3d, AABB_box3d &b3d_w);

#endif //SRC_OBJ_DET_UTIL_H
