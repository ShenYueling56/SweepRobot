//
// Created by qzj on 2020/9/24.
//

#ifndef SRC_SELFDEFINE_H
#define SRC_SELFDEFINE_H

/********save map as pcd********/
//#define SAVE_MAP_PCD

/*******************DEBUG*******************/
//#define LOCAL_BA_TIME_LOGGING

/******************Stereo Accelerate*********************/
// as an alternative of stereo pipeline, undistort the keypoints and perform stereo matching
// no image recitification is required, nor does the subpixel stereo refine is used
#define ALTER_STEREO_MATCHING
#define LIMIT_NEW_KEYFRAME
#define LIMIT_LOCAL_BA
#define GOOD_GRAPH_KF_THRES             10
//#define GOOD_GRAPH_KF_THRES             100000
#define GOOD_GRAPH_FIXEDKF_THRES        5
//#define GOOD_GRAPH_FIXEDKF_THRES        100000
#define GOOD_GRAPH_EDGES_THRES          2000
//#define GOOD_GRAPH_EDGES_THRES          100000
#define GOOD_GRAPH_NN_KF                5          //default 10

#define REDUNDANT_TH                    0.9 //default 0.9去除关键帧的参数   try 0.5

#define WAIT_FOR_LOOP
#define WAIT_FOR_LOCAL_MAPPING

#define GLOABL_BA_KF                    1 //默认关闭global ba

/******************Semi Dense Map*********************/
//#define USE_SEMI_DENSE_MAP
#ifdef USE_SEMI_DENSE_MAP
#include "include/CARV/ProbabilityMapping.h"
#endif


/********create vocabulary use keyframe images********/
//#define CEATE_VOC
/******************Bag of words**********************/
//#define USE_FBOW
#define USE_DBOW3
//#define USE_DBOW2

/*********Good Graph**********/
//#define ENABLE_GOOD_GRAPH
// number of free KF in local BA to trigger good graph

/*************Good Feature(在此版本上不能用,因为在另外一个程序上测试没用)todo but no use******************/
/* --- options to priortize feature matching wrt local map --- */
#define ORB_SLAM_BASELINE
#ifndef ORB_SLAM_BASELINE
/* --- options of additional search after pose estimation --- */
#define DELAYED_MAP_MATCHING

    /* --- options to priortize feature matching wrt local map --- */
#define GOOD_FEATURE_MAP_MATCHING
    // pre-compute Jacobian for next frame at the end of tracking
    // TODO disable it when using map hash; check the latency vs. performance
#define PRECOMPUTE_WITH_MOTION_MODEL

    /* --- parameters used in good feature --- */
#define USE_INFO_MATRIX

    // limit the budget of computing matrices of existing matches at current frame to 2ms
#define MATRIX_BUDGET_REALTIME  0.002
    // limit the budget of predicting matrices at next frame to 2ms
#define MATRIX_BUDGET_PREDICT   0.002

    // For low-power devices with 2-cores, disable multi-thread matrix building
#define USE_MULTI_THREAD        true // false //
#endif
/* --- options to fair comparison wrt other VO pipelines --- */
//#define DISABLE_RELOC
// time to init tracking with full feature set
#define TIME_INIT_TRACKING          5 // 10 //
#define MAX_FRAME_LOSS_DURATION     999 // 5

/* --- options of non-necessary viz codes --- */
// when running on long-term large-scale dataset, this will save us a lot of time!
#define DISABLE_MAP_VIZ


#endif //SRC_SELFDEFINE_H
