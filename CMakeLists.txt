cmake_minimum_required(VERSION 2.8)

SET(system_frame "Linux64")
#SET(system_frame "ARM32")

if(system_frame MATCHES "ARM32")
    message("Cross Compile Model")
    #改为自己系统里交叉编译工具链的路径
    SET(CMAKE_C_COMPILER /home/houj/cross_compile/tool_chains/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc)
    SET(CMAKE_CXX_COMPILER /home/houj/cross_compile/tool_chains/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++)
    #交叉编译系统路径（模拟嵌入式开发板环境）
    SET(CMAKE_SYSROOT /home/houj/sysroot/)
endif()

project(sweepSLAM)

#SET (ENABLE_CRATE_MAP "TRUE")
#SET (ENABLE_FBOW "TRUE")
#SET (ENABLE_DBOW2 "TRUE")
SET(ENABLE_DBOW3 "TRUE")

set(CMAKE_WARN_DEPRECATED 0)
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

#默认debug 如果clion选择Debug则为debug
SET(CMAKE_BUILD_TYPE Release)
IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#debug版的项目生成的可执行文件需要有调试信息并且不需要进行优化,而release版的不需要调试信息但需要优化。
#这些特性在gcc/g++中是通过编译时的参数来决定的,如果将优化程度调到最高需要设置参数-O3,最低是-O0即不做优化;添加调试信息的参数是-g -ggdb,如果不添加这个参数,调试信息就不会被包含在生成的二进制文件中。
#CMake中有一个变量CMAKE_BUILD_TYPE,可以的取值是Debug、Release、RelWithDebInfo和MinSizeRel。
#   当这个变量值为Debug的时候,CMake会使用变量CMAKE_CXX_FLAGS_DEBUG和CMAKE_C_FLAGS_DEBUG中的字符串作为编译选项生成Makefile
#   当这个变量值为Release的时候,工程会使用变量CMAKE_CXX_FLAGS_RELEASE和CMAKE_C_FLAGS_RELEASE选项生成Makefile
#-march=native也是用于加速的，并且生成的库文件可能会因为系统架构报错

#-O3 运行速度快，不过debug只能锁定汇编代码
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pthread ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread")

if(system_frame MATCHES "Linux64")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -march=native")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
IF (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
ENDIF (CMAKE_COMPILER_IS_GNUCC)
# Check C++17 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC17)
    add_definitions(-DCOMPILEDWITHC11)

    message(STATUS "Using flag -std=c++17.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif ()

set(lib_path lib)
set(build_path build)

if(system_frame MATCHES "ARM32")
    add_definitions(-DRKNN_MODEL)

    #rknn_yolov5  32位
    set(LIB_ARCH lib)
    set(lib_path lib32)
    set(build_path build_for_32)
    ##########以下为rknn的部分，需要保证共享库为32位且支持ARM架构

    ## rga
    set(RGA_DIR ${CMAKE_SOURCE_DIR}/Thirdparty/rga)
    include_directories(${RGA_DIR}/include)
    #
    ## drm
    set(DRM_DIR ${CMAKE_SOURCE_DIR}/Thirdparty/drm)
    include_directories(${DRM_DIR}/include)
    include_directories(${DRM_DIR}/include/libdrm)

    include_directories(${CMAKE_SOURCE_DIR}/include/detection/yolov5/)

    # rknn api
    set(RKNN_API_PATH ${CMAKE_SYSROOT}/aLibrary/librknn_api/)
    include_directories(${RKNN_API_PATH}/include)
    set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

    SET(OpenCV_DIR /home/houj/opencv-3.4.5/build_for_arm/)  #SLAM， 改为交叉编译后的build文件路径
    SET(BOOST_ROOT /home/houj/sysroot/aLibrary/BOOST)  #for 32，  改为交叉编译后的build文件路径
    SET(PCL_DIR /home/houj/sysroot/aLibrary/pcl-1.8.1/build_for_arm/)  #设为自己的路径

    #set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/eigen3) #设为自己的路径
    set(EIGEN3_INCLUDE_DIR ${CMAKE_SYSROOT}/../sweepSLAM/include/)
    set(Sophus_INCLUDE_DIR ${CMAKE_SYSROOT}/aLibrary/)
    ##########
elseif(system_frame MATCHES "Linux64")
    set(Sophus_INCLUDE_DIR /home/houj/projects/ov2_slam_ws/src/ov2slam/Thirdparty/Sophus)
    SET(OpenCV_DIR /home/houj/dependency/opencv-3.4.5/build/)
    find_package(Eigen3 REQUIRED)
endif()

#指定目录寻找OpenCVConfig.cmake 文件

find_package(OpenCV 3.4.5 REQUIRED) # and don't look anywhere else

if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})
MESSAGE("OpenCV_VERSION" ${OpenCV_VERSION})
MESSAGE("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS regex thread system filesystem)
add_definitions(-DBOOST_NO_CXX17_SCOPED_ENUMS)

#find_package(Eigen3 REQUIRED)

#find_package(PCL REQUIRED
#        PATHS /home/qzj/Downloads/tool/pcl-1.8.1/build # look here
#        NO_DEFAULT_PATH)

#find_package(Ceres REQUIRED)

#find_package(catkin REQUIRED COMPONENTS
#        rosbag
#        roscpp
#        cv_bridge
#        )


find_package(PCL REQUIRED)

include_directories(
#        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIR}


)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/CameraModels SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/orb_slam3 SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/Util SRC_DIR)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/detection SRC_DIR)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/preintegration PREINTE_SRC_DIR)

add_library(${PROJECT_NAME} SHARED
        ${SRC_DIR}
        )

add_library(preint SHARED
        ${PREINTE_SRC_DIR}
        )

add_subdirectory(Thirdparty/g2o)

target_link_libraries(preint
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Boost_LIBRARIES}
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Boost_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        preint
        )

if(system_frame MATCHES "ARM32")
    target_link_libraries(${PROJECT_NAME}
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_features.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_filters.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_segmentation.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_octree.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_search.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_sample_consensus.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_kdtree.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_common.so.1.8.1
            ${CMAKE_SYSROOT}/aLibrary/pcl-1.8.1/build_for_arm/lib/libpcl_ml.so.1.8.1
            ${RKNN_API_LIB}
            dl 		#不要遗漏这一行
            )
elseif(system_frame MATCHES "Linux64")
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
    include(cmake/pangolin.cmake)
endif()

IF (DEFINED ENABLE_DBOW2)
    target_link_libraries(${PROJECT_NAME}
            ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/${lib_path}/libDBoW2.so
            )
ENDIF ()
IF (DEFINED ENABLE_DBOW3)
    target_link_libraries(${PROJECT_NAME}
            ${PROJECT_SOURCE_DIR}/Thirdparty/DBow3/${build_path}/src/libDBoW3.so
            )
ENDIF ()
IF (DEFINED ENABLE_FBOW)
    target_link_libraries(${PROJECT_NAME}
            ${PROJECT_SOURCE_DIR}/Thirdparty/fbow/${lib_path}/libfbow.so
            )
ENDIF ()

if(system_frame MATCHES "Linux64")
    # slam
    #set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
    add_executable(stereo_odo_3
            Examples/Stereo/stereo_odo_robot.cc)
    target_link_libraries(stereo_odo_3 ${PROJECT_NAME} preint)

    # obstacle detection
    add_executable(obstacle_detector
            Examples/detection/obstacle_detector.cpp)
    target_link_libraries(
            obstacle_detector
            ${PROJECT_NAME}
    )


    # object_detection + obstacle detection
    add_executable(object_detector
             Examples/detection/object_detector.cpp)
    target_link_libraries(
             object_detector
             ${PROJECT_NAME}
    )

    # main_direction_detection
    add_executable(main_direction_detection
            Examples/detection/main_direction_detecton.cpp)
    target_link_libraries(
            main_direction_detection
            ${PROJECT_NAME}
    )

    # sweepe robot
    add_executable(sweeper_robot
            Examples/sweep_robot/sweep_robot.cc)
    target_link_libraries(
            sweeper_robot
            ${PROJECT_NAME}
            preint
    )

elseif(system_frame MATCHES "ARM32")
    # slam
    #set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
    add_executable(stereo_odo_3
            Examples/Stereo/stereo_odo_robot.cc
            src/detection/yolov5/drm_func.c
            src/detection/yolov5/rga_func.c
            src/detection/yolov5/postprocess.cc)
    target_link_libraries(stereo_odo_3 ${PROJECT_NAME} preint)

    # obstacle detection
    add_executable(obstacle_detector
            Examples/detection/obstacle_detector.cpp
            src/detection/yolov5/drm_func.c
            src/detection/yolov5/rga_func.c
            src/detection/yolov5/postprocess.cc)
    target_link_libraries(
            obstacle_detector
            ${PROJECT_NAME}
    )

    # object_detection + obstacle detection
    add_executable(object_detector
            Examples/detection/object_detector.cpp
            src/detection/yolov5/drm_func.c
            src/detection/yolov5/rga_func.c
            src/detection/yolov5/postprocess.cc)
    target_link_libraries(
            object_detector
            ${PROJECT_NAME}
    )

    # main_direction_detection
    add_executable(main_direction_detection
            Examples/detection/main_direction_detecton.cpp
            src/detection/yolov5/drm_func.c
            src/detection/yolov5/rga_func.c
            src/detection/yolov5/postprocess.cc)
    target_link_libraries(
            main_direction_detection
            ${PROJECT_NAME}
    )

    # sweepe robot
    add_executable(sweeper_robot
            Examples/sweep_robot/sweep_robot.cc
            src/detection/yolov5/drm_func.c
            src/detection/yolov5/rga_func.c
            src/detection/yolov5/postprocess.cc)
    target_link_libraries(
            sweeper_robot
            ${PROJECT_NAME}
            preint
    )
endif()
