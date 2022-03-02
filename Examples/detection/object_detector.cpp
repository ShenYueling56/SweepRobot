//
// Created by shenyl on 2020/11/1.
//

#ifdef RKNN_MODEL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include "iostream"

#define _BASETSD_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"
#include "postprocess.h"

#include "/home/houj/sweepSLAM/include/detection/detector.h"
#include <fstream>
#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/
using namespace std;

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
           attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenamesLeft, vector<string> &vstrImageFilenamesRight,
           vector<double> &vTimestamps);

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{

    if (argc != 2) {
        cerr << endl << "Usage: ./object_detector path_to_config_file" << endl;
        return 1;
    }
    string config_file = argv[1];
    //image path
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    string data_path;
    // result save path
    fs_read["data_path"] >> data_path;
    string save_path;
    fs_read["save_path"] >> save_path;
    string model_path;
    fs_read["rknn_model_path"] >> model_path;

    ofstream obs_boxes_file;//创建文件
    obs_boxes_file.open(save_path + "obs_boxes.txt");

    ofstream obj_boxes_file;//创建文件
    obj_boxes_file.open(save_path + "obj_boxes.txt");

    cout << "Loading Images ..." << endl;
    // Retrieve paths to images
    vector<string> vstrImageFilenamesLeft, vstrImageFilenamesRight;
    vector<double> vTimestamps;
    LoadImages(data_path, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);

    cout << "Finish Loading Images ..." << endl;
    int nImages = vstrImageFilenamesLeft.size();

    // Create Detector.
    Detector detector(config_file);

    // Vector for tracking time statistics
    vector<float> vTimesDetect;
    vTimesDetect.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im_l;
    cv::Mat im_r;
    cv::Mat orig_img;


    int status = 0;

    unsigned int handle;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    int model_data_size = 0;
    int ret, model_img_width, model_img_height;

    //added
    bool isNCHW = false;

//  Create the neural network
    printf("Loading rknn model...\n");

//    model_name = "/data/ttt/rknn_yolov5_myset/model/yolo_self_onnx.rknn";      //rknn model path
    const char* model_name = model_path.data();
    unsigned char *model_data = load_model(model_name, &model_data_size);
    rknn_context ctx;
    rknn_input_output_num io_num;    //including input and output number
    rknn_tensor_attr input_attrs[1];   //input
    rknn_tensor_attr output_attrs[3];     //output
    rknn_input inputs[1];

    detector.rknn_initialization(model_data, model_data_size, model_img_width, model_img_height, ctx, io_num, inputs, input_attrs, output_attrs, isNCHW);

    printf("Model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
    printRKNNTensor(&(input_attrs[0]));
    printRKNNTensor(&(output_attrs[0]));
    printRKNNTensor(&(output_attrs[1]));
    printRKNNTensor(&(output_attrs[2]));
    printf("Model input height=%d, width=%d, channel=%d\n", model_img_height, model_img_width, 3);

    double carpet_tp_count = 0;
    int test_frames_num = 0;

//    nImages = 4500;
    int start_ni = 0;
//    nImages = 40;
//    int last_ni = start_ni - 14;
    for (int ni = start_ni; ni < nImages; ni++) {

//        if (ni - last_ni != 14) continue;
//        if ((ni > 3932 && ni < 4000) || (ni >= 5600 && ni <= 5605) || (ni >= 6184 && ni < 6200)){
//            last_ni = ni;
//            continue;
//        }
        cout << endl;
        cout << "/////////////////Frame" << ni << "///////////////////" << endl;

//        cv::Mat orig_img = cv::imread("/data/ttt/rectify.jpg", CV_LOAD_IMAGE_UNCHANGED);
        orig_img = cv::imread(vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        if (!orig_img.data) {
            printf("cv::imread fail!\n");
            return -1;
        }

        im_r = cv::imread(vstrImageFilenamesRight[ni], CV_LOAD_IMAGE_UNCHANGED);

//        img_width = orig_img.cols;
//        img_height = orig_img.rows;
//        printf("img width = %d, img height = %d\n", img_width, img_height);

        double tframe = vTimestamps[ni];

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//        cv::Mat resize_M;
//        cv::resize(orig_img, resize_M, cv::Size(width, height), (0, 0),
//                   (0, 0), cv::INTER_LINEAR);

//        void *in_data = NULL;
//        ProcessInput(resize_M, &in_data, &(input_attrs[0]), isReorder210, isNCHW);

//        inputs[0].buf = in_data;


        detector.run(orig_img, im_r, ni, ctx, io_num.n_input, io_num.n_output,
                     model_img_width, model_img_height, inputs, input_attrs, output_attrs, isNCHW);


        vector<AABB_box3d> obj_boxes = detector.obj_boxes;
        vector<AABB_box3d> obs_boxes = detector.obs_boxes;

        cout << "Total number of objects after verification: " << obj_boxes.size() << endl;
        for (int i = 0; i < obj_boxes.size(); i++) {
            obj_boxes_file << ni << " " << 1 << " " << obj_boxes[i]._c << " " << obj_boxes[i]._position_x << " "
                           << obj_boxes[i]._position_y << " " << obj_boxes[i]._position_z << " " << obj_boxes[i]._width
                           << " " << obj_boxes[i]._height << " " << obj_boxes[i]._x_min << " " << obj_boxes[i]._x_max <<
                           " " << obj_boxes[i]._y_min << " " << obj_boxes[i]._y_max << endl;
        }

        cout << "Total number of obstacles: " << obs_boxes.size() << endl;
        for (int i = 0; i < obs_boxes.size(); i++) {
//            cout << "位置(x,y,z)： " << obs_boxes[i]._position_x << ", " << obs_boxes[i]._position_y << ", "
//                 << obs_boxes[i]._position_z;
//            cout << " 尺寸(w,h)： " << obs_boxes[i]._width << ", " << obs_boxes[i]._height << endl;
            obs_boxes_file << ni << " " << obs_boxes[i]._position_x << " " << obs_boxes[i]._position_y << " "
                           << obs_boxes[i]._position_z << " " << obs_boxes[i]._width << " " << obs_boxes[i]._height
                           << endl;
        }

//        cout << "obj_boxes.size " << obj_boxes.size() << endl;
//        for (int i = 0; i < obj_boxes.size(); i++) {
//            obj_boxes_file << ni << " " << obj_boxes[i]._c << " " << endl;
//            if (obj_boxes[i]._c) carpet_tp_count += 1;
//        }
//                last_ni = ni;
//        test_frames_num += 1;
//

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        free(detector.qnt_buf);
        free(detector.nchw_buf);

        double tdetect = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        cout << "tdetection_total: " << tdetect << endl << endl;
        vTimesDetect[ni - start_ni] = tdetect;

        // Wait to load the next frame
//        double T = 0;
//        if (ni < nImages - 1)
//            T = vTimestamps[ni + 1] - tframe;
//        else if (ni > 0)
//            T = tframe - vTimestamps[ni - 1];
//
//        if (tdetect < T)
//            usleep((T - tdetect) * 1e6);
    }

//    cout << "Carpet Recall: " << (carpet_tp_count / test_frames_num) << endl;

    // Tracking time statistics
//    sort(vTimesDetect.begin(), vTimesDetect.end());
    float totaltime = 0;
    for (int ni = 0; ni < (nImages - start_ni); ni++) {
        totaltime += vTimesDetect[ni];
    }

    obj_boxes_file.close();
    obs_boxes_file.close();

    // release

    ret = rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }

//    cout << "done" << endl;

    return 0;
}

#else

#include "detection/detector.h"
#include <fstream>

using namespace std;

void
LoadImages(const string &strSequence, vector<string> &vstrImageFilenamesLeft, vector<string> &vstrImageFilenamesRight,
           vector<double> &vTimestamps);

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl << "Usage: ./obstacle_detector path_to_config_file" << endl;
        return 1;
    }
    string config_file = argv[1];
    //image path
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    string data_path;
    // result save path
    fs_read["data_path"] >> data_path;
    string save_path;
    fs_read["save_path"] >> save_path;
    ofstream obj_boxes_file;//创建文件
//    ofstream obs_boxes_file;//创建文件
    obj_boxes_file.open(save_path + "obj_boxes.txt");
//    obs_boxes_file.open(save_path + "obs_boxes.txt");

    // Retrieve paths to images
    vector<string> vstrImageFilenamesLeft, vstrImageFilenamesRight;
    vector<double> vTimestamps;
    LoadImages(data_path, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    int nImages = vstrImageFilenamesLeft.size();

    // Create Detector.
    Detector detector(config_file);

    // Vector for tracking time statistics
    vector<float> vTimesDetect;
    vTimesDetect.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im_l;
    cv::Mat im_r;
    for (int ni = 0; ni < nImages; ni++) {
//        cout<<"/////////////////frame"<<ni<<"///////////////////"<<endl;
        // Read image from file
        im_l = cv::imread(vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        im_r = cv::imread(vstrImageFilenamesRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im_l.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenamesLeft[ni] << endl;
            return 1;
        }
        if (im_r.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenamesRight[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the Detector system
        detector.run(im_l, im_r, ni);
        //获得障碍物框和物体框
        vector<AABB_box3d> obs_boxes = detector.obs_boxes;
        vector<AABB_box3d> obj_boxes = detector.obj_boxes;
//        cout << "obs result:" << endl;
//        for (int i = 0; i < obs_boxes.size(); i++) {
//            cout << "位置(x,y,z)： " << obs_boxes[i]._position_x << ", " << obs_boxes[i]._position_y << ", "
//                 << obs_boxes[i]._position_z;
//            cout << " 尺寸(w,h)： " << obs_boxes[i]._width << ", " << obs_boxes[i]._height << endl;
//            obs_boxes_file << ni << " " << obs_boxes[i]._position_x << " " << obs_boxes[i]._position_y << " "
//                           << obs_boxes[i]._position_z << " " << obs_boxes[i]._width << " " << obs_boxes[i]._height
//                           << endl;
//        }
//        cout << "obj result" << endl;
        for (int i = 0; i < obj_boxes.size(); i++) {
//            cout << "位置(x,y,z)： " << obj_boxes[i]._position_x << ", " << obj_boxes[i]._position_y << ", "
//                 << obj_boxes[i]._position_z;
//            cout << " 尺寸(w,h)： " << obj_boxes[i]._width << ", " << obj_boxes[i]._height << endl;
            obj_boxes_file << ni << " " << 1 << " " << obj_boxes[i]._c << " " << obj_boxes[i]._position_x << " "
                           << obj_boxes[i]._position_y << " " << obj_boxes[i]._position_z << " " << obj_boxes[i]._width
                           << " " << obj_boxes[i]._height << endl;
        }

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();


        double tdetect = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesDetect[ni] = tdetect;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (tdetect < T)
            usleep((T - tdetect) * 1e6);
    }


    // Tracking time statistics
    sort(vTimesDetect.begin(), vTimesDetect.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesDetect[ni];
    }
    cout << "Finish processing" << endl;
    cout << "-------" << endl << endl;
//    cout << "median tracking time: " << vTimesDetect[nImages / 2] << endl;
//    cout << "mean tracking time: " << totaltime / nImages << endl;

//    obs_boxes_file.close();
    obj_boxes_file.close();

    return 0;
}
#endif

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
    }
}