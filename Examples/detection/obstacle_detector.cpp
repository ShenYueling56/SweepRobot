//
// Created by shenyl on 2020/10/29.
//

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
    char result_file[20];
    sprintf(result_file, "obs_boxes.txt");
    ofstream obs_boxes_file;//创建文件
    obs_boxes_file.open(save_path + "obs_boxes.txt");


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

#ifdef RKNN_MODEL
    rknn_context ctx;
    rknn_tensor_attr input_attrs[1];   //input
    rknn_tensor_attr output_attrs[3];     //output
    rknn_input inputs[1];
#endif

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
#ifdef RKNN_MODEL
        detector.run(im_l, im_r, ni, ctx, 0, 0, 0, 0, inputs, input_attrs, output_attrs, true);
#else
        detector.run(im_l, im_r, ni);
#endif
        //获得障碍物框
        vector<AABB_box3d> obs_boxes = detector.obs_boxes;
//        cout << "result:" << endl;
        for (int i = 0; i < obs_boxes.size(); i++) {
//            cout << "位置(x,y,z)： " << obs_boxes[i]._position_x << ", " << obs_boxes[i]._position_y << ", "
//                 << obs_boxes[i]._position_z;
//            cout << " 尺寸(w,h)： " << obs_boxes[i]._width << ", " << obs_boxes[i]._height << endl;
            obs_boxes_file << ni << " " << obs_boxes[i]._position_x << " " << obs_boxes[i]._position_y << " "
                           << obs_boxes[i]._position_z << " " << obs_boxes[i]._width << " " << obs_boxes[i]._height
                           << endl;
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

    obs_boxes_file.close();

    return 0;
}

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