#include "detection/carpet_detector.h"
#include <chrono>

//读取目录下所有的图片
void
LoadImages(const string &strSequence, vector<string> &vstrImageFilenamesLeft, vector<string> &vstrImageFilenamesRight,
           vector<double> &vTimestamps);

int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << endl << "Usage: ./carpet_detector path_to_config_file" << endl;
        return 1;
    }
    string config_file = argv[1];
    cout << "Carpet Process..." << endl;

    //读取图片
    string file_num_str, left_img_path, right_img_path;

    int read_num = 0;
    bool show_process_flag = false;   //是否需要可视化


    Carpet_detector det1(config_file);

    //Detector Part Initialization
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    string data_path;
    fs_read["data_path"] >> data_path;
    fs_read.release();

    Detector det2(config_file);
    //End

    // 读取图片
    // Retrieve paths to images
    vector<string> vstrImageFilenamesLeft, vstrImageFilenamesRight;
    vector<double> vTimestamps;
    LoadImages(data_path, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    int nImages = vstrImageFilenamesLeft.size();

    auto start = chrono::steady_clock::now();
    Mat rectified_with_carpet;

    for (int i = 0; i < nImages; i++) {
        // Read image from file
//        cout << vstrImageFilenamesLeft[i] << endl;
//        cout << vstrImageFilenamesRight[i] << endl;
        Mat im_l = cv::imread(vstrImageFilenamesLeft[i], CV_LOAD_IMAGE_UNCHANGED);
        Mat im_r = cv::imread(vstrImageFilenamesRight[i], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[i];

        //obstacle_detector
        det2.obstacle_det(im_l, im_r, read_num);
        cv::Mat mask_iou = cv::Mat(det2.obstacle_mask, cv::Rect(0, det1.start_y, det2.obstacle_mask.cols,
                                                                det2.obstacle_mask.rows - det1.start_y));
        cout << "finish obstacle detection" << endl;
        //carpet_detector
        det1.run(im_l, mask_iou, show_process_flag);
//        (det1.carpet_region_after_mask).copyTo(mask1);
//
//        bitwise_and(det1.rectified_left_roi, det1.rectified_left_roi, carpet_region_after_mask, det1.carpet_region_after_mask);
//        bitwise_and(carpet_region_after_mask1, carpet_region_after_mask1, carpet_region_after_mask12, mask2_iou);

//        imshow("origin img", im_l);
//        imshow("obstacle Region", det2.obstacle_mask);
//        imshow("Carpet Region", det1.carpet_region_after_mask);
//        waitKey(20);

        read_num += 1;
        if (det1.quit_flag == true) break;
    }
    auto end_all = chrono::steady_clock::now();
    chrono::duration<double, micro> elapsed_all = end_all - start; // std::micro 表示以微秒为时间单位

//    cout << "Carpet Detected Number: " << det1.detected_times << endl;
//    cout << "Image Number: " << read_num << endl;
//    cout << "Accuracy(Assume that each image includes at least one carpet): "
//         << (det1.detected_times / float(read_num)) * 100 << "%" << endl;
//    cout << "Process Time: " << elapsed_all.count() / 1000000 << "s" << endl;
//    cout << "Process Time Per Image: " << (elapsed_all.count() / 1000000) / read_num << "s" << endl;

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