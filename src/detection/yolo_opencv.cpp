//YOLOv3 on OpenCV
//reference：https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
//by:Andyoyo@swust
//data:2018.11.20

#include "detection/yolo_opencv.h"
//#define ros_yolo_pub //发布物体识别检测结果图片
//#define yolo_mask_pub //发布物体识别检测结果mask

#ifdef ros_yolo_pub
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
ros::Publisher yolo_pub_;
#endif

#ifdef yolo_mask_pub
ros::Publisher yolo_mask_pub_;
#endif

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold

//void yolo_det(string root_path, Mat frame, int count, vector<string> classes, cv::dnn::Net net, vector<bbox>& bbox_list, int v0, Mat& object_mask)
//{
//// Process frames.
//    auto start1 = std::chrono::steady_clock::now();
//    // Create a 4D blob from a frame.
//    cv::Mat blob;
//    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
//
//    //Sets the input to the network
//    net.setInput(blob);
//
//    // Runs the forward pass to get output of the output layers
//    std::vector<cv::Mat> outs;
//    net.forward(outs, getOutputsNames(net));
//
//    // Remove the bounding boxes with low confidence
//    postprocess(frame, outs, bbox_list, classes);
//
//#ifdef yolo_mask_pub
//    // get yolo mask
//    for(int v = 0 ;v<frame.rows;v++)
//    {
//        for(int u = 0; u<frame.cols;u++)
//        {
//            for(int i=0; i<bbox_list.size();i++)
//            {
//                bbox b2d = bbox_list[i];
//                if ((u>b2d._xmin)&&(u<b2d._xmax)&&(v>b2d._ymin)&&(v<b2d._ymax))
//                {
//                    object_mask.at<unsigned char>(v, u)=255;
//                }
//
//            }
//        }
//    }
//    //publish rectified images
//    sensor_msgs::ImagePtr yolo_mask_img = cv_bridge::CvImage(std_msgs::Header(), "mono8", object_mask).toImageMsg();
//    yolo_mask_pub_.publish(yolo_mask_img);
//#endif
//
//
//    auto end1 = std::chrono::steady_clock::now();
//    std::chrono::duration<double, std::micro> elapsed1 = end1 - start1; // std::micro 表示以微秒为时间单位
//    cout<<"time for yolo detection: "<<elapsed1.count()/1000000<<endl;
//
//    // Write the frame with the detection boxes
//#ifdef ros_yolo_pub
//    //publish rectified images
//    sensor_msgs::ImagePtr yolo_det_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
//    yolo_pub_.publish(yolo_det_img);
//#endif
//}


// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net &net) {
    static std::vector<cv::String> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

#ifdef RKNN_MODEL

// Draw the predicted bounding box
void drawPred(int classId, float conf, double obj_h, double obj_d, int left, int top, int right, int bottom, cv::Mat &frame, vector<string> classes) {
    //Draw a rectangle displaying the bounding box
    int b, g, r;
    if (classId == 0){
        b = 255;
        g = 0;
        r = 0;
    }
    if (classId == 1){
        b = 255;
        g = 255;
        r = 0;
    }
    if (classId == 2){
        b = 255;
        g = 0;
        r = 255;
    }
    if (classId == 3){
        b = 0;
        g = 255;
        r = 255;
    }
    if (classId == 4){
        b = 0;
        g = 255;
        r = 0;
    }

    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b, g, r), 3);

    //Get the label for the class name and its confidence
    std::string label = cv::format("h: %.3f, d: %.3f", obj_h, obj_d);
//    cout << "class ID:   " << classId << endl;
//    cout << "size:  " << (int) classes.size() << endl;
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
//        label = "h:" + to_string(obj_h) + " " + classes[classId];
//        label = classes[classId];
    } else {
        std::cout << "classes is empty..." << std::endl;
    }

    //Display the label at the top of the bounding box
    int baseLine;
//    fontFace:   cv::FONT_HERSHEY_SIMPLEX
    cv::Size labelSize = cv::getTextSize(label, 1, 2, 3, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left - 20, top), 1, 2, cv::Scalar(255, 255, 255), 2);

}

#else
// Remove the bounding boxes with low confidence using non-maxima suppr_ession
void postprocess(Mat &frame, vector<Mat> &outs, vector<bbox> &bbox_list, vector<string> classes) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    int _c;
    int _xmin;
    int _ymin;
    int _xmax;
    int _ymax;
    int _index;
    int _pair_index;
    double _score;
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        bbox box2d(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height);
        bbox_list.push_back(box2d);
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classes);
    }
}

// Draw the predicted bounding box
void
drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, vector<string> classes) {
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0));

    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ":" + label;
    } else {
        std::cout << "classes is empty..." << std::endl;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 2, 3, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 0, 0));

}

#endif