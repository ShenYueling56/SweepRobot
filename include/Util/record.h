//
// Created by qzj on 2020/10/25.
//

#ifndef SRC_RECORD_H
#define SRC_RECORD_H

#include <Eigen/Core>
#include <vector>

class cTime {
public:
    cTime() {
        sum = 0.0;
        cnt = 0;
        times.reserve(2000);
    }

    inline float update(float time) {
        sum = sum + time;
        cnt++;
        times.push_back(time);
        return sum / cnt;
    }

    inline float get() {
        return sum / cnt;
    }

    inline float getMax() {
        int n = times.size();
        float max_t = 0.0;
        for (int i = 0; i < n; ++i) {
            if (times[i] > max_t) {
                max_t = times[i];
            }
        }
        return max_t;
    }

    float sum;
    uint32_t cnt;
    std::vector<float> times;
};

class LocalMapTime {

public:
    LocalMapTime() {
        procKF = cTime();
        MPcull = cTime();
        CheckMP = cTime();
        searchNeigh = cTime();
        Opt = cTime();
        KF_cull = cTime();
        Insert = cTime();
        kf_cnt.reserve(2000);
        timesCur.reserve(2000);
    }

    cTime procKF;
    cTime MPcull;
    cTime CheckMP;
    cTime searchNeigh;
    cTime Opt;
    cTime KF_cull;
    cTime Insert;
    std::vector<int> kf_cnt;
    std::vector<double> timesCur;

};

class LoopCloseTime {

public:
    LoopCloseTime() {
        Detect = cTime();
        Merge = cTime();
        Loop = cTime();
        timesCur.reserve(2000);
    }

    cTime Detect;
    cTime Merge;
    cTime Loop;
    std::vector<double> timesCur;

};

class TrackingTime {

public:
    TrackingTime() {
        ExtractFeature = cTime();
        StereoMatch = cTime();
        TrackFrame = cTime();
        TrackMap = cTime();
        PostTrack = cTime();
        timesCur.reserve(10000);
    }

    cTime ExtractFeature;
    cTime StereoMatch;
    cTime TrackFrame;
    cTime TrackMap;
    cTime PostTrack;
    std::vector<double> timesCur;

};

void SetTimeNow(float time);

float GetTimeNow();

template<typename T>
T GetMax(std::vector<T> vec) {
    int n = vec.size();
    T max_t = 0.0;
    for (int i = 0; i < n; ++i) {
        if (vec[i] > max_t) {
            max_t = vec[i];
        }
    }
    return max_t;
}

template<typename T>
float GetMean(std::vector<T> vec) {
    int n = vec.size();
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum = sum + vec[i];
    }
    return sum / n;
}

#endif //SRC_RECORD_H
