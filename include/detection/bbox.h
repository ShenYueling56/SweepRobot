//
// Created by shenyl on 2020/6/27.
//

#ifndef OBJECT_DETECTION_BBOX_H
#define OBJECT_DETECTION_BBOX_H


class bbox {
public:
    int _c;
    int _xmin;
    int _ymin;
    int _xmax;
    int _ymax;
    int _index;
    int _pair_index;
    double _score;
public:
    bbox(int c = -1, double score = -1, int xmin = -1, int ymin = -1, int xmax = -1, int ymax = -1) {
        _c = c;
        _xmin = xmin;
        _ymin = ymin;
        _xmax = xmax;
        _ymax = ymax;
        _score = score;
        _index = -1;
        _pair_index = -1;
    }
};


#endif //OBJECT_DETECTION_BBOX_H
