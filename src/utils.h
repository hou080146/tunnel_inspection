#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <opencv2/opencv.hpp>

struct OutputDet {
    int id;
    float confidence;
    cv::Rect box;


    bool isAdjacentOrOverlapping(const OutputDet& other) const {
        int dx = std::max(0, std::max(other.box.x - (box.x + box.width), box.x - (other.box.x + other.box.width)));
        int dy = std::max(0, std::max(other.box.y - (box.y + box.height), box.y - (other.box.y + other.box.height)));
        return (dx <= 20 && dy <= 20) || (box & other.box).area() > 0;
    }



    OutputDet merge(const OutputDet& other) const {
        int new_x = std::min(box.x, other.box.x);
        int new_y = std::min(box.y, other.box.y);
        int new_width = std::max(box.x + box.width, other.box.x + other.box.width) - new_x;
        int new_height = std::max(box.y + box.height, other.box.y + other.box.height) - new_y;
        return { id, std::max(confidence, other.confidence), cv::Rect(new_x, new_y, new_width, new_height) };
    }
};

void DrawPred(cv::Mat& img, std::vector<OutputDet> result,
    std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
    cv::Vec4d& params,
    const cv::Size& newShape = cv::Size(640, 640),
    bool autoShape = false,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114)); 
#endif
