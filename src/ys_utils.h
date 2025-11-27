#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>

#define YOLO_P6 false //是否使用P6模型
#define ORT_OLD_VISON 12  //ort1.12.0 之前的版本为旧版本API
struct OutputSeg {
    int id;             // 结果类别id
    float confidence;   // 结果置信度
    cv::Rect box;       // 矩形框
    cv::Mat boxMask;    // 矩形框内mask，节省内存空间和加快速度

    bool isAdjacentOrOverlapping(const OutputSeg& other) const {
        int dx = std::max(0, std::max(other.box.x - (box.x + box.width), box.x - (other.box.x + other.box.width)));
        int dy = std::max(0, std::max(other.box.y - (box.y + box.height), box.y - (other.box.y + other.box.height)));
        return (dx <= 20 && dy <= 20) || (box & other.box).area() > 0;
    }

    OutputSeg merge(const OutputSeg& other) const {
        int new_x = std::min(box.x, other.box.x);
        int new_y = std::min(box.y, other.box.y);
        int new_width = std::max(box.x + box.width, other.box.x + other.box.width) - new_x;
        int new_height = std::max(box.y + box.height, other.box.y + other.box.height) - new_y;
        cv::Rect new_box(new_x, new_y, new_width, new_height);

        // 创建新的mask
        cv::Mat new_boxMask = cv::Mat::zeros(new_height, new_width, boxMask.type());

        // 将当前的boxMask复制到新的位置
        boxMask.copyTo(new_boxMask(cv::Rect(box.x - new_x, box.y - new_y, box.width, box.height)));

        // 将other的boxMask复制到新的位置
        other.boxMask.copyTo(new_boxMask(cv::Rect(other.box.x - new_x, other.box.y - new_y, other.box.width, other.box.height)));

        return { id, std::max(confidence, other.confidence), new_box, new_boxMask };
    }
};
struct MaskParams {
    int segChannels = 32;
    int segWidth = 160;
    int segHeight = 160;
    int netWidth = 640;
    int netHeight = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;

};
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
void DrawPred_seg(cv::Mat& img, std::vector<OutputSeg> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox_seg(const cv::Mat& image, cv::Mat& outImage,
    cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
    const cv::Size& newShape = cv::Size(640, 640),
    bool autoShape = false,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputSeg>& output, const MaskParams& maskParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);
