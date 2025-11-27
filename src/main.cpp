#include "tunnel_inspection.h"
#include <QtWidgets/QApplication>
#include"img.h"
using namespace cv;
using namespace std;

//void compressJpeg(const std::string& inputPath, const std::string& outputPath, int quality = 75) {
//    // 读取图像
//    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        std::cerr << "Error: Could not open or find the image!" << std::endl;
//        return;
//    }
//
//    // 设置 JPEG 压缩参数
//    std::vector<int> compressionParams;
//    compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
//    compressionParams.push_back(quality); // 质量参数，范围为 0-100
//
//    // 保存图像
//    if (!cv::imwrite(outputPath, image, compressionParams)) {
//        std::cerr << "Error: Could not save the image!" << std::endl;
//    }
//    else {
//        std::cout << "Image compressed and saved successfully." << std::endl;
//    }
//}

int main(int argc, char *argv[])
{     


    /*
    
    std::string videoFilePath = "I:/test3.mp4";  // 替换为你的视频文件路径
    VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件: " << videoFilePath << endl;
        return -1;
    }

    Mat prevGray, gray, frame;
    vector<Point2f> prevPts, nextPts;
    vector<uchar> status;
    vector<float> err;

    // 读取第一帧并转换为灰度图
    cap >> frame;
    cvtColor(frame, prevGray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(prevGray, prevPts, 100, 0.3, 7);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 转换为灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 计算光流
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, nextPts, status, err);

        // 遍历每个点，计算运动向量
        for (size_t i = 0; i < nextPts.size(); i++) {
            if (status[i]) {
                line(frame, prevPts[i], nextPts[i], Scalar(0, 255, 0), 2);
                circle(frame, nextPts[i], 5, Scalar(0, 255, 0), -1);
            }
        }

        // 显示结果
        imshow("Optical Flow", frame);

        // 判断是否存在障碍物
        int movingDownCount = 0;
        for (size_t i = 0; i < nextPts.size(); i++) {
            if (status[i] && (nextPts[i].y - prevPts[i].y) > 2) {
                movingDownCount++;
            }
        }

        if (movingDownCount > nextPts.size() * 0.5) {
            cout << "检测到障碍物" << endl;
        }

        // 更新上一帧
        std::swap(prevGray, gray);
        std::swap(prevPts, nextPts);

        if (waitKey(30) >= 0) break;
    }

    return 0;



    */






    //std::string inputPath = "d:/1_2024.10.24_35.jpg";
    //std::string outputPath = "d:1_2024.10.24_35.jpg";

    //// 调用压缩函数
    //compressJpeg(inputPath, outputPath);
    
  /*
    using clock = std::chrono::high_resolution_clock;
    clock::time_point now, last;
    clock::duration duration_alg, duration_capture, duration_capture1, duration_capture2;
    //Yolov8Onnx yd;
    ysOnnx yvo;

    yvo.ReadModel("seepage_block114.onnx", 0, 0, true);
    while (1) {
        last = clock::now();
        auto timg = cv::imread("d:/2.jpg");
       // cv::resize(timg, timg, cv::Size(640, 640));
        //cv::imwrite("d:/test.jpg", timg);
        //cv::cvtColor(timg, timg, cv::COLOR_GRAY2BGR);
        std::vector<OutputSeg> output, output2;


        std::vector<cv::Scalar> color;
        for (int i = 0; i < 80; i++) {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            color.push_back(cv::Scalar(0, 0, 255));
        }

        bool find = yvo.OnnxDetect(timg, output);

        if (find) {
            DrawPred_seg(timg, output, yvo._className, color);

            cv::resize(timg, timg, cv::Size(1024, 1024));
            

            // of << ret.area << "," <<ret.angle<< "," <<ret.bar_value<<","<<ret.injury_image_name<<","<<ret.length<<","<< ret.mileage<<","<<ret.ok<<","<<ret.order<<","<<ret.proc_time<<","<<std::endl;
        }

        cv::imshow("1", timg);
        //cv::imwrite("d:/mask.jpg", output[0].boxMask);
      
        cv::waitKey();
    }
    
    
      
    
    return 0;
    */
    QApplication a(argc, argv);
    tunnel_inspection w;
    w.show();     
    return a.exec();
}
