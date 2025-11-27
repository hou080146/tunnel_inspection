#ifndef RESULT_H
#define RESULT_H
#include<string>
#define CAMERANUMBER 8
#define IMGWIDTH 8192
#define IMGHEIGHT 8192
#define NETHEIGHT 640
#define NETWIDTH 640
struct result {
    result() = default;
    float mileage = 0;
    int  order = 0;//线别
    int tunnel_list = 0;//隧道号
    float angle = 0;//j角度
    float length = 0;//长度
    float width = 0;//裂纹宽度
    float area = 0;//面积
    std::string injury_image_name;//病害图像名称
    std::string img_path;
    float bar_value;//进度值
    float confidence = 0;
    bool right = true;
    bool ok = true;
    double proc_time = 0.0;
    float crack_width = 0.0;
};
#endif
