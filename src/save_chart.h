#pragma once
#include<string>

#include <vector>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#pragma execution_character_set("utf-8")
class save_chart {

public:
    //±£´æexcel
    //void save_excel(vector<string>excel_list, string save_path);
    // cv::Mat colorImage
    void save_excel(vector<string>qx_label, vector<string>lc, vector<string> qx_path, string save_path, string qslc, int sxx);
    //void save_excel(vector<string>qx_label, vector<string>lc, cv::Mat colorImage, string save_path, string qslc, int sxx);
    void save_word(vector<string>qx_label, vector<string>lc, vector<string> qx_path, int tt_lj, string save_path, string qslc, int sxx);
    void split(string zfc, string image_path, string defect_category, string defect_coordinate);


};