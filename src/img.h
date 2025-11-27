#ifndef _IMG_H_
#define _IMG_H_

#include <opencv2/opencv.hpp>
namespace img {
    void get_next_minloc(cv::Mat &result, cv::Point minloc, int maxvalue, int templatw, int templath);//赋值最大值到模板匹配结果中，便于查找下一个最相似图片
    cv::Mat get_histogram_image(const cv::Mat &image,int width);//输出图像直方图
    cv::Mat get_gray_histogram(const cv::Mat & image, int &position);//输出总图片的差值直方图，并输出差值到达阈值的位置到position 
    void get_templet_position(const cv::Mat &image, const cv::Mat &tempat_image, int &position);
    void get_templet_position(const cv::Mat &image, const cv::Mat &tempat_image, cv::Point &position1, cv::Point &position2,std::vector<cv::Mat> &images );
    cv::Mat get_binary_histogram(const cv::Mat & image, int &position);//输出每行值为零个数的直方图，并输出差值到达阈值的位置到position
    cv::Mat get_cols_histogram(const cv::Mat & image, int &position1,int &position2);//输出每列值为零个数的直方图，并输出差值到达阈值的位置到position
    /// @brief 获取水平方向钢轨区域
    /// @params [in]    image    输入图像
    /// @params [out]   rail_img 水平方向图像
    /// @params [out]   position 水平方向位置

    /// @return 结果直方图
    cv::Mat get_rail_img(const cv::Mat & image, cv::Mat &rail_img,cv::Rect &position);
    /// @brief 获取轨枕区域
    /// @params [in]    image    输入图像
    /// @params [out]   sleeper_image 水平方向图像
    /// @params [out]   position 水平方向位置

    /// @return 结果直方图
    cv::Mat get_sleeper_img(const cv::Mat & image, cv::Mat &sleeper_image, int &position);

    	/// @brief 带通二值化
	/// @params [in] src   输入图像
	/// @params [out] dst   输出图像
	/// @params [in] lbound  低阈值
	/// @params [in] ubound 高阈值
	void threshold_bound(const cv::Mat& src, cv::Mat& dst, uchar lbound, uchar ubound);


        	/// @brief 带通二值化
	/// @params [in] src   输入图像
	/// @params [out] dst   输出图像
	/// @params [in] lbound  低阈值
	/// @params [in] ubound 高阈值
    cv::Mat threshold_bound(const cv::Mat& src, uchar lbound, uchar ubound);

    /// @brief 合并矩形框
    /// @params [in] rect1 rect2  输入
    cv::Rect get_combine_rect(const cv::Rect &rect1, const cv::Rect &rect2);

    void delete_image_line(cv::Mat &image);
    void zero_image(cv::Mat &image);
    cv::Mat get_fastener_position(const cv::Mat & image, int &position);//输出每行值为零个数的直方图，并输出差值到达阈值的位置到position

    cv::Mat get_fastener_position2(const cv::Mat & image, int &position);//输出每行值为零个数的直方图，并输出差值到达阈值的位置到position

    cv::Mat adjust_fastener_position(const cv::Mat & image, int &position);//校正扣件图位置，并输出差值到达阈值的位置到position
	cv::Mat get_fastener_positions(const cv::Mat & image, std::vector<int>& positions);
    // 骨架化函数
    cv::Mat skeletonize(const cv::Mat& src);
    void calculate_crack_dimensions(const cv::Mat& mask,const cv::Mat& skeleton, float &length, float &width);
    cv::Mat removeSmallComponents(const cv::Mat& image, int minSize);
    bool isValid(int x, int y, int rows, int cols);
    int countNonZeroNeighbors(const cv::Mat& img, int x, int y);
    int countTransitions(const cv::Mat& img, int x, int y);
    cv::Mat zhangSuenThinning(const cv::Mat& binaryImage);
}

#endif