#ifndef _FILE_DATA_H_
#define _FILE_DATA_H_

#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>
#include<thread>

#include <mutex>
#include <condition_variable>
#include"detect.h"
#include"ys_onnx.h"
#include"ys_utils.h"
class file_data
{
public:
    struct crack_result {
        cv::Rect box;
        float confidence = 0.0;
        float length = 0.0;
        float width = 0.0;

    };
	struct frame
	{
		frame() = default;
		frame(const cv::Mat& _data, int _camera_id, long long _sample_time, long long _frame_number,long mileage):
			data(_data), camera_id(_camera_id), sample_time(_sample_time), frame_number(_frame_number) , results() {}
        

		frame clone() const {
			frame ret;
			ret.data = data.clone();
			ret.camera_id = camera_id;
			ret.sample_time = sample_time;
			ret.frame_number = frame_number;
            ret.results = results;
			return ret;
		}

		cv::Mat data;			///< 帧数据
		int camera_id;			///< 相机id
		long long sample_time;	///< 帧采样开始时间
		long long frame_number; ///< 帧号
        long mileage;//里程
        std::vector<OutputSeg> results;
        int changeh_[8] = { 0,0,0,0,100,0,93,167 };
	

	};



	/// 构造
	file_data();

	/// 析构
	~file_data();


	/// 定义帧回调函数类型
	typedef std::function<void(frame&)> frame_ready_callback;



	/// 初始化，将帧回调函数传入init
	bool init(frame_ready_callback frame_ready_callback, const std::string& file_name = std::string(), int id = 0);

	/// 打开文件
	bool start();

	void run();

	bool stop();
	bool read_images(const std::string &path, std::vector<std::string> &image_files);


	/// 获得相机id
	int id() const {
		return id_;
	};
	void set_params(const std::string path, const std::string result_path, bool is_save) {
		path_ = path;
		is_save_ = is_save;
        result_files_name_ = result_path;
	}

private:
	/// 相机id
	int id_ = 0;

	/// 
	std::string file_name_;

	/// 定义帧回调函数
	frame_ready_callback  frame_ready_callback_ = nullptr;
    /*回调函数如下
    [this](file_data::frame& frame) {
        alg_thread_.push_frame(frame.clone());// 1. 把克隆的帧传给算法线程处理（比如后续的识别算法）
        if (camera_id_ == -1) {
            camera_id_ = frame.camera_id; // 2. 如果camera_id_是-1（还没赋值），就设置为当前帧的相机ID
        }
        if (camera_id_ == frame.camera_id&&ui.save_radio_button->isChecked()) {
            signals_bar(frame.frame_number + 2); // 3. 如果当前帧是选中的摄像头帧且保存选项被勾选，更新进度条
        }
    },
        files_names[i],
        i);
    */
	std::thread thread_;
	bool running_ = false;
	bool is_save_ = false;
	std::string path_;
	std::mutex mutex_;
	std::condition_variable condvar_;
    Yolov8Onnx yd_;//废弃方案

    ysOnnx yvonnx_;//用于分割，对特定缺陷（例如裂缝）的像素级轮廓做识别。输出掩码（mask），标出哪些像素属于裂缝。后续精确计算：裂缝长度（骨架化），渗水面积（像素统计）
    //int changeh_[8] = { 5500,5400,5355,5344,5379,5317,5349,5282 };
    std::string  result_files_name_;
    int changeh_[8] = { 0 };//{ 218,118,73,62,97,35,97,0 };
};

#endif