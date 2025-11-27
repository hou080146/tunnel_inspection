#ifndef ALGTHREAD_H
#define ALGTHREAD_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <queue>
#include <opencv2/opencv.hpp>
#include"result.h"
#include "file_data.h"
#include"ys_onnx.h"
#include"ys_utils.h"
#include"detect.h"


/// 算法线程
class alg_thread final
{
public:
	/// 默认构造
	alg_thread() = default;

	/// 析构
	~alg_thread();
    typedef unsigned short WORD;
    typedef struct _SYSTEMTIME {
        WORD wYear;
        WORD wMonth;
        WORD wDayOfWeek;
        WORD wDay;
        WORD wHour;
        WORD wMinute;
        WORD wSecond;
        WORD wMilliseconds;
    } SYSTEMTIME, *PSYSTEMTIME, *LPSYSTEMTIME;
    struct RefPoint {
        int type;
        char Linename[32];
        char Linetype[32];
        char Operator[32];
        char TrackDir[4];
        SYSTEMTIME stTime;
        float fLocation;
        float unitspercycle;
        int cycle;
        int RailType;
    };


    bool readRefPointFromFile(const std::string& filename, RefPoint& refPoint);

	/// 算法完成回掉函数类型
    typedef std::function<void(const result &ret, const cv::Mat& frame)> alg_finished_callback;

	/// 初始化
	bool init(alg_finished_callback alg_finished_callback);

	/// 开始检测
	bool start();

	/// 停止检测
	bool stop();

	/// 加入检测数据
	void push_frame(const file_data::frame& frame);

	/// 运行线程
	void run();

	/// 是否正在运行
	bool running() const {
		return running_;
	}

	unsigned int set_data_name(std::string files_name1, std::string  files_name2, std::string  store_files_name, std::string  result_files_name);
    std::mutex &get_mutex() {
        return mutex_;
    }
    int nearestMultipleOf100(int number);
   
private:
	/// 线程
	std::shared_ptr<std::thread> thread_ = nullptr;

    std::string files_name1_;
    std::string  files_name2_;
    std::string  store_files_name_;
    std::string  result_files_name_;

	/// 待检测数据
	std::vector<file_data::frame> frames_[CAMERANUMBER];
	///处理数据
	std::vector<file_data::frame> process_frames_;

	///有效数据位置//成功打开摄像头存储目录的摄像头ID号
	std::vector<int> effectives_;

	/// 互斥锁
	std::mutex mutex_;

	/// 条件变量
	std::condition_variable condition_;

	/// 运行标志
	bool running_ = false;

	/// 算法完成回掉函数
	alg_finished_callback alg_finished_callback_;
    /*
    [this](const result& ret, const cv::Mat& frame) {
            // cv::cvtColor(frame, frame, CV_BGR2RGB);
            //return;//test
            if (frame.empty()) {
                ui.progress_bar->setMaximum(ret.bar_value);
                return;
            }
            //显示处理耗时（毫秒）和当前进度值到time_label和signals_bar。
            ui.time_label->setText(QString::number(ret.proc_time)+"==="+ QString::number(ret.bar_value));
            signals_bar(ret.bar_value+2);

            if (!ui.radio_button->isChecked())return;

            cv::Mat tframe = frame.clone();
            cv::cvtColor(tframe, tframe, CV_BGR2RGB);//QT是RGB格式
            //将 OpenCV 图像转换为 Qt 可显示的 QImage
            QImage qimage = QImage((uchar*)tframe.data, tframe.cols, tframe.rows,
                tframe.cols * tframe.channels(), QImage::Format_RGB888);
            ui.oringinal_label->setPixmap(QPixmap::fromImage(qimage));//显示图像
        //ui.progress_bar->setValue(ret.bar_value);
        }
    */
	unsigned int frame_number_;
    ysOnnx yvonnx_;
    bool is_ysonnx_ = false;
    Yolov8Onnx yolov_detect_;
    
	
};

#endif
