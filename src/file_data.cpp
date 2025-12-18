#pragma execution_character_set("utf-8")
#include "file_data.h"
#include<vector>
#include<string>
#include"result.h"
#include<QDebug>
#include"img.h"

//合并重叠或相邻的矩形框。
inline std::vector<OutputSeg> mergeRectangles(const std::vector<OutputSeg>& rects) {
    std::vector<OutputSeg> merged;
    std::vector<bool> merged_flags(rects.size(), false);

    for (size_t i = 0; i < rects.size(); ++i) {
        if (merged_flags[i]) continue;
        OutputSeg current = rects[i];
        for (size_t j = i + 1; j < rects.size(); ++j) {
            if (merged_flags[j]) continue;
            if (current.isAdjacentOrOverlapping(rects[j])) {
                current = current.merge(rects[j]);
                merged_flags[j] = true;
            }
        }
        merged.push_back(current);
    }
    return merged;
}
file_data::file_data()
{
}   
   
file_data::~file_data()
{
    stop();
}

bool file_data::init(frame_ready_callback frame_ready_callback,
	 const std::string& file_name, int id)
{
	id_ = id;//存ID到类，run中使用
	file_name_ = file_name;//用于run中读取二进制图像文件

    //把回调函数保存为成员变量，之后 run() 处理到一帧时会调用这个回调把 frame 交出去
	frame_ready_callback_ = frame_ready_callback;
	
    //yd_.ReadModel("5.23_cracks_bg.onnx", 1, 0, true);
    std::vector<std::string> crack_names = {
     "crack"//,"shield"裂纹、渗水、掉块
    };
    //这里只进行加载裂纹分割的模型
    yvonnx_.set_className(crack_names);
    auto is_ysonnx_ = yvonnx_.ReadModel("1125.onnx", true, 0, true);
	return true;
}



bool file_data::start()
{
	
	//condvar_.notify_one();
	if (!running_) {
		running_ = true;
		thread_ = std::thread(&file_data::run, this);
	}
	
    return true;
}



bool file_data::stop()
{
	if (running_) {
		running_ = false;
		condvar_.notify_one();
		thread_.join();
	}
	return true;

}

/*
void cv::dnn::NMSBoxes(
    const std::vector<cv::Rect>& bboxes,   // 输入的边界框
    const std::vector<float>& scores,      // 对应的置信度
    const float score_threshold,           // 置信度阈值
    const float nms_threshold,             // NMS阈值
    std::vector<int>& indices,             // 输出的保留下来的边界框的索引
    const float eta = 1.0,                 // Soft-NMS的参数
    const int top_k = 0                    // 保留的最大边界框数量，0表示保留全部
)

*/
void file_data::run() {
    //计算高精度时间，计算处理耗时
    using clock = std::chrono::high_resolution_clock;
    clock::time_point now, last;
    clock::duration duration_alg, duration_capture, duration_capture1, duration_capture2;
    auto tNow = std::chrono::system_clock::now();
    auto tmNow = std::chrono::system_clock::to_time_t(tNow);
    auto locNow = std::localtime(&tmNow);
    std::ostringstream oss;
    oss << std::put_time(locNow, "%Y.%m.%d");
    std::string monthday = "_" + oss.str() + "_";

    //获取当前摄像头的物理分辨率
    // 逻辑：ID 0(1号)和5(6号)是4k；ID 1-4(2-5号)是16k
    auto get_real_width = [](int id) -> int {
        if (id == 0 || id == 5) return 4096;  // 4k相机
        return 16384;                         // 16k相机
    };
    std::vector<OutputSeg> output;//定义检测结果

	while (running_) {
        last = clock::now();
		std::ifstream is(file_name_, std::ios::in | std::ios::binary);

		if (is) {
			is.seekg(0, is.end);
			unsigned long long fsize = is.tellg();// 文件总字节数

            //获取真实宽度，计算单帧字节数
            int real_w = get_real_width(id_);
            unsigned int real_len = real_w * IMGHEIGHT;

			
            
            //SeekG 跳过头部时，必须乘真实宽度
            // changeh_ 是行数偏移，乘以 real_w 才是字节偏移
            is.seekg(changeh_[id_] * real_w, is.beg);//changeh_ = {0} 没有跳过任何数据

			
            unsigned int numbers = fsize / real_len;//取整


            //分配最大内存
			char * buffer = new char[16384 * IMGHEIGHT];//分配内存缓冲区 buffer 用来读取一帧。

			for (int j = 0; j < numbers; j++) {
                last = clock::now();

				is.read(buffer, real_len);// 从文件中读取一帧

                duration_capture2 = clock::now() - last;
                auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture2).count();

                last = clock::now();


                
                // 先把 buffer 包装成真实的 Mat (4k 或 16k)
                cv::Mat raw_img(IMGHEIGHT, real_w, CV_8UC1, buffer);
                cv::Mat tempimg;

                // 统一缩放到 8192 (IMGWIDTH)
                if (real_w != IMGWIDTH) {
                    // 1号/6号(4k)会被拉伸，2-5号(16k)会被压缩
                    cv::resize(raw_img, tempimg, cv::Size(IMGWIDTH, IMGHEIGHT));
                }
                else {
                    tempimg = raw_img.clone();
                }

				frame cframe(tempimg, id_, j, j,0);//创建 frame 对象（自定义结构，用于回调传递）。
				//如果 is_save_ 为真，直接保存帧为 .jpg，然后回调一个空帧。
                //保存模式
				if (is_save_) {
					cv::imwrite(path_ + std::to_string(id_+1) + monthday +std::to_string(j)+".jpg", tempimg);
					cv::Mat empty_image;
					frame eframe(empty_image, id_, j, j,0);
					//frame_ready_callback_(eframe);//帧回调函数对帧进行处理
				}
                //否则 检测模式
				{
                    cv::imwrite(path_ + std::to_string(id_ + 1) + monthday + std::to_string(j) + ".jpg", tempimg);//保存原始图片，准备进入分块检测。
                    std::vector<cv::Scalar> color;
                    for (int m = 0; m < 5; m++) {
                        color.push_back(cv::Scalar(0, 0, 255));
                    }

                    int rows = IMGHEIGHT;
                    int cols = IMGWIDTH;
             

                    // 网格的大小（8x8）  
                    int gridSize = 7;
                    int cellRows = rows / gridSize;
                    int cellCols = cols / gridSize;

                    //分割7*7，128*128
                    //将大图分成 7x7 网格，每块大小固定（1152×1152）。
                    cellRows = 1280;
                    cellCols = 1280;
                    int cell_width = 1152;
                    int cell_height = 1152;

                    std::vector<float> confidences;
                    std::vector<cv::Rect> boxes;
                    std::vector<int> nms_result;
                    std::vector<cv::Mat> masks;


                    std::vector<crack_result> crack_results;
                    // 遍历网格并保存每一份  
                    for (int m = 0; m < gridSize; ++m) {
                        for (int n = 0; n < gridSize; ++n) {
                            // 计算当前网格的起始和结束坐标  
                            cv::Rect roi(n * cell_width, m * cell_height, cellCols, cellRows);
                            // 从原图中提取ROI（感兴趣区域）  
                            cv::Mat cell = tempimg(roi).clone();
                            cv::cvtColor(cell, cell, cv::COLOR_GRAY2BGR);//将每个 ROI 转成 BGR 格式（可能模型需要彩色输入）。

                            bool find = yvonnx_.OnnxDetect(cell, output);//调用 yvonnx_.OnnxDetect() 进行 ONNX 推理，output 存储检测结果。

                            //处理检测结果
                            if (find&&output.size() > 0) {
                                // 按照置信度排序会融合失败
                                //按 x 坐标排序，避免部分结果错位。
                                std::sort(output.begin(), output.end(), [](const OutputSeg& a, const OutputSeg& b) {
                                    return a.box.x < b.box.x;
                                });
                                for (int k = 0; k < output.size(); k++) {
                                    //将局部检测框映射回原图的全局坐标。
                                    //存储置信度、位置框、分割 mask。
                                    confidences.push_back(output[k].confidence);
                                    cv::Rect unitary_rect;
                                    unitary_rect = output[k].box;
                                    unitary_rect.x = output[k].box.x + n * cell_width;
                                    unitary_rect.y = output[k].box.y + m * cell_height;
                                    boxes.push_back(unitary_rect);
                                    masks.push_back(output[k].boxMask);
                                    //调用 skeletonize 生成骨架图，calculate_crack_dimensions 计算裂缝长度和宽度。
                                    auto skimg = img::skeletonize(output[k].boxMask);
                                    float length, width;
                                    img::calculate_crack_dimensions(output[k].boxMask,skimg,length,width);

                                }
                          
                         
                                auto outputmerge = mergeRectangles(output);
                                auto cell2 = cell.clone();
                                //yoloV8检测的class_name
                                DrawPred_seg(cell, outputmerge, _className, color);
                                //DrawPred(cell2, output, yd_._className, color);
                                cv::resize(cell, cell, cv::Size(1024, 1024));
                            }
                        }
                    }
                    //7. NMS 去重
                    //使用 OpenCV 的 NMSBoxes（非极大值抑制）去掉重叠检测框。
                    //把保留的框整理到 output2。
                    cv::dnn::NMSBoxes(boxes, confidences, 0.2, 0.5, nms_result);
                    std::vector<OutputSeg>  output2;
      
                    for (size_t i = 0; i < nms_result.size(); ++i) {
                        int idx = nms_result[i];
                        OutputSeg result;
                        result.id = 0;
                        result.confidence = confidences[idx];
                        result.box = boxes[idx];
                        result.boxMask = masks[idx];
                        output2.push_back(result);
     /*                 crack_result cresult;
                        cresult.box = boxes[idx];
                        cresult.confidence = confidences[idx];
                        cresult.length = cresult.box.width;
                        cresult.width = cresult.box.height;
                        crack_results.push_back(cresult);*/
                    }
                    if (nms_result.size() > 0) {
                        //cv::imwrite("I:/result/" + std::to_string(id_ + 1) + monthday + std::to_string(j) + "_" + ".jpg", tempimg);
                        //cv::Mat result_img = tempimg.clone();
                        //cv::cvtColor(result_img, result_img, cv::COLOR_GRAY2BGR);
                        //DrawPred_seg(result_img, output2, yd_._className, color);
                        //cv::imwrite("d:/3.jpg", result_img);
                       
                        //cv::resize(result_img, result_img, cv::Size(1024, 1024));
                       // cv::imwrite("I:/result/" + std::to_string(id_ + 1) + monthday + std::to_string(j) + "_result" + ".jpg", result_img);
                    }
                    //8. 绘制检测结果
                    output2 = mergeRectangles(output2);
                    if (output2.size() > 0) {
                        cv::Mat result_img = tempimg.clone();
                        cv::cvtColor(result_img, result_img, cv::COLOR_GRAY2BGR);
                        //合并检测框，绘制分割结果（DrawPred_seg）。这里用的是yolov8的class_name
                        DrawPred_seg(result_img, output2, _className, color);

                        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 75 };
                        //缩放结果图到 1024×1024
                        cv::resize(result_img, result_img, cv::Size(1024, 1024));
                        //cv::imshow("1", result_img);
                        //cv::waitKey();
                        //cv::imwrite(result_files_name_ + std::to_string(id_ + 1) + monthday + std::to_string(j) + "_result" + ".jpg", result_img, params);
                        //cv::imwrite(result_files_name_ + std::to_string(id_ + 1) + monthday + std::to_string(j)  + ".jpg", tempimg);

                    }
                    //把检测结果存入 cframe 并回调。
                    cframe.results = output2;
                    //帧回调函数处理
					frame_ready_callback_(cframe);
				}
				//9. 循环控制
				_sleep(5000);//每帧处理完暂停 5 秒（可能是防止处理过快或与外部同步）
                duration_capture1 = clock::now() - last;
                 timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
                //qDebug() << "i="<<id_<<"==j==" << j<<"=timeuse="<< timeuse;
                //qDebug() << "=timeuse=" << timeuse;

			}
            //10. 退出时释放资源
			is.close();
			delete[] buffer;
            return;
		}

	}
	return;
}



