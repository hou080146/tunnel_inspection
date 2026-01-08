#include "alg_thread.h"
#include<QDateTime>
#include<QDebug>
#include"img.h"
#include "AppConfig.h"

#pragma execution_character_set("utf-8")

//调用函数读取的数据没有使用
bool alg_thread::readRefPointFromFile(const std::string& filename, RefPoint& refPoint) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&refPoint), sizeof(RefPoint));
    if (!file) {
        std::cerr << "读取文件失败: " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}
int alg_thread::nearestMultipleOf100(int number) {
    // 计算离给定整数最近的可以整除 100 的数
    int remainder = number % 100;
    if (remainder >= 50) {
        return number + (100 - remainder);
    }
    else {
        return number - remainder;
    }
}
bool alg_thread::init(alg_finished_callback alg_finished_callback)
{

	alg_finished_callback_ = alg_finished_callback;
    is_ysonnx_ = yvonnx_.ReadModel("seepage1129_2.onnx", true, 0, true);//seepage_blockall
    //is_ysonnx_ = yolov_detect_.ReadModel("all6490.onnx", true, 0, true);
	return true;
}

bool alg_thread::start()
{

	//正在运行，直接返回
	condition_.notify_one();
	if (running_)
		return true;

	running_ = true;
    for (auto frames : frames_) {
        frames.clear();
    }
	
	frame_number_ = 0;



	//创建线程
	thread_.reset(new std::thread(&alg_thread::run, this));

	return true;
}

void alg_thread::run()
{
    // 角度和位置修正数组，调整图像拼接时的位置偏移
    //int changeh[6] = { 0,0,0,0,0,0 };
    // 图像水平方向修正偏移，单位为像素
    int changew[6] = { 0,0,0,0,0,0 };
    // 缺陷名称对应索引

    // 【修正 1】X轴精度 (经过Resize后的有效精度, mm/pixel)
    // 1,6号(4k->8k): 0.28/2 = 0.14
    // 2-5号(16k->8k): 0.21*2 = 0.42 (假设2,5号是50mm镜头), 0.245*2 = 0.49 (3,4号是40mm镜头)
    float x_pixel_accuracy[6] = { 0.14f, 0.42f, 0.49f, 0.49f, 0.42f, 0.14f };

    // 【新增 2】Y轴精度 (原生扫描行精度, mm/pixel) - 这是一个关键的新增数组
    // Y轴高度始终是8192，没有缩放，所以必须用原生精度
    //float y_pixel_accuracy[6] = { 0.28f, 0.21f, 0.245f, 0.245f, 0.21f, 0.28f };
    float global_y_acc = 0.21f;
    if (AppConfig::Y_ACC > 0)
    {
        global_y_acc = AppConfig::Y_ACC;
    }
    float y_pixel_accuracy[6];
    for (int k = 0; k < 6; k++) {
        y_pixel_accuracy[k] = global_y_acc;
    }
    std::vector<std::string> defect = {
        "渗水","渗水","掉块","里程"//,"shield"裂纹、渗水、掉块
    };
	//std::vector<cv::Mat> destination_frames;
    // 存放采集的原始图像帧，后续处理用
	std::vector<cv::Mat> source_frames;
    // 每个摄像头拍摄角度起始点（度数）
    float angles_start[6] = { 55.65, 67.965, 118.965, 169.765, 220.765, 272.35 };//每个相机起始拍摄角度
    float unit_angle[6] = { 0.007812, 0.004349, 0.004349, 0.004349, 0.004349, 0.00781 };// 每个像素对应的角度值32.0/4096.0;71.27.0/16384.0

    // 里程相关起始参数
    float mileage_start = AppConfig::Mileage;//里程起始点
    //float unit_mileage = 8192.0*0.5/1000.0;//每张图片里程长度//南昌地铁0.25\0.314mm每触发一次
    int mileage_direction = AppConfig::MileageDown;// 里程增加方向，-1表示递减
    
    
    // 读取参考点文件（里程信息、线名等）读取的没用
    RefPoint refPoint;
    std::string filename = files_name1_+"/Milepost0101.mp"; // 参考点文件路径
    
    if (readRefPointFromFile(filename, refPoint)) {
        auto linename = std::string(refPoint.Linename);
        std::wcout << L"读取成功!" << std::endl;
    }
    else {
        std::cerr << "读取失败!" << std::endl;
    }
    std::string linename = "4_up_-_ck";// 默认线路名，备用





	using clock = std::chrono::high_resolution_clock;
	clock::time_point now, last;
	clock::duration duration_alg, duration_capture, duration_capture1, duration_capture2;
    auto tNow= std::chrono::system_clock::now();
    auto tmNow = std::chrono::system_clock::to_time_t(tNow);
    auto locNow = std::localtime(&tmNow);
    std::ostringstream oss;
    oss << std::put_time(locNow, "%Y.%m.%d");
    std::string monthday = "_"+oss.str()+"_";
    // 结果文件路径（CSV和Word报告）
    std::string csvname = result_files_name_ + linename+monthday + "result.csv";
    std::string wordname = result_files_name_ + linename + monthday + "result.doc";
    // 打开CSV文件，用于写入检测结果
    std::ofstream of(csvname);
    // 创建CLAHE（自适应直方图均衡）对象，用于图像增强
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);  // 设置对比度限制
    clahe->setTilesGridSize(cv::Size(8, 8));  // 设置网格大小
    
    // 写入CSV表头
    of << "缺陷名" << "," << "角度" << "," << "帧号" << "," << "面积" << "," << "长度" << "," << "宽度" << "," << "里程（km)" << "," << "置信度" << "," << "相机号" << "," << "执行时间" << "," << "图像路径" << "," << std::endl;
    
    // 主循环，线程运行期间持续处理数据
	while (running_)
	{
		
		std::vector<file_data::frame> frames;
        // 记录当前时间点，测量处理时间
		last = clock::now();
		{
            // 上锁，保护共享资源 process_frames_
			std::unique_lock<std::mutex> lock(mutex_);
			duration_capture1 = clock::now() - last;

            // 等待条件变量唤醒（有数据可处理），此时在push_frame中已经将本批次需要处理的数据帧准备好
			condition_.wait(lock);
			//condition_.wait_for(lock, std::chrono::milliseconds(100));

			//while (running_ && (condition_.wait_for(lock, std::chrono::milliseconds(50)) != std::cv_status::timeout || !action_));
			duration_capture2 = clock::now() - last;

			if (!running_) // 线程停止，退出循环
				break;
            // 将待处理的帧交换到本地变量，避免阻塞采集线程
			std::swap(frames, process_frames_);
		}
        auto duration_capture1 = clock::now() - last;
        auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
       // qDebug() <<"wait data=="<< timeuse;
        //last = clock::now();
        frame_number_++;// 处理帧计数器增加
        // 没有帧或帧数据为空则跳过
		if (frames.size() < 1)continue;
        if (frames[0].data.empty())continue;
        // 按摄像头id排序，确保顺序处理
		std::sort(frames.begin(), frames.end(), [](const file_data::frame &a,const file_data::frame &b) {

			return a.camera_id < b.camera_id;
		
		});
        //生成多路摄像头图像拼接的画布，宽度为帧数乘单帧宽度，高度为单帧高度，下面for中遍历每个摄像头时会拼接到这个画布上
		cv::Mat mergedImage( IMGHEIGHT, frames.size() * IMGWIDTH , CV_8UC3);
       // cv::Mat mergedImage(1024, 1024, CV_8UC3);
        float area = 0;// 缺陷面积初始化



        // 遍历同一时刻6个摄像头帧进行检测和处理
		for (int i = 0; i < frames.size(); i++) {
            result ret,crack_ret;// 存储检测结果的结构体

            // 【核心修正】分别获取 X轴 和 Y轴 的精度
            float acc_x = x_pixel_accuracy[i]; // 宽被拉伸/压缩后的精度
            float acc_y = y_pixel_accuracy[i]; // 高未变，原生精度
            // 【新增】计算单帧代表的物理长度 (米) = 8192行 * Y轴精度 / 1000
            float frame_len_m = 8192.0f * acc_y / 1000.0f;

            cv::Mat timg = frames[i].data.clone();// 拷贝当前帧图像
           // cv::resize(timg, timg, cv::Size(NETWIDTH, NETHEIGHT), cv::INTER_LINEAR);
            if(timg.channels()<3)
                cv::cvtColor(timg, timg, cv::COLOR_GRAY2BGR);// 保证3通道
            std::vector<OutputSeg> output, output2;// 模型输出结果
            //to do 算法处理 

            // 生成颜色列表，用于绘制结果（此处固定红色），绘制裂纹的颜色
            std::vector<cv::Scalar> color;
            for (int m = 0; m < 80; m++) {
                color.push_back(cv::Scalar(0, 0, 255));// 红色
            }

            ret.length = 0.0;
            float max_confidence = 0;
            crack_ret.length = 0.0;
            bool find = false;

            // 渗水掉块检测模型
            find = yvonnx_.OnnxDetect(timg, output);
            bool meters_mark = false;

            // 如果是第0号摄像头，额外处理里程标记检测
            // 如果是第0号摄像头（即1号相机，4k），额外处理里程标记检测
            if (frames[i].camera_id == 0) {
                
                for (auto mileage : output) {
                    if (mileage.id == 3 && mileage.confidence > 0.9 && mileage.box.width > 10 && mileage.box.height > 10 && frames[i].frame_number != 249) {//frames[i].frame_number != 249为什么？

                        // 【修正】使用 frame_len_m 和 acc_y 计算
                        float y_offset_m = float(output[0].box.y) * acc_y / 1000.0f;
                        auto mileage_mark = mileage_start + (frames[i].frame_number * frame_len_m + y_offset_m) * mileage_direction / 1000.0;

                        auto real_mileage_mark = float(nearestMultipleOf100(mileage_mark * 1000));
                        mileage_start = mileage_start + (real_mileage_mark / 1000.0) - mileage_mark;
                        meters_mark = true;
                        qDebug() << "mileage=" << frames[i].frame_number << ":real_mileage_mark=" << real_mileage_mark;

                        cv::imwrite(result_files_name_ + std::to_string(frames[i].camera_id) + "_" + std::to_string(frames[i].frame_number) + "_" + std::to_string(int(real_mileage_mark)) + "_mileage" + ".jpg", timg(mileage.box));
                        break;
                    }
                }
            }

            bool iswrite1 = false;// 是否写渗水、掉块等结果
            bool iswrite_carck = false;// 是否写裂纹结果

            // 如果检测到缺陷且置信度合格
           if (find&&output.size()>0&&output[0].confidence>0.5) {
               for (int k = 0; k < output.size(); k++) {
                   if (output[k].confidence <= 0.5|| output[k].id==3)continue;
                   // 填充缺陷信息
                   ret.injury_image_name = defect[output[k].id];//ret是结果结构体result
                   iswrite1 = true;

                   // 【修改】计算缺陷面积
                    // 原代码: area = area2 * 0.25*0.25/1000000;
                    // 新逻辑: 像素数 * (精度x * 精度y) / 100万 (转为平方米)
                    // 假设纵向(行)精度与横向精度近似相同
                   float area2 = cv::countNonZero(output[k].boxMask);
                   
                   area = area2 * acc_x * acc_y / 1000000.0f;
                   ret.area = area;
                   // 计算角度，结合摄像头起始角度和像素位置
                   ret.angle = angles_start[frames[i].camera_id] + unit_angle[frames[i].camera_id] * float(output[0].box.x);
                   
                   // 【修改】计算里程
                   // 原代码: frames[i].frame_number*unit_mileage + float(output[0].box.y)*0.25/1000
                   // 新逻辑: 
                   // 1. 基础里程: 帧号 * 图像高度(8192) * 精度 / 1000 (转为米)
                   // 2. 框偏移: box.y * 精度 / 1000
                   //float frame_len_m = 8192.0f * acc_y / 1000.0f; // 单帧代表的物理长度(米)
                   float y_offset_m = float(output[0].box.y) * acc_y / 1000.0f;
                   float total_dist_m = frames[i].frame_number * frame_len_m + y_offset_m;

                   // 修正后的公式
                   ret.mileage = mileage_start + (total_dist_m * mileage_direction) / 1000.0f;
                   //qDebug() << frames[i].frame_number << "==" << ret.mileage;
                   ret.order = frames[i].camera_id;
                   ret.confidence = output[k].confidence;
                   ret.bar_value = frames[i].frame_number;
                   // 生成图片保存路径
                   ret.img_path = result_files_name_ + std::to_string(frames[i].camera_id) + monthday + std::to_string(frames[i].frame_number) + "_result" + ".jpg";
                   // 写入CSV
                   of << ret.injury_image_name << "," << ret.angle << "," << ret.bar_value << "," << ret.area << "," << ret.length << "," << ret.width << "," << ret.mileage << "," << ret.confidence << "," << ret.order << "," << ret.proc_time << "," << "\"=HYPERLINK(\"\"" << ret.img_path << "\"\", \"\"" << ret.img_path << "\"\")\"" << "," << std::endl;


               }
           }
       

           // 处理裂纹检测结果，裂纹检测一般会有多个结果
           //这里裂纹检测已经结束，结果保存在alg_thread中
           for (int m = 0; m < frames[i].results.size(); m++) {
               /*if (frames[i].results[m].confidence < 0.8)
                  continue;*/
               if (max_confidence < frames[i].results[m].confidence)
                   max_confidence = frames[i].results[m].confidence;
               crack_ret.injury_image_name = "裂纹";
               // 骨架化处理（提取裂纹骨架）
               auto sktimg = img::skeletonize(frames[i].results[m].boxMask);

               /*   cv::imshow("1", sktimg);
                  cv::waitKey();*/
               float width = 0.0;
               float length = 0.0;
               // 计算裂纹长度和宽度
               img::calculate_crack_dimensions(frames[i].results[m].boxMask, sktimg, length, width);
               
               // 【修改】计算裂纹物理长度和宽度
               // // 【修正】裂纹长度使用混合精度估算 (这里略微复杂，简单起见用X精度，更严谨应该根据裂纹走向积分)
               // 但由于裂纹大多是横向的，且为了代码简洁，这里可以使用 acc_x，或者使用 sqrt(acc_x*acc_y)
               // 修正里程计算必须用 Y精
               // 原代码: crack_ret.length = length * 0.21/1000.0;
               // 原代码: crack_ret.width = width * 0.021;
               crack_ret.length = length * acc_x / 1000.0f; // 像素长度 * 精度 / 1000 => 米
               if (crack_ret.length > 200)// 长度阈值（单位不太确定，可能mm）
                   iswrite_carck = true;
               crack_ret.width = width * acc_x;             // 像素宽度 * 精度 => 毫米
               crack_ret.angle = angles_start[frames[i].camera_id] + unit_angle[frames[i].camera_id] * float(frames[i].results[0].box.x);//起始角度+每个像素所占的角度*裂纹在图像上的x坐标
               // 【修正】里程计算使用 Y轴精度
               //float frame_len_m = 8192.0f * acc_y / 1000.0f;
               float defect_y_m = float(frames[i].results[0].box.y) * acc_y / 1000.0f;
               float total_dist_m = frames[i].frame_number * frame_len_m + defect_y_m;
               crack_ret.mileage = mileage_start + (total_dist_m * mileage_direction) / 1000.0f;
               //qDebug() << frames[i].frame_number << "==" << ret.mileage;
               crack_ret.order = frames[i].camera_id;
               crack_ret.confidence = frames[i].results[0].confidence;
               crack_ret.bar_value = frames[i].frame_number;
               crack_ret.img_path = result_files_name_ + std::to_string(frames[i].camera_id) + monthday + std::to_string(frames[i].frame_number) + "_result" + ".jpg";
           

           }
           // 写裂纹检测结果到CSV，且长度满足条件
           if (frames[i].results.size() > 0&& crack_ret.length>0.2) {
               of << crack_ret.injury_image_name << "," << crack_ret.angle << "," << crack_ret.bar_value << "," << crack_ret.area << "," << crack_ret.length << "," << crack_ret.width << "," << crack_ret.mileage << "," << crack_ret.confidence << "," << crack_ret.order << "," << crack_ret.proc_time << "," << "\"=HYPERLINK(\"\"" << crack_ret.img_path << "\"\", \"\"" << crack_ret.img_path << "\"\")\"" << "," << std::endl;
               // 在图像上绘制检测结果 
               DrawPred_seg(timg, frames[i].results, yolov_detect_._className, color);
               // 保存绘制结果图，质量30（较高压缩）
               std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 30 };
               cv::imwrite(result_files_name_ + std::to_string(frames[i].camera_id) + monthday + std::to_string(frames[i].frame_number) + "_result" + ".jpg", timg, params);
               // cv::imwrite(result_files_name_ + std::to_string(frames[i].camera_id + 1) + monthday + std::to_string(frames[i].frame_number) + ".jpg", frames[i].data);

           }//&& max_confidence > 0.6)
           // 图像均衡化处理，改善图像对比度，部分摄像头使用原图，其余用CLAHE
           cv::Mat equalizedImage2;
           if (iswrite1 || iswrite_carck) {

               cv::Mat equalizedImage;
               if (frames[i].camera_id == 4 || frames[i].camera_id == 6) {
                   equalizedImage = timg;// 特定摄像头不处理
               }
               else {
                   cv::Mat gray;
                   cvtColor(timg, gray, cv::COLOR_BGR2GRAY);
                   //cv::equalizeHist(gray, equalizedImage);
                   clahe->apply(gray, equalizedImage);
                   cvtColor(equalizedImage, equalizedImage, cv::COLOR_GRAY2BGR);
               }
              
               //equalizedImage2 = equalizedImage.clone();
               // 绘制渗水、掉块、裂纹检测结果
               DrawPred_seg(equalizedImage, output, yvonnx_._className, color);//color红色
               DrawPred_seg(equalizedImage, frames[i].results, yolov_detect_._className, color);
               
               std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 30 };
               // 保存增强后的结果图
               cv::imwrite(result_files_name_ + std::to_string(frames[i].camera_id) + monthday + std::to_string(frames[i].frame_number) + "_result" + ".jpg", equalizedImage, params);
               //cv::imwrite(result_files_name_ + std::to_string(frames[i].camera_id + 1) + monthday + std::to_string(frames[i].frame_number) + ".jpg", frames[i].data);
           }
           // 拼接合成大图，利用changew数组修正水平方向偏移
           //cv::Mat roi(mergedImage, rect) 这样构造出的 roi 是 mergedImage 指定矩形区域的一个“视图”（浅拷贝），对 roi 的修改会直接影响 mergedImage 对应区域。
           
           
           cv::Mat roi(mergedImage, cv::Rect(i * IMGWIDTH - changew[i], 0, IMGWIDTH, IMGHEIGHT));//第i个相机拼接到changew[i]位置上
		  // cv::Mat roi(mergedImage, cv::Rect(i * IMGWIDTH , 0, IMGWIDTH ,  IMGHEIGHT));
           // 最后一幅图特殊均衡化处理，拼接到合成图中
           if (i == 5) {
               cv::Mat gray;
               equalizedImage2 = timg.clone();
               cvtColor(equalizedImage2, gray, cv::COLOR_BGR2GRAY);
               //cv::equalizeHist(gray, equalizedImage);
               clahe->apply(gray, equalizedImage2);
               cvtColor(equalizedImage2, equalizedImage2, cv::COLOR_GRAY2BGR);
               equalizedImage2.copyTo(roi);
           }
           else {
               timg.copyTo(roi);
           }
           
		}
        // 裁剪合成图有效部分并调整大小，方便后续显示
        cv::Mat mergedImage2(mergedImage, cv::Rect(0, 0, 6 * IMGWIDTH - changew[5], IMGHEIGHT));
        
        
        // 将裁剪后的图像调整大小为2048x2048，方便后续显示或处理
		cv::resize(mergedImage2, mergedImage2, cv::Size(2048, 2048));
        // 定义结果结构体变量ret，准备存储当前处理的一些统计数据
		result ret;
        ret.area = area;// 把之前计算的缺陷面积area赋值给ret.area
		ret.bar_value = frames[0].frame_number;// 设置当前帧号（使用第0号摄像头帧的帧号）给ret.bar_value，代表当前处理进度或帧标识
        auto duration_capture2 = clock::now() - last;// 计算从上一个时间点last到当前的时间差，用于统计本次算法处理耗时
         timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture2).count();// 转换耗时为毫秒数
       // qDebug() << "process==" << timeuse;
        ret.proc_time = timeuse;// 将耗时赋值给ret.proc_time，便于后续展示或记录
        // 调用回调函数，传递处理结果和拼接图
		alg_finished_callback_(ret, mergedImage2);// 调用预先绑定的回调函数，将处理结果ret和裁剪缩放后的合成图mergedImage2传递出去
		
	}

    of.close();
}



bool alg_thread::stop()
{
	if (running_)
	{
		mutex_.lock();
		running_ = false;
		condition_.notify_one();
		mutex_.unlock();
		thread_->join();
	}

	return true;
}

void alg_thread::push_frame(const file_data::frame& frame)
{
    using clock = std::chrono::high_resolution_clock;
    clock::time_point now, last;
    clock::duration duration_alg, duration_capture, duration_capture1, duration_capture2;

    last = clock::now();// 记录函数开始时间
	std::unique_lock<std::mutex> lock(mutex_);
    // 将传入的帧frame按照camera_id存入对应的frames_队列中
	frames_[frame.camera_id].push_back(frame);

	unsigned int size = 0;
    // 遍历所有有效摄像头effectives_的ID，成功打开摄像头目录的ID号
	for (int i = 0; i < effectives_.size(); i++) {
        // 如果对应摄像头的frames_队列不为空
		if (frames_[effectives_[i]].size() > 0) {
            // 判断该队列首帧的frame_number是否等于当前全局的frame_number_，
            // 只有所有摄像头的帧号一致时，才统一取出来进行后续处理（保证帧同步）
			if (frames_[effectives_[i]][0].frame_number == frame_number_ ){//&& frames_[effectives_[i]][0].camera_id == process_frames_.size() + 1) {
                // 把该摄像头对应帧放入process_frames_中，准备统一处理
				process_frames_.push_back(frames_[effectives_[i]][0]);
                // 从原队列中删除已转移的帧
				frames_[effectives_[i]].erase(frames_[effectives_[i]].begin());
			}
		}
	}
	lock.unlock();// 解锁，让其他线程可以访问frames_等共享资源
    // 如果当前准备处理的帧数和有效摄像头数量相同，说明所有摄像头该帧号的数据都准备好了
	if (process_frames_.size() == effectives_.size()) {

		condition_.notify_one();// 通知等待该条件的线程（例如算法处理线程）可以开始工作了
	}
     duration_capture1 = clock::now() - last;// 计算push_frame函数执行的耗时（毫秒），便于性能监控
    auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
}

unsigned int alg_thread::set_data_name(std::vector<QString> &files_name, std::string  store_files_name, std::string result_files_name) {
    //std::vector<QString> files_name = files_name;
    store_files_name_ = store_files_name;
    result_files_name_ = result_files_name;
	unsigned int numbers = 0;// 用于存储计算的帧数（假设所有摄像头帧数相同）
	std::vector< std::ifstream> ifstreams(CAMERANUMBER); // 创建一个大小为CAMERANUMBER的ifstream向量，用于打开所有摄像头对应的文件
    // 遍历所有摄像头索引
    // 【修改】循环次数改为 CAMERANUMBER (6)
	for (int i = 0; i < CAMERANUMBER; i++) {
        QString fname;
        fname = files_name[i] + "/Recv.img";
        ifstreams[i].open(fname.toStdString(), std::ios::in | std::ios::binary);

        if (ifstreams[i].is_open()) {
            ifstreams[i].seekg(0, ifstreams[i].end);
            unsigned long long fsize = ifstreams[i].tellg();
            
            // 【修改】计算真实帧大小
            int real_w = 16384;
            if (i == 0 || i == 5) real_w = 4096; // 1号6号是4k

            unsigned int length = real_w * IMGHEIGHT;
            
            ifstreams[i].seekg(0, ifstreams[i].beg);
            numbers = fsize / length;
            qDebug() << "第" << i << "个相机，有" << numbers << "张图像";
            effectives_.push_back(i);
        }

	}
    // 关闭所有打开的文件流
	for (int j = 0; j < ifstreams.size(); j++) {

		ifstreams[j].close();
	}
    // 返回最后一个成功打开文件的帧数（注意：此处numbers被不断覆盖，返回的是最后一个打开文件的帧数）
	return numbers;
}

alg_thread::~alg_thread()
{
    
	stop();
}
