#include "ys_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

/*
加载ONNX模型并创建ORT Session。
根据是否使用GPU动态配置执行提供者。
处理输入输出节点信息，动态调整输入尺寸。
判断模型是否为分割模型（需有2个输出）。
提供GPU warm-up机制。
*/
bool ysOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
    if (_batchSize < 1) _batchSize = 1;
    try
    {
        std::vector<std::string> available_providers = GetAvailableProviders();// 查询可用的执行提供者（CPU, CUDA等）
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");

        if (isCuda && (cuda_available == available_providers.end()))
        {
            std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        else if (isCuda && (cuda_available != available_providers.end()))
        {
            std::cout << "************* Infer model on GPU! *************" << std::endl;
            #if ORT_API_VERSION < ORT_OLD_VISON
            			OrtCUDAProviderOptions cudaOption;
            			cudaOption.device_id = cudaID;
            			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
            #else

               auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
                     
                    
            #endif
        }
        else
        {
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        //// 设置图优化等级
        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        //创建 Session（加载模型）
#ifdef _WIN32
        //Windows 下 Ort::Session 需要宽字符路径；其他平台用 char* 即可
        std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);//析构函数没释放，注意修改
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif
        //读取输入节点信息
        Ort::AllocatorWithDefaultOptions allocator;
        //// 初始化输入信息
        _inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
        _inputName = _OrtSession->GetInputName(0, allocator);
        _inputNodeNames.push_back(_inputName);
#else
        _inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
        _inputNodeNames.push_back(_inputName.get());
#endif

        Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputNodeDataType = input_tensor_info.GetElementType();
        _inputTensorShape = input_tensor_info.GetShape();//[B,C,H,W]
        //处理动态尺寸，-1表示模型中该变量是动态的，并不是固定死的
        if (_inputTensorShape[0] == -1)
        {
            _isDynamicShape = true;
            _inputTensorShape[0] = _batchSize;

        }
        if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
            _isDynamicShape = true;
            _inputTensorShape[2] = _netHeight;
            _inputTensorShape[3] = _netWidth;
        }
        //init output
        _outputNodesNum = _OrtSession->GetOutputCount();
        if (_outputNodesNum != 2) {
            cout << "This model has " << _outputNodesNum << "output, which is not a segmentation model.Please check your model name or path!" << endl;
            return false;
        }
#if ORT_API_VERSION < ORT_OLD_VISON
        _output_name0 = _OrtSession->GetOutputName(0, allocator);
        _output_name1 = _OrtSession->GetOutputName(1, allocator);
#else
        _output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
        _output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));
#endif
        Ort::TypeInfo type_info_output0(nullptr);
        Ort::TypeInfo type_info_output1(nullptr);
        // 确保output0和output1顺序正确，按名字字典序排序,不同版本的ort可能输出的顺序不同，
        bool flag = false;
#if ORT_API_VERSION < ORT_OLD_VISON
        flag = strcmp(_output_name0, _output_name1) < 0;
#else
        flag = strcmp(_output_name0.get(), _output_name1.get()) < 0;
#endif
        if (flag)  //make sure "output0" is in front of  "output1"
        {

            type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
            type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name0);
            _outputNodeNames.push_back(_output_name1);
#else
            _outputNodeNames.push_back(_output_name0.get());
            _outputNodeNames.push_back(_output_name1.get());
#endif

        }
        else {
            type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
            type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1
#if ORT_API_VERSION < ORT_OLD_VISON
            _outputNodeNames.push_back(_output_name1);
            _outputNodeNames.push_back(_output_name0);
#else
            _outputNodeNames.push_back(_output_name1.get());
            _outputNodeNames.push_back(_output_name0.get());
#endif
        }

        auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
        _outputNodeDataType = tensor_info_output0.GetElementType();
        _outputTensorShape = tensor_info_output0.GetShape();
        auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();
        //_outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
        //_outputMaskTensorShape = tensor_info_output1.GetShape();
        //if (_outputTensorShape[0] == -1)
        //{
        //	_outputTensorShape[0] = _batchSize;
        //	_outputMaskTensorShape[0] = _batchSize;
        //}
        //if (_outputMaskTensorShape[2] == -1) {
        //	//size_t ouput_rows = 0;
        //	//for (int i = 0; i < _strideSize; ++i) {
        //	//	ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
        //	//}
        //	//_outputTensorShape[1] = ouput_rows;

        //	_outputMaskTensorShape[2] = _segHeight;
        //	_outputMaskTensorShape[3] = _segWidth;
        //}
        //warm up
        /**/
        // GPU模式下的warm-up，执行几次推理以稳定性能
        if (isCuda && warmUp) {
            //draw run
            cout << "Start warming up" << endl;
            size_t input_tensor_length = VectorProduct(_inputTensorShape);
            float* temp = new float[input_tensor_length];
            std::vector<Ort::Value> input_tensors;
            std::vector<Ort::Value> output_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                _inputTensorShape.size()));
    
            for (int i = 0; i < 3; ++i) {
                output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                    _inputNodeNames.data(),
                    input_tensors.data(),
                    1,
                    _outputNodeNames.data(),
                    _outputNodeNames.size());
            }
            
            delete[]temp;
        }
    }
    catch (const std::exception&) {
        return false;
    }
    return true;
}
/*
对输入图片进行LetterBox缩放（保持比例，填充到指定大小）。
返回缩放比例参数供后续恢复坐标。
保证输入batch满足模型要求大小（填充空白图）。
*/
int ysOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
    outSrcImgs.clear();
    Size input_size = Size(_netWidth, _netHeight);
    for (int i = 0; i < srcImgs.size(); ++i) {
        Mat temp_img = srcImgs[i];
        Vec4d temp_param = { 1,1,0,0 };// 缩放参数，初始化为无缩放无偏移
        if (temp_img.size() != input_size) {
            Mat borderImg;
            // 按比例缩放并填充到指定尺寸，同时保证尺寸为32的倍数
            LetterBox_seg(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            //cout << borderImg.size() << endl;
            outSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        }
        else {
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }

    // 如果图片数不是batchSize的倍数，则补0填充
    int lack_num = srcImgs.size() % _batchSize;
    if (lack_num != 0) {
        for (int i = 0; i < lack_num; ++i) {
            Mat temp_img = Mat::zeros(input_size, CV_8UC3);
            Vec4d temp_param = { 1,1,0,0 };
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }
    return 0;

}
//实际使用的detect函数
bool ysOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output) {
    std::vector<cv::Mat> input_data = { srcImg };
    std::vector<std::vector<OutputSeg>> tenp_output;
    if (OnnxBatchDetect(input_data, tenp_output)) {
        output = tenp_output[0];
        return true;
    }
    else return false;
}

bool ysOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
    // 预处理：把每张原图 resize + letterbox 到网络输入大小，记录还原参数params
    Preprocessing(srcImgs, input_images, params);
    // OpenCV 把一批图像打成 NCHW 的 blob，并做归一化到 [0,1]，输入比例为input_size，scalar不减均值， true交换R、B通道，false不拉伸
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    // 把 blob 内存直接“包”成 ORT 的输入 Tensor（不拷贝）
    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;//一般情况下是只有一个元素（图像），但是由于ort支持多输入，所以这里是vector
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo,//需要提供的内存信息
        (float*)blob.data,// 连续的内存指针
        input_tensor_length,// 数据元素总数
        _inputTensorShape.data(),// 张量形状: [N,C,H,W]
        _inputTensorShape.size()));// 5. 形状维度数

    // 运行推理
    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),// 输入节点名数组
        input_tensors.data(),// 直接输入vector.data()返回首元素地址，即第一个ort::value的地址
        _inputNodeNames.size(),// 输入个数（通常=1即一张图片）
        _outputNodeNames.data(),// 输出节点名数组（2个：检测和mask）
        _outputNodeNames.size()// 输出个数（=2）
    );

    //post-process

    //int net_width = _className.size() + 4 + _segChannels;//modify by xcj
    // 解析输出
    float* all_data = output_tensors[0].GetTensorMutableData<float>();//获取[0]的可变数据指针

    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();//查询类型和形状,输出[batch_size, 预测向量长度[框位置，置信度，表示类别概率的向量], 总预测框数]
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();//输出mask原型，shape：[batch_size, channels, height, width]

    int net_width = _outputTensorShape[1];//modify by xcj// 每条预测向量长度=4（0-3，xyhw)+1（置信度）+类别数（每个类别一个数字表示该类别的概率）注：[0]=batch_size，[2]=总预测框数
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);//计算mask原型张量总元素个数
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];// 单张图的 output0 元素数  [0]是batch_size

    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        // 转置输出张量形状，方便处理
        // 把这张图的 output0 指针封成 Mat，并转置成 [N, net_width]（N=候选数）
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;// 指针跳到下一张图的起始
        float* pdata = (float*)output0.data;
        MaskParams mask_params;
        mask_params.params = params[img_index];//// 记录缩放和偏移参数
        mask_params.srcImgShape = srcImgs[img_index].size();// 原图尺寸
        int rows = output0.rows;// 候选数 N

        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask// 保存每个候选的 mask 系数（长度=segChannels）


        for (int r = 0; r < rows; ++r) {    //stride
            if (pdata == nullptr) {
                cerr << "pdata 指针无效或数据不足" << endl;
                continue;
            }
            // pdata + 4 跳过了前 4 个元素（bbox 的 x, y, w, h），直接指向类别分数数组的起始位置
            cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);// 1行，_className.size()列，内容从pdata+4开始道指针结尾
            Point classIdPoint;
            double max_class_socre;
            if (scores.empty()) {
                cerr << "图像为空，无法处理" << endl;
                continue;
            }
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            // 置信度阈值过滤
            if (max_class_socre >= _classThreshold) {
                //vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);//modify by xcj
                // 取出这一行最后 segChannels 个值作为本实例的 mask 系数
                vector<float> temp_proto(pdata + net_width - mask_params.segChannels, pdata + net_width);//modify by xcj
                picked_proposals.push_back(temp_proto);
                // 预测框是以 letterbox 后坐标系为基准的，要用 params 还原
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                if (w <= 0 || h <= 0)continue;
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;//// 跳到下一行候选
        }

        // 非极大值抑制，过滤重叠框
        vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
        std::vector<vector<float>> temp_mask_proposals;
        cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
        std::vector<OutputSeg> temp_output;
        for (int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];
            OutputSeg result;
            result.id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx] & holeImgRect;

            if (result.box.width < 1 || result.box.height < 1)continue;

            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

    
        Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
        for (int i = 0; i < temp_mask_proposals.size(); ++i) {
            GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
        }

        //******************** ****************
        // \C0?汾\B5?\BD\B0\B8\A3\AC\C8\E7\B9\FB\C9\CF\C3\E6\D4?\AA\C6\F4\CE\D2?\CA??\BF\B7\D6?\BA\F3\BB\B9??\B1\A8\B4\ED\A3\AC\BD\A8\D2\E9?\D3\C3\D5\E2\B8\F6\A1\A3
        // If the GetMask2() still reports errors , it is recommended to use GetMask().
        // Mat mask_proposals;
        //for (int i = 0; i < temp_mask_proposals.size(); ++i)
        //	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
        //GetMask(mask_proposals, mask_protos, output, mask_params);
        //*****************************************************/
        output.push_back(temp_output);

    }

    if (output.size())
        return true;
    else
        return false;
}