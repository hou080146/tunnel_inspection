#include "ys_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

bool ysOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
    if (_batchSize < 1) _batchSize = 1;
    try
    {
        std::vector<std::string> available_providers = GetAvailableProviders();
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

               //auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
                     
                    
            #endif
        }
        else
        {
            std::cout << "************* Infer model on CPU! *************" << std::endl;
        }
        //
        _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
        std::wstring model_path(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        //init input
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
        _inputTensorShape = input_tensor_info.GetShape();

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

int ysOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
    outSrcImgs.clear();
    Size input_size = Size(_netWidth, _netHeight);
    for (int i = 0; i < srcImgs.size(); ++i) {
        Mat temp_img = srcImgs[i];
        Vec4d temp_param = { 1,1,0,0 };
        if (temp_img.size() != input_size) {
            Mat borderImg;
            LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
            //cout << borderImg.size() << endl;
            outSrcImgs.push_back(borderImg);
            params.push_back(temp_param);
        }
        else {
            outSrcImgs.push_back(temp_img);
            params.push_back(temp_param);
        }
    }

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
    //preprocessing
    Preprocessing(srcImgs, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );

    //post-process

    int net_width = _className.size() + 4 + _segChannels;
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
        for (int r = 0; r < rows; ++r) {    //stride
            cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= _classThreshold) {
                vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
                picked_proposals.push_back(temp_proto);
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;//\CF\C2?\D0\D0
        }

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
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
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
bool ysOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
    //preprocessing
    Preprocessing(srcImgs, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );

    //post-process

    int net_width = _className.size() + 4 + _segChannels;
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
        for (int r = 0; r < rows; ++r) {    //stride
            cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= _classThreshold) {
                vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
                picked_proposals.push_back(temp_proto);
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;//\CF\C2?\D0\D0
        }

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
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
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
bool ysOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
    //preprocessing
    Preprocessing(srcImgs, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );

    //post-process

    int net_width = _className.size() + 4 + _segChannels;
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
        for (int r = 0; r < rows; ++r) {    //stride
            cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= _classThreshold) {
                vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
                picked_proposals.push_back(temp_proto);
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;//\CF\C2?\D0\D0
        }

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
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
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
bool ysOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
    vector<Vec4d> params;
    vector<Mat> input_images;
    cv::Size input_size(_netWidth, _netHeight);
    //preprocessing
    Preprocessing(srcImgs, input_images, params);
    cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

    int64_t input_tensor_length = VectorProduct(_inputTensorShape);
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
        _inputNodeNames.data(),
        input_tensors.data(),
        _inputNodeNames.size(),
        _outputNodeNames.data(),
        _outputNodeNames.size()
    );

    //post-process

    int net_width = _className.size() + 4 + _segChannels;
    float* all_data = output_tensors[0].GetTensorMutableData<float>();
    _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
    int mask_protos_length = VectorProduct(mask_protos_shape);
    int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
    for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
        Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
        all_data += one_output_length;
        float* pdata = (float*)output0.data;
        int rows = output0.rows;
        std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
        std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
        std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
        std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
        for (int r = 0; r < rows; ++r) {    //stride
            cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
            Point classIdPoint;
            double max_class_socre;
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
            max_class_socre = (float)max_class_socre;
            if (max_class_socre >= _classThreshold) {
                vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
                picked_proposals.push_back(temp_proto);
                //rect [x,y,w,h]
                float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                float w = pdata[2] / params[img_index][0];  //w
                float h = pdata[3] / params[img_index][1];  //h
                int left = MAX(int(x - 0.5 * w + 0.5), 0);
                int top = MAX(int(y - 0.5 * h + 0.5), 0);
                class_ids.push_back(classIdPoint.x);
                confidences.push_back(max_class_socre);
                boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
            }
            pdata += net_width;//\CF\C2?\D0\D0
        }

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
            temp_mask_proposals.push_back(picked_proposals[idx]);
            temp_output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params[img_index];
        mask_params.srcImgShape = srcImgs[img_index].size();
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

tunnel_inspection::tunnel_inspection(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    ui.oringinal_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui.oringinal_label->setScaledContents(true);
    ui.save_pushbutton->hide();
    // 算法线程初始化
    alg_thread_.init([this](const result& ret, const cv::Mat& frame) {
        // cv::cvtColor(frame, frame, CV_BGR2RGB);
        if (frame.empty()) {
            ui.progress_bar->setMaximum(ret.bar_value);
            return;
        }
        ui.time_label->setText(QString::number(ret.proc_time));
        signals_bar(ret.bar_value + 2);
        if (!ui.radio_button->isChecked())return;
        cv::Mat tframe = frame.clone();
        QImage qimage = QImage((uchar*)tframe.data, tframe.cols, tframe.rows,
            tframe.cols * tframe.channels(), QImage::Format_Grayscale8);
        ui.oringinal_label->setPixmap(QPixmap::fromImage(qimage));


        //ui.progress_bar->setValue(ret.bar_value);

    });




    connect(this, &tunnel_inspection::signals_bar, ui.progress_bar, [=](int value) {
        ui.progress_bar->setValue(value); // 更新进度条的值  
    });
}

tunnel_inspection::~tunnel_inspection()
{}
void tunnel_inspection::on_load_pushbutton_clicked() {
    //文件夹路径
    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "choose src Directory",
        "/");
    ui.load_lineedit->setText(src_dirpath);

}
void tunnel_inspection::on_load_pushbutton_2_clicked() {
    //文件夹路径
    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "choose src Directory",
        "/");
    ui.load_lineedit_2->setText(src_dirpath);

}
void tunnel_inspection::on_path_pushbutton_clicked() {
    //文件夹路径
    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "choose src Directory",
        "/");
    ui.path_lineedit->setText(src_dirpath);
    for (int i = 1; i <= CAMERANUMBER; ++i) {
        QString folderName = QString("%1").arg(i);
        QString fullPath = src_dirpath + "/" + folderName;
        bool result = QDir().mkpath(fullPath);
    }
    if (ui.path_lineedit->text() != nullptr)
        ui.save_pushbutton->setEnabled(true);

}
void tunnel_inspection::on_result_path_pushbutton_clicked() {
    //文件夹路径
    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "choose src Directory",
        "/");
    ui.result_path_lineedit->setText(src_dirpath);
    for (int i = 1; i <= CAMERANUMBER; ++i) {
        QString folderName = QString("%1").arg(i);
        QString fullPath = src_dirpath + "/" + folderName;
        bool result = QDir().mkpath(fullPath);
    }

}
void save_all(tunnel_inspection *ti, std::string files_name1, std::string  files_name2, std::string  store_files_name)
{

    std::vector<std::string>files_names;
    std::string dataname = "Camera";
    for (int i = 0; i < CAMERANUMBER; i++) {
        if (i < CAMERANUMBER / 2) {

            files_names.push_back(files_name1 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }
        else {
            files_names.push_back(files_name2 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }

    }



    for (int i = 0; i < files_names.size(); i++) {



        std::ifstream is(files_names[i], std::ios::in | std::ios::binary);
        if (is) {
            is.seekg(0, is.end);
            unsigned long long fsize = is.tellg();
            unsigned int  length = CAMERASIZE * CAMERASIZE;

            is.seekg(0, is.beg);//从开始移动20个字节//55

            // is.seekg((length*1000)-fsize, is.end);

            unsigned int numbers = fsize / length;

            float bed = float(10) / float(numbers);

            char * buffer = new char[CAMERASIZE * CAMERASIZE];

            for (int j = 0; j < numbers; j++) {
                emit ti->signals_bar(i * 10 + bed * j);

                is.read(buffer, length);
                cv::Mat tempimg(CAMERASIZE, CAMERASIZE, CV_8UC1, buffer);
                cv::imwrite(store_files_name + "/" + std::to_string(i + 1) + "/" + std::to_string(j) + ".jpg", tempimg);

            }
            is.close();
            delete[] buffer;
        }

    }
}
void tunnel_inspection::on_save_pushbutton_clicked() {

    auto files_name1 = ui.load_lineedit->text().toStdString();
    auto files_name2 = ui.load_lineedit_2->text().toStdString();
    auto store_files_name = ui.path_lineedit->text().toStdString();


    /*
        ui.progress_bar->setMaximum(80);
        std::thread ta(save_all, this, files_name1, files_name2, store_files_name);
        ta.detach();
        return;
    */


    std::vector<std::string>files_names;
    std::string dataname = "DalsaCamera5";
    for (int i = 0; i < CAMERANUMBER; i++) {
        if (i < 4) {

            files_names.push_back(files_name1 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }
        else {
            files_names.push_back(files_name2 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }

    }
    ui.progress_bar->setMaximum(80);


    for (int i = 0; i < files_names.size(); i++) {

        ui.progress_bar->setValue(i * 10);

        std::ifstream is(files_names[i], std::ios::in | std::ios::binary);
        if (is) {
            is.seekg(0, is.end);
            unsigned long long fsize = is.tellg();
            unsigned int  length = CAMERASIZE * CAMERASIZE;

            is.seekg(0, is.beg);//从开始移动20个字节//55

            // is.seekg((length*1000)-fsize, is.end);

            unsigned int numbers = fsize / length;

            float bed = float(10) / float(numbers);

            char * buffer = new char[CAMERASIZE * CAMERASIZE];

            for (int j = 0; j < numbers; j++) {
                emit signals_bar(i * 10 + bed * j);

                is.read(buffer, length);
                cv::Mat tempimg(CAMERASIZE, CAMERASIZE, CV_8UC1, buffer);
                cv::imwrite(store_files_name + "/" + std::to_string(i + 1) + "/" + std::to_string(j) + ".jpg", tempimg);

            }
            is.close();
            delete[] buffer;
        }





    }







}
#pragma once
#include "ys_utils.h"
using namespace cv;
using namespace std;
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize) {
    if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
    {
        cout << "Error:_netHeight and _netWidth must be multiple of max stride " << netStride[strideSize - 1] << "!" << endl;
        return false;
    }
    return true;
}

void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
    bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputSeg>& output, const MaskParams& maskParams) {
    //cout << maskProtos.size << endl;

    int seg_channels = maskParams.segChannels;
    int net_width = maskParams.netWidth;
    int seg_width = maskParams.segWidth;
    int net_height = maskParams.netHeight;
    int seg_height = maskParams.segHeight;
    float mask_threshold = maskParams.maskThreshold;
    Vec4f params = maskParams.params;
    Size src_img_shape = maskParams.srcImgShape;

    Mat protos = maskProtos.reshape(0, { seg_channels,seg_width * seg_height });

    Mat matmul_res = (maskProposals * protos).t();
    Mat masks = matmul_res.reshape(output.size(), { seg_width,seg_height });
    vector<Mat> maskChannels;
    split(masks, maskChannels);
    for (int i = 0; i < output.size(); ++i) {
        Mat dest, mask;
        //sigmoid
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);

        Rect roi(int(params[2] / net_width * seg_width), int(params[3] / net_height * seg_height), int(seg_width - params[2] / 2), int(seg_height - params[3] / 2));
        dest = dest(roi);
        resize(dest, mask, src_img_shape, INTER_NEAREST);

        //crop
        Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > mask_threshold;
        output[i].boxMask = mask;
    }
}

void GetMask2(const Mat& maskProposals, const Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams) {
    int seg_channels = maskParams.segChannels;
    int net_width = maskParams.netWidth;
    int seg_width = maskParams.segWidth;
    int net_height = maskParams.netHeight;
    int seg_height = maskParams.segHeight;
    float mask_threshold = maskParams.maskThreshold;
    Vec4f params = maskParams.params;
    Size src_img_shape = maskParams.srcImgShape;

    Rect temp_rect = output.box;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

    //如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
    rang_w = MAX(rang_w, 1);
    rang_h = MAX(rang_h, 1);
    if (rang_x + rang_w > seg_width) {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x -= 1;
    }
    if (rang_y + rang_h > seg_height) {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y -= 1;
    }

    vector<Range> roi_rangs;
    roi_rangs.push_back(Range(0, 1));
    roi_rangs.push_back(Range::all());
    roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
    roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

    //crop
    Mat temp_mask_protos = mask_protos(roi_rangs).clone();
    Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
    Mat matmul_res = (maskProposals * protos).t();
    Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
    Mat dest, mask;

    //sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
    int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
    int width = ceil(net_width / seg_width * rang_w / params[0]);
    int height = ceil(net_height / seg_height * rang_h / params[1]);

    resize(dest, mask, Size(width, height), INTER_NEAREST);
    float tt[10] = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 };

    Mat testm(1, 10, CV_32FC1, tt);




    for (int i = 0; i < testm.rows; ++i)
    {
        //获取第 i 行首像素指针
        float * p = testm.ptr<float>(i);
        //对第 i 行的每个像素(byte)操作
        for (int j = 0; j < testm.cols; ++j) {
            p[j] = 0.1*j;

        }


    }
    testm = testm > 0.5;
    auto kk1 = testm.at<uchar>(0, 0);
    auto kk2 = testm.at<uchar>(0, 7);



    mask = mask(temp_rect - Point(left, top)) > mask_threshold;
    output.boxMask = mask;

}

void DrawPred(Mat& img, vector<OutputSeg> result, std::vector<std::string> classNames, vector<Scalar> color) {
    Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        int left, top;
        left = result[i].box.x;
        top = result[i].box.y;
        int color_num = i;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);
        if (result[i].boxMask.rows&& result[i].boxMask.cols > 0)
            mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
        string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
    addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
//    imshow("1", img);
    //imwrite("out.bmp", img);
//    waitKey();
    //destroyAllWindows();

}
void tunnel_inspection::on_start_pushbutton_clicked() {
    auto files_name1 = ui.load_lineedit->text().toStdString();
    auto files_name2 = ui.load_lineedit_2->text().toStdString();
    auto store_files_name = ui.path_lineedit->text().toStdString();
    auto  maxbar_value = alg_thread_.set_data_name(files_name1, files_name2, store_files_name);
    ui.progress_bar->setMaximum(maxbar_value);

    std::vector<std::string>files_names;
    for (int i = 0; i < CAMERANUMBER; i++) {
        if (i < 4) {

            files_names.push_back(files_name1 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }
        else {
            files_names.push_back(files_name2 + "/DalsaCamera" + std::to_string(i + 1) + ".img");
        }

    }


    //for (int i = 0; i < CAMERANUMBER; i++) {
    //	file_datas_[i].init([this](file_data::frame& frame) {
    //		alg_thread_.push_frame(frame.clone());
    //	}, files_names[i], i);
    //	file_datas_[i].start();

    //	file_datas_[i].set_params(store_files_name+"/"+std::to_string(i+1)+"/", ui.save_radio_button->isChecked()&&ui.path_lineedit->text()!=nullptr);
    //	
    //}

    alg_thread_.start();




}
void tunnel_inspection::on_picture_pushbutton_clicked() {

}
void tunnel_inspection::update_bar() {


}

#pragma once
#include "ys_utils.h"
using namespace cv;
using namespace std;
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize) {
    if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
    {
        cout << "Error:_netHeight and _netWidth must be multiple of max stride " << netStride[strideSize - 1] << "!" << endl;
        return false;
    }
    return true;
}

void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
    bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
    if (false) {
        int maxLen = MAX(image.rows, image.cols);
        outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputSeg>& output, const MaskParams& maskParams) {
    //cout << maskProtos.size << endl;

    int seg_channels = maskParams.segChannels;
    int net_width = maskParams.netWidth;
    int seg_width = maskParams.segWidth;
    int net_height = maskParams.netHeight;
    int seg_height = maskParams.segHeight;
    float mask_threshold = maskParams.maskThreshold;
    Vec4f params = maskParams.params;
    Size src_img_shape = maskParams.srcImgShape;

    Mat protos = maskProtos.reshape(0, { seg_channels,seg_width * seg_height });

    Mat matmul_res = (maskProposals * protos).t();
    Mat masks = matmul_res.reshape(output.size(), { seg_width,seg_height });
    vector<Mat> maskChannels;
    split(masks, maskChannels);
    for (int i = 0; i < output.size(); ++i) {
        Mat dest, mask;
        //sigmoid
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);

        Rect roi(int(params[2] / net_width * seg_width), int(params[3] / net_height * seg_height), int(seg_width - params[2] / 2), int(seg_height - params[3] / 2));
        dest = dest(roi);
        resize(dest, mask, src_img_shape, INTER_NEAREST);

        //crop
        Rect temp_rect = output[i].box;
        mask = mask(temp_rect) > mask_threshold;
        output[i].boxMask = mask;
    }
}

void GetMask2(const Mat& maskProposals, const Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams) {
    int seg_channels = maskParams.segChannels;
    int net_width = maskParams.netWidth;
    int seg_width = maskParams.segWidth;
    int net_height = maskParams.netHeight;
    int seg_height = maskParams.segHeight;
    float mask_threshold = maskParams.maskThreshold;
    Vec4f params = maskParams.params;
    Size src_img_shape = maskParams.srcImgShape;

    Rect temp_rect = output.box;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

    //如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
    rang_w = MAX(rang_w, 1);
    rang_h = MAX(rang_h, 1);
    if (rang_x + rang_w > seg_width) {
        if (seg_width - rang_x > 0)
            rang_w = seg_width - rang_x;
        else
            rang_x -= 1;
    }
    if (rang_y + rang_h > seg_height) {
        if (seg_height - rang_y > 0)
            rang_h = seg_height - rang_y;
        else
            rang_y -= 1;
    }

    vector<Range> roi_rangs;
    roi_rangs.push_back(Range(0, 1));
    roi_rangs.push_back(Range::all());
    roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
    roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

    //crop
    Mat temp_mask_protos = mask_protos(roi_rangs).clone();
    Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
    Mat matmul_res = (maskProposals * protos).t();
    Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
    Mat dest, mask;

    //sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
    int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
    int width = ceil(net_width / seg_width * rang_w / params[0]);
    int height = ceil(net_height / seg_height * rang_h / params[1]);

    resize(dest, mask, Size(width, height), INTER_NEAREST);
    float tt[10] = { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 };

    Mat testm(1, 10, CV_32FC1, tt);




    for (int i = 0; i < testm.rows; ++i)
    {
        //获取第 i 行首像素指针
        float * p = testm.ptr<float>(i);
        //对第 i 行的每个像素(byte)操作
        for (int j = 0; j < testm.cols; ++j) {
            p[j] = 0.1*j;

        }


    }
    testm = testm > 0.5;
    auto kk1 = testm.at<uchar>(0, 0);
    auto kk2 = testm.at<uchar>(0, 7);



    mask = mask(temp_rect - Point(left, top)) > mask_threshold;
    output.boxMask = mask;

}

void DrawPred(Mat& img, vector<OutputSeg> result, std::vector<std::string> classNames, vector<Scalar> color) {
    Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        int left, top;
        left = result[i].box.x;
        top = result[i].box.y;
        int color_num = i;
        rectangle(img, result[i].box, color[result[i].id], 2, 8);
        if (result[i].boxMask.rows&& result[i].boxMask.cols > 0)
            mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
        string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
    }
    addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
//    imshow("1", img);
    //imwrite("out.bmp", img);
//    waitKey();
    //destroyAllWindows();

}
#include <fstream>
#include <string> 

#include "data_process.h"
#include "img.h"
#include<chrono>
#include <numeric>
#define CKAI 1
//using namespace cv;
struct linepoints
{
    std::vector<cv::Point> points;
    cv::Vec4f lines;
};
data_process::data_process() {
    std::ifstream inf;
    inf.open("60.txt");
    std::string s;

    int line_number = 0;
    while (getline(inf, s) && line_number < 8) {
        params_[line_number] = atof(s.c_str());
        line_number++;
    }

    inf.close();
    is_ysonnx_ = yvonnx_.ReadModel("ck.par", false, 0, false);


}
result data_process::process_profile(bool left, float *profilex0, float *profiley0, int size0, float *profilex1, float *profiley1, int size1, int rails) {
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();


    frequency_ = frequency_ + 1;
    result rst;
    if (size0 < 100 || size1 < 100) {
        int pose0 = (frequency_ - 1) % 5;
        width_[pose0] = 0;
        height_[pose0] = 0;
        return rst;

    }

    std::vector<float> source_x;
    std::vector<float> source_y;
    std::vector<float> standard_x;
    std::vector<float> standard_y;
    std::vector<cv::Point2f> source0s, sources;
    std::vector<float> trans_x;
    std::vector<float> trans_y;


    for (int i = 0; i < size1; i++) {
        profilex1[i] = profilex1[i] * (-1);
        profiley1[i] = profiley1[i] - 500;

        cv::Point2f point, tpoint;
        point.y = profilex1[i];
        point.x = profiley1[i];

        tpoint.y = profilex1[i];
        tpoint.x = profiley1[i];

        profiley1[i] = tpoint.y;
        profilex1[i] = tpoint.x;
        sources.push_back(point);
    }



    for (int i = 0; i < size0; i++) {
        profilex0[i] = profilex0[i] * (-1);
        profiley0[i] = profiley0[i] - 500;

        cv::Point2f  tpoint;


        tpoint.y = profilex0[i];
        tpoint.x = profiley0[i];

        profiley0[i] = tpoint.y;
        profilex0[i] = tpoint.x;
        source0s.push_back(tpoint);
    }

    double angles[2];


    angles[0] = atan(params_[1] / params_[0]);
    angles[1] = atan(params_[5] / params_[4]);



    double x0 = params_[2];
    double y0 = params_[3];

    float icpx, icpy;

    double angle = 0;


    //开始校正
    double  mtheta = 0;
    double mx = 0;
    double my = 0;
    double minx = 0;
    double maxx = 0;
    icpx = 0;
    icpy = 0;

    std::vector<cv::Point2f> tsources, lines;
#if 1 //通过顶面的斜率校正角度，误差较大
    std::sort(sources.begin(), sources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y > b.y; });

    int minHessian = 400;



    for (int i = 0; i < sources.size(); i++) {
        if (sources[0].y - sources[i].y < 30.0)
            tsources.push_back(sources[i]);
        else break;

    }

    std::sort(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.x < b.x; });


    maxx = std::max_element(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.x < b.x; })->x - 26.5;
    minx = maxx - 20;

    for (auto point : tsources) {
        if (point.x > minx&&point.x < maxx) {
            lines.push_back(point);
        }
    }
    /*   cv::Vec4f line;

       cv::fitLine(lines, line, cv::DIST_L2, 0, 1e-2, 1e-2);
       auto linek = line[1] / line[0];
       double lineb = line[3] - linek * line[2];
       auto parallel = atan(linek)*180.0 / PI;
       mtheta = atan(linek);*/
       // return mtheta;


#endif



    //    cv::Mat  dynamicdlt = cv::Mat::zeros(8, 1, CV_32F);

    bool result1 = false;
    bool result2 = false;

    float dynamic_angle = 0;
    //找不到圆心会耗时加重
    //auto havecircle = img::find_dynamic_circle_points(sources, left,source_x, source_y, standard_x, standard_y);//圆心位置会有一定影响
    //if (havecircle) {
        //dynamic_angle = img::translate(1, source_x, source_y, standard_x, standard_y, trans_x, trans_y, dynamicdlt);
        //icpx = dynamicdlt.at<float>(2 + 1 * 4, 0);
       // icpy = dynamicdlt.at<float>(3 + 1 * 4, 0);
    //}


    if (abs(dynamic_angle) < 0.00000001) {
        dynamic_angle = angles[1];
        icpx = params_[6];
        icpy = params_[7];
    }

    std::vector<cv::Point2f> cstandards, csources, csource2s;

    /********************
    由于拟合使用的是轨腰部位400mm与20mm的圆心拟合，拟合算法为随机采样，角度一致性存在1度左右误差。
    所以校正时采用不同部位，如轨鄂和顶面中心的点进行，查找特定的两点，轨鄂与中心点，通过两点进行icp校正，并保证一致性。
    *******************/
    csources = sources;
    tsources.clear();
    source_x.clear();
    source_y.clear();
    standard_x.clear();
    standard_y.clear();
    std::vector<cv::Point2f> tsources2;


    /******************************************************************
   数据分段，校正后总共产生上下两部分，tsources，tsources2，当轮廓仪经过螺孔时
   数据拟合圆心会失败，因其螺孔周边产生了太多噪声
   *************************************************************************/


    for (int i = 0; i < sources.size(); i++) {
        csources[i].x = params_[4] * sources[i].x + params_[5] * sources[i].y + params_[6];
        csources[i].y = params_[5] * (-sources[i].x) + params_[4] * sources[i].y + params_[7];

    }
    auto heightpoint = *std::max_element(csources.begin(), csources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y < b.y; });
    auto hvalue = heightpoint.y - 60;

    for (int i = 0; i < csources.size(); i++) {
        if (csources[i].y > hvalue) {
            if (tsources.size() == 0) {
                tsources.push_back(csources[i]);
                continue;
            }
            if (tsources.size() != 0 && sqrt(pow(csources[i].x - tsources[tsources.size() - 1].x, 2) + pow(csources[i].y - tsources[tsources.size() - 1].y, 2)) < 5) { tsources.push_back(csources[i]); }//去除孤立点
        }

        if (csources[i].y < hvalue && csources[i].x>0) {
            tsources2.push_back(csources[i]);
        }

    }












    if (tsources.size() < 1)return  rst;
    auto  minpoint = *std::max_element(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y > b.y; });

    left ? standard_x.push_back(36.4264) : standard_x.push_back(-36.4264);
    standard_y.push_back(141.066);
    standard_x.push_back(0);
    standard_y.push_back(176);
    left ? standard_x.push_back(10.2) : standard_x.push_back(-10.2);
    standard_y.push_back(39.5701);

    double tdistance = sqrt(pow(standard_x[0], 2) + pow(176 - 141.066, 2));
    double tdistance2 = sqrt(pow(standard_x[0] - standard_x[2], 2) + pow(39.5701 - 141.066, 2));
    cv::Point2f maxpoint, maxpoint2;
    double mindistance = 10000;
    double mindistance2 = 10000;



    for (auto point : tsources) {
        auto distance = abs(sqrt(pow(point.x - minpoint.x, 2) + pow(point.y - minpoint.y, 2)) - tdistance);
        if (distance < mindistance) {
            maxpoint = point;
            mindistance = distance;
        }

    }
    for (auto point : tsources2) {
        auto distance = abs(sqrt(pow(point.x - minpoint.x, 2) + pow(point.y - minpoint.y, 2)) - tdistance2);
        if (distance < mindistance2) {
            maxpoint2 = point;
            mindistance2 = distance;
        }

    }


    source_x.push_back(minpoint.x);
    source_y.push_back(minpoint.y);
    source_x.push_back(maxpoint.x);
    source_y.push_back(maxpoint.y);
    source_x.push_back(maxpoint2.x);
    source_y.push_back(maxpoint2.y);
    cv::Mat dynamicdlt = cv::Mat::zeros(8, 1, CV_32F);
    auto check_angle = img::translate(1, source_x, source_y, standard_x, standard_y, trans_x, trans_y, dynamicdlt);
    angle = angles[1];
    if (abs(check_angle) > 0.00000001&&abs(check_angle) < 0.1) {
        auto ttt = dynamic_angle + check_angle;
        auto ticpx = icpx;
        auto ticpy = icpy;
        angle = dynamic_angle + check_angle;
        icpx = cos(check_angle)*ticpx + sin(check_angle)*ticpy + dynamicdlt.at<float>(2 + 1 * 4, 0);
        icpy = cos(check_angle)*ticpy - sin(check_angle)*ticpx + dynamicdlt.at<float>(3 + 1 * 4, 0);
    }

    //if (dynamic_angle + check_angle > 0.2) {
    //    angle = dynamic_angle + check_angle;
    //}



    angle = angle - angles[1];

    //   angles[1] = angle + angles[1];
    angles[0] += angle;
    angles[1] += angle;
    auto alpha = atan(y0 / x0) - angle;
    auto sl = sqrt(pow(x0, 2) + pow(y0, 2));
    x0 = sl * cos(alpha);
    y0 = sl * sin(alpha);


    for (int i = 0; i < size0; i++) {
        source0s[i].x = cos(angles[0])*profilex0[i] + sin(angles[0])*profiley0[i];
        source0s[i].y = -sin(angles[0])*profilex0[i] + cos(angles[0])*profiley0[i];
        profilex0[i] = source0s[i].x;
        profiley0[i] = source0s[i].y;

    }
    for (int i = 0; i < size1; i++) {
        sources[i].x = cos(angles[1])*profilex1[i] + sin(angles[1])*profiley1[i];
        sources[i].y = -sin(angles[1])*profilex1[i] + cos(angles[1])*profiley1[i];
        profilex1[i] = sources[i].x + x0;
        profiley1[i] = sources[i].y + y0;

    }









#if 0

    cv::Mat images = cv::Mat::zeros(1000, 1000, CV_8UC3);

    for (int i = 0; i < source0s.size(); i++) {
        float cvy = ((source0s[i].y + 600));
        float cvx = ((source0s[i].x + 600));
        if (0 < cvx&&cvx < images.cols && 0 < cvy&&cvy < images.rows)
            images.at<cv::Vec3b>(cvy, cvx) = cv::Vec3b(255, 255, 255);

    }




    cv::resize(images, images, cv::Size(1000, 1000));
    // cv::flip(images, images, 0);
    imshow("source0s", images);
    cv::waitKey(2);

#endif





    cv::Point2f fpoints[2];//高度与拉出由不同的点计算


    std::vector<cv::Point2f> underside_points;

    cv::Point2f abrasion;
    bool existance;
    bool expanding = false;
    fpoints[0] = img::dynamic_all_third_featch(source0s, underside_points, abrasion, existance, expanding, rails);
    fpoints[1] = img::get_featch_point(sources);
    rst.imgtype = feature::none;

    //拉出值为 a = L + x’t - xt
// 导高值为   b = yt + y’t - H

    rst.width = abs(fpoints[0].x) - abs(fpoints[1].x) + x0;
    rst.height = abs(fpoints[0].y) + abs(fpoints[1].y) - y0;

    int shield_size = 0;
    double distance = 0.0;
    auto joints = img::expansion_jionts(source0s, shield_size, distance);
    rst.shield = shield_size;







    int pose = (frequency_ - 1) % 5;
    width_[pose] = rst.width;
    height_[pose] = rst.height;
    shild_[pose] = shield_size;
    int hshild = 0;
    int hend = 0;
    if (frequency_ > 5) {
        double sumw = std::accumulate(width_, width_ + 5, 0) / 5.0;
        double sumh = std::accumulate(height_, height_ + 5, 0) / 5.0;
        double wm = 0;
        double wh = 0;
        double sw = 100;
        double sh = 100;
        double sw2 = width_[(frequency_ - 2) % 5];
        double sh2 = height_[(frequency_ - 2) % 5];
        int wi = 0;
        int hi = 0;
        for (int i = 0; i < 5; i++) {
            if (shild_[i] > 130)
                hshild++;

            if (height_[i] > 250)
                hend++;
            if (abs(sumw - width_[i]) > wm) {
                wm = abs(sumw - width_[i]);
                wi = i;

            }
            if (abs(sumh - height_[i]) > wh) {
                wh = abs(sumh - height_[i]);
                hi = i;

            }
            if (abs(width_[i] - 752.5) < sw) {
                sw2 = width_[i];
                sw = abs(width_[i] - 752.5);
            }
            if (abs(height_[i] - 200) < sh) {
                sh2 = height_[i];
                sh = abs(height_[i] - 200);
            }


        }
        //if (wi == pose && wm > 5) {
        if (abs(752.5 - width_[pose]) > 30 && width_[pose] != 0 && abs(752.5 - width_[pose]) < 146) {
            rst.width = width_[(frequency_ - 2) % 5];
            //rst.width = width_[hi];
            //rst.width = sw2;
            //width_[pose] = rst.width;

        }
        //if (hi == pose && wh > 5) {
        if (abs(sumh - height_[pose]) > 5 && height_[pose] != 0) {
            rst.height = height_[(frequency_ - 1) % 5];
            // rst.height = height_[hi];
            // rst.height = sh2;
             //height_[pose] = rst.height;

        }



    }



    // std::cout << "0=" << width_[0] << ":1=" << width_[1] << ":2=" << width_[2] << ":3=" << width_[3] << ":4=" << width_[4] << ":5=" << std::endl;







    bool  insulating = false;
    int sdistance0 = 0;
    if (source0s.size() > 2) {
        sdistance0 = abs(source0s[0].x - source0s[source0s.size() - 1].x);
    }

    if (shield_size < 91 && hend>3)rst.imgtype = feature::endbend;

    if (90 < shield_size&&shield_size < 131 && distance < 1.0&&rst.height < 290 && rst.height > 150 && rst.width < 800 && hshild < 1)insulating = true;


    // if (shield_size < 91 && height_[0] > 250 && height_[1] > 250 && height_[2] > 250 && height_[3] > 250 && height_[4] > 250)rst.imgtype = feature::endbend;

    if (rst.imgtype == feature::endbend&&rst.height < 290 && rst.height > 270 && distance < 1.0)insulating = true;
    if (insulating)rst.imgtype = feature::insulatorbracket;
    if (shield_size > 130 && sdistance0 > 50 && distance < 10 && source0s.size()>650 && height_[pose] > 150 && width_[pose] > 700 && hshild > 1)rst.imgtype = feature::expansion;
    //if(expanding&&distance>3&&distance<10)rst.imgtype = feature::expansion;

   // rst.gap[0] = distance;
    if (rst.width > 900 || rst.width < 500 || rst.height>320)rst.imgtype = feature::none;


    if (!existance) {
        auto duration_capture = clock::now() - last;
        rst.time = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        return rst;

    }





    cv::Vec4f line;
    cv::fitLine(underside_points, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    auto linek = line[1] / line[0];
    double lineb = line[3] - linek * line[2];
    rst.parallel = (atan(linek)*180.0 / PI);

    rst.parallel = rst.parallel - int(rst.parallel);

    rst.abrasion = abs(abrasion.y - linek * abrasion.x - lineb) / sqrt(1 + linek * linek);
    if (rails == 0) {
        rst.abrasion > 3 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 3 - rst.abrasion;

    }
    if (rails == 1) {
        rst.abrasion > 13 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 13 - rst.abrasion;
        rst.abrasion > 6 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : 1;
    }
    if (rails == 2) {
        rst.abrasion > 12.4 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 12.4 - rst.abrasion;
        rst.abrasion > 5 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : 1;
    }

    rst.pointdepth = -1;

    for (auto point : underside_points) {
        auto distance = abs(point.y - linek * point.x - lineb) / sqrt(1 + linek * linek);
        if (rst.pointdepth < distance)
            rst.pointdepth = distance;
    }











    //rst.parallel = angle * 180.0 / PI;//测试用
    auto duration_capture = clock::now() - last;

    rst.time = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
    //rst.parallel = angle;
    return rst;


}
result data_process::process_img(unsigned char  *datas, int size, int rails) {
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();
    result rst;
    if (size / 2048 < 1500)return rst;
    cv::Mat palyimg, img2, dstImage;
    auto image = cv::Mat::Mat(size / 2048, 2048, CV_8UC1, datas).clone();







    cv::Rect roi;
    if (rails % 2 != 0) {
        rst = process_shield_img(datas, roi, palyimg, size, rails);
        return  rst;
    }


    if (rails % 2 != 0 || image.cols < 1500) {
        // cv::imwrite("H:/third-rail/img-qingdao/testgap/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);
        return  rst;
    }


#if CKAI

    if (!is_ysonnx_)return rst;
    std::vector<OutputSeg> output, output2;
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    bool find = yvonnx_.OnnxDetect(image, output);
    std::vector<cv::Scalar> color;

    cv::RotatedRect rotate_rects[2];

    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(0, 0, 255));
    }

    if (find) {

        int size_gap = 0;
        for (int k = 0; k < output.size(); k++) {
            if (output[k].id != 0)continue;

            auto bmask = output[k].boxMask;
            std::vector<cv::Point>points;
            for (int i = 0; i < bmask.rows; i++)
            {
                const uchar* ph = bmask.ptr<uchar>(i);
                for (int j = 0; j < bmask.cols; j++)
                {
                    if (ph[j] > 0) {
                        //points.push_back(cv::Point(j,i));
                        points.push_back(cv::Point(j + output[k].box.x, i + output[k].box.y));//真实坐标
                    }
                }

            }
            if (points.size() > 1 && size_gap < 2) {
                rotate_rects[size_gap] = cv::minAreaRect(points);
                output2.push_back(output[k]);
                size_gap++;
            }


        }


















        float h1, h2;
        abs(rotate_rects[0].angle) < 45 ? h1 = rotate_rects[0].size.height : h1 = rotate_rects[0].size.width;
        abs(rotate_rects[1].angle) < 45 ? h2 = rotate_rects[1].size.height : h2 = rotate_rects[1].size.width;

        if (output2.size() > 1) {
            if (output2[0].box.y < output2[1].box.y) {
                auto th = h1;
                h1 = h2;
                h2 = th;
                auto sangle = rotate_rects[0].angle;
                rotate_rects[0].angle = rotate_rects[1].angle;
                rotate_rects[1].angle = sangle;
            }
        }


        auto duration_capture1 = clock::now() - last;
        float radian1 = abs(rotate_rects[0].angle) / 180 * CV_PI;
        float radian2 = abs(rotate_rects[1].angle) / 180 * CV_PI;
        if (abs(rotate_rects[0].angle) > 45) {
            radian1 = (90 - abs(rotate_rects[0].angle)) / 180 * CV_PI;


        }
        if (abs(rotate_rects[1].angle) > 45) {

            radian2 = (90 - abs(rotate_rects[1].angle)) / 180 * CV_PI;

        }


        rst.gap[0] = h1 * 0.775 / abs(cos(radian1));
        rst.gap[1] = h2 * 0.775 / abs(cos(radian2));
        auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
        rst.time = timeuse;









        /************************显示开始*************************/


        DrawPred(image, output, yvonnx_._className, color);
        cv::Point2f vertices[4], vertices1[4];
        rotate_rects[0].points(vertices);//获取矩形的四个点
        rotate_rects[1].points(vertices1);//获取矩形的四个点
        std::string label = "TimeUse: " + std::to_string(timeuse) + "=" + std::to_string(rst.gap[0]) + "=" + std::to_string(rst.gap[1]);

        for (int i = 0; i < 4; i++) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::line(image, vertices1[i], vertices1[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            // cv::line(palyimg, cv::Point2f(vertices[i].x + output[0].box.x, vertices[i].y + output[0].box.y), cv::Point2f(vertices[(i + 1) % 4].x + output[0].box.x, vertices[(i + 1) % 4].y + output[0].box.y), cv::Scalar(0, 255, 0));
             //cv::line(palyimg, cv::Point2f(vertices1[i].x + output[1].box.x, vertices1[i].y + output[1].box.y), cv::Point2f(vertices1[(i + 1) % 4].x + output[1].box.x, vertices1[(i + 1) % 4].y + output[1].box.y), cv::Scalar(0, 255, 0));

        }
        cv::putText(image, label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2, 8);
        cv::resize(image, image, cv::Size(1024, 1024));
        cv::imshow("result", image);
        std::cout << h1 << "=h1h2=" << h2 << std::endl;
        cv::waitKey();
        /************************显示结束*************************/


        return  rst;
    }

    else

        return  rst;








#endif

    roi.y = 0;
    roi.height = image.rows;
    if (image.rows < 2600 || roi.width<1000 || roi.height<1600 || roi.x + roi.width>image.cols || roi.y + roi.height>image.rows) {
        roi.x = 400;
        roi.width = image.cols - 400;//600
        roi.y = 0;
        roi.height = image.rows;
    }




    img2 = image(roi).clone();
    {

        int mean_value = static_cast<int>(cv::mean(img2).val[0]);
        unsigned int gray_value = 20;
        gray_value = mean_value / 2;
        cv::medianBlur(img2, img2, 11);
        cv::Canny(img2, img2, gray_value, (gray_value) * 2);
        // cv::imwrite("d:/61.bmp", img2);

      /*   img = img2.clone();
         cv::cvtColor(img, img, CV_GRAY2BGR);*/

        std::vector<std::vector<cv::Point>>contours, contoursfilter;
        std::vector<cv::Vec4i>hierarchy;
        cv::findContours(img2, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        bool is_coincidence = false;
        std::vector<cv::Vec4f> lines;
        std::vector<double> ks, negatives, positives, ppoints;
        std::vector<cv::Point> linepoints;
        for (int i = 0; i < contours.size(); i++)
        {
            for (auto contour : contours[i]) {
                linepoints.push_back(contour);
            }

            if (contours[i].size() > 100)//crect.width / crect.height>3&&
            {
                cv::Vec4f line;
                cv::fitLine(contours[i], line, cv::DIST_L2, 0, 1e-2, 1e-2);
                auto crect = cv::boundingRect(contours[i]);

                double k = line[1] / line[0];





                if (crect.width > crect.height) {
                    if (abs(k) > 0.2&& abs(k) < 0.4) {
                        k = atan(k)*180.0 / CV_PI;
                        k > 0 ? positives.push_back(k) : negatives.push_back(k);

                    }


                    contoursfilter.push_back(contours[i]);


                    float kk = 0.3;
                    k > 0 ? kk = 0.3 : kk = -0.3;
                    std::vector<float> bs;
                    for (int j = 0; j < contours[i].size(); j++) {
                        bs.push_back(contours[i][j].y - kk * contours[i][j].x);
                    }
                    cv::Mat meanb, stdb;
                    cv::meanStdDev(bs, meanb, stdb);
                    bs.size();

                    auto tmean = meanb.at<double>(0, 0);
                    auto tstdb = stdb.at<double>(0, 0);





                }




            }

        }
        int direction = 1;





        positives.size() > negatives.size() ? direction = -1 : direction = 1;
        direction > 0 ? ppoints = negatives : ppoints = positives;

        if (ppoints.size() < 2)return rst;
        cv::Mat pmean, pstd;
        cv::meanStdDev(ppoints, pmean, pstd);
        auto tpmean = pmean.at<double>(0, 0);
        double tpp = 1000;
        double anglel = 17.5;

        for (auto point : ppoints) {
            if (abs(point - tpmean) < tpp) {
                tpp = abs(point - tpmean);
                anglel = point;
            }

        }






        //tpmean < 0 ? anglel = anglel * -1 : anglel = anglel;


        float radian = (float)(-1) * anglel / 180 * CV_PI;

        cv::Mat outimg(2000, 2000, CV_8U, cv::Scalar(0));
        cv::cvtColor(outimg, outimg, CV_GRAY2BGR);//结果图片为彩色

        for (auto &point : linepoints) {
            auto tpoint = point;
            point.x = tpoint.x*cos(radian) - tpoint.y*sin(radian);
            point.y = tpoint.x*sin(radian) + tpoint.y*cos(radian);


        }


        cv::Mat hist_img(2000, 2000, CV_8U, cv::Scalar(255));


        cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色


        if (linepoints.size() < 2)return rst;


        std::sort(linepoints.begin(), linepoints.end(), [](cv::Point point1, cv::Point point2) {
            return point1.y < point2.y;
        });



        int poffset = 0;
        if (-linepoints[0].y > 0)poffset = -linepoints[0].y;
        int prow = linepoints[linepoints.size() - 1].y + poffset + 1;
        std::vector<cv::Point> anglepoints(prow);


        for (auto &point : linepoints) {
            anglepoints[point.y + poffset].x++;
            anglepoints[point.y + poffset].y = point.y + poffset;
        }


        std::vector<cv::Point3i> results;
        std::vector<cv::Point2i> points50;
        for (int i = 0; i < anglepoints.size(); i++) {

            int maxpoint = anglepoints[i].x;
            int maxp = i;

            cv::Point2i p2;
            p2.x = maxpoint;
            p2.y = maxp;
            if (anglepoints[i].x > 50)
                points50.push_back(p2);
            continue;


        }

        if (points50.size() < 4)return rst;
        std::sort(points50.begin(), points50.end(), [](cv::Point point1, cv::Point point2) {
            return point1.x > point2.x;
        });

        cv::Point frontp1, frontp, frontp2, behindp1, behindp, behindp2;

        bool fb = false;
        bool bb = false;

        for (int i = 1; i < points50.size(); i++) {
            auto distance = points50[i].y - points50[0].y;
            if (abs(distance) > 200) {
                for (int j = 1; j < points50.size(); j++) {
                    if (abs(points50[j].y - points50[0].y) < 190 && !fb&&abs(points50[j].y - points50[0].y) > 15) {

                        int py = points50[j].y;
                        if (points50[j].y < points50[0].y)  py = py - 20;
                        if (py < 0)py = 0;
                        int bignumber = 0;
                        for (int k = py; k < py + 20 && k < anglepoints.size(); k++) {
                            if (anglepoints[k].x > 50)bignumber++;
                        }
                        if (bignumber < 17) {

                            fb = true;
                            if (distance > 0) {
                                frontp1 = points50[j];
                                frontp2 = points50[0];
                            }
                            if (distance < 0) {
                                behindp1 = points50[j];
                                behindp2 = points50[0];
                            }
                        }










                    }
                    if (abs(points50[j].y - points50[i].y) < 190 && i != j && !bb&&abs(points50[j].y - points50[i].y) > 15) {
                        int py = points50[j].y;
                        if (points50[j].y < points50[i].y)  py = py - 20;
                        if (py < 0)py = 0;
                        int bignumber = 0;
                        for (int k = py; k < py + 20 && k < anglepoints.size(); k++) {
                            if (anglepoints[k].x > 50)bignumber++;
                        }
                        if (bignumber < 17) {




                            bb = true;
                            if (distance > 0) {
                                behindp1 = points50[j];
                                behindp2 = points50[i];
                            }
                            if (distance < 0) {
                                frontp1 = points50[j];
                                frontp2 = points50[i];
                            }
                        }

                    }
                }
                break;
            }


        }






        cv::putText(hist_img, std::to_string(frontp1.y - frontp2.y), cv::Point(20, 30 + 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
        cv::line(hist_img, cv::Point(frontp1.y, 1990 - frontp1.x), cv::Point(frontp2.y, 1990 - frontp2.x), cv::Scalar(255, 0, 0));



        auto duration_capture = clock::now() - last;

        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();

        rst.time = ttime;
        rst.gap[0] = abs(frontp1.y - frontp2.y) * 0.775 / abs(cos(-radian));
        rst.gap[1] = abs(behindp1.y - behindp2.y) * 0.775 / abs(cos(-radian));




#if 0

        cv::Mat playmat = image.clone();
        cv::cvtColor(playmat, playmat, cv::COLOR_GRAY2BGR);
        cv::Point vertices[2][4];
        vertices[0][0].x = 0;
        vertices[0][0].y = frontp1.y - poffset;
        vertices[0][1].x = 2000;
        vertices[0][1].y = frontp1.y - poffset;

        vertices[0][2].x = 0;
        vertices[0][2].y = frontp2.y - poffset;
        vertices[0][3].x = 2000;
        vertices[0][3].y = frontp2.y - poffset;

        vertices[1][0].x = 0;
        vertices[1][0].y = behindp1.y - poffset;
        vertices[1][1].x = 2000;
        vertices[1][1].y = behindp1.y - poffset;

        vertices[1][2].x = 0;
        vertices[1][2].y = behindp2.y - poffset;
        vertices[1][3].x = 2000;
        vertices[1][3].y = behindp2.y - poffset;

        for (int j = 0; j < 4; j++) {
            auto tpoint0 = vertices[0][j];
            auto tpoint1 = vertices[1][j];
            vertices[0][j].x = tpoint0.x*cos(-radian) - tpoint0.y*sin(-radian) + roi.x;
            vertices[0][j].y = tpoint0.x*sin(-radian) + tpoint0.y*cos(-radian) + roi.y;
            vertices[1][j].x = tpoint1.x*cos(-radian) - tpoint1.y*sin(-radian) + roi.x;
            vertices[1][j].y = tpoint1.x*sin(-radian) + tpoint1.y*cos(-radian) + roi.y;
        }
        for (int j = 0; j < 4; j = j + 2) {
            cv::line(playmat, vertices[0][j], vertices[0][(j + 1) % 4], cv::Scalar(0, 255, 0), 1, 16);
            cv::line(playmat, vertices[1][j], vertices[1][(j + 1) % 4], cv::Scalar(0, 255, 0), 1, 16);
        }
        cv::putText(playmat, std::to_string(rst.gap[0]), cv::Point(100, 100 + 30), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        cv::putText(playmat, std::to_string(rst.gap[1]), cv::Point(100, 100 + 230), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        cv::putText(playmat, std::to_string(anglel), cv::Point(100, 100 + 430), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        //cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", playmat);
       // cv::imwrite("G:/datas/test/biaoji/" + std::to_string(numbers_) + ".bmp", playmat);
        cv::imshow("1", playmat);
        cv::waitKey();
        rst.gap_position[0] = vertices[0][1].y;
        rst.gap_position[1] = vertices[1][1].y;
#endif
        return rst;


    }

    int mean_value = static_cast<int>(cv::mean(img2).val[0]);
    unsigned int gray_value = 20;
    gray_value = mean_value / 2;
    if (gray_value > 20)gray_value = 18;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    //cv::adaptiveThreshold(dstImage, dstImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 4);
    //cv::dilate(dstImage, dstImage, element2);
    //morphologyEx(dstImage, dstImage, CV_MOP_OPEN, element, cv::Point(-1, -1), 3);
   // cv::medianBlur(dstImage, dstImage, 15);

    auto img3 = img2.clone();




    cv::threshold(img2, img2, mean_value / 2, 255, CV_THRESH_BINARY_INV);
    morphologyEx(img2, img2, CV_MOP_OPEN, element, cv::Point(-1, -1), 1);
    morphologyEx(img2, img2, CV_MOP_OPEN, element, cv::Point(-1, -1), 2);
    dstImage = img2(cv::Rect(500, 500, 400, 1000)).clone();
    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(dstImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //cv::imwrite("d:/img2.bmp", dstImage);
    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> points1, std::vector<cv::Point>points2)
    {return  points1.size() > points2.size(); });

    int updown = 0;
    cv::RotatedRect rrect[2];
    cv::Point2f vertices[2][4];

    for (int i = 0; i < contours.size(); i++) {

        std::sort(contours[i].begin(), contours[i].end(), [](cv::Point point1, cv::Point point2)
        {return  point1.x < point2.x; });

        auto hrect = cv::boundingRect(contours[i]);

        if (contours[i][0].x < 200 && updown < 2 && hrect.height / hrect.width < 3) {

            rrect[updown] = cv::minAreaRect(contours[i]);

            rrect[updown].points(vertices[updown]);//获取矩形的四个点

            updown++;
        }
        if (updown == 2)break;

    }
    if (updown < 2) {

        contours.clear();
        hierarchy.clear();
        cv::threshold(img3, img3, mean_value / 2, 255, CV_THRESH_BINARY_INV);
        //cv::dilate(img3, img3, element2);
        morphologyEx(img3, img3, CV_MOP_OPEN, element2, cv::Point(-1, -1), 1);
        dstImage = img3(cv::Rect(500, 500, 400, 1000)).clone();


        cv::findContours(dstImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> points1, std::vector<cv::Point>points2)
        {return  points1.size() > points2.size(); });

        int updown = 0;


        for (int i = 0; i < contours.size(); i++) {

            std::sort(contours[i].begin(), contours[i].end(), [](cv::Point point1, cv::Point point2)
            {return  point1.x < point2.x; });

            auto hrect = cv::boundingRect(contours[i]);

            if (contours[i][0].x < 200 && updown < 2 && hrect.height / hrect.width < 3) {

                rrect[updown] = cv::minAreaRect(contours[i]);

                rrect[updown].points(vertices[updown]);

                updown++;
            }
            if (updown == 2)break;

        }

    }





    cv::cvtColor(dstImage, dstImage, CV_GRAY2RGB);



    for (int i = 0; i < 4; i++) {
        cv::line(dstImage, vertices[0][i], vertices[0][(i + 1) % 4], cv::Scalar(0, 255, 0));
        cv::line(dstImage, vertices[1][i], vertices[1][(i + 1) % 4], cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices[0][i].x + roi.x + 500, vertices[0][i].y + roi.y + 500), cv::Point2f(vertices[0][(i + 1) % 4].x + roi.x + 500, vertices[0][(i + 1) % 4].y + roi.y + 500), cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices[1][i].x + roi.x + 500, vertices[1][i].y + roi.y + 500), cv::Point2f(vertices[1][(i + 1) % 4].x + roi.x + 500, vertices[1][(i + 1) % 4].y + roi.y + 500), cv::Scalar(0, 255, 0));


    }
    float h1, h2;
    abs(rrect[0].angle) < 45 ? h1 = rrect[0].size.height : h1 = rrect[0].size.width;
    abs(rrect[1].angle) < 45 ? h2 = rrect[1].size.height : h2 = rrect[1].size.width;

    auto duration_capture = clock::now() - last;

    auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();

    rst.time = ttime;
    rst.gap[0] = h1 * 0.75;
    rst.gap[1] = h2 * 0.75;

    //  cv::imshow("1", dstImage);
      //cv::waitKey();


      //rst.gap[0] = rrect[0].center.x + roi.x + 500;
     // rst.gap[1] = rrect[1].center.x + roi.x + 500;
    rst.gap_position[0] = rrect[0].center.y + roi.y + 500;
    rst.gap_position[1] = rrect[1].center.y + roi.y + 500;
    //rst.width = rrect[0].size.width;
    //rst.height = rrect[0].size.height;
    //rst.parallel = rrect[1].size.width;
    //rst.pointdepth= rrect[1].size.height;
    //rst.abrasion = rrect[0].angle;
    //rst.time=rrect[1].angle;

  // cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);
   // cv::resize(palyimg, palyimg, cv::Size(1024, 1024));



    return rst;



#if 0


    cv::Mat canyimg, img;

    //cv::medianBlur(img, img, 7);
    //cv::Canny(image, img, 30, 60);//125,280
//    cv::resize(canyimg, canyimg, cv::Size(1024, 1024));


    int p1, p2;
    cv::Rect r1;
    // auto himg = img::get_cols_roi(img,!rails, r1);

    int gray_value = static_cast<int>(cv::mean(image).val[0]);//根据整张图片的平均灰度值进行阈值的设定
    cv::Mat  sobel_imgh, sobel_imgw;

    if (image.rows > 3000 && !roi.empty() && image.rows > roi.y + roi.height&&image.cols > roi.x + roi.width) {

        img = image(roi).clone();
    }
    else {
        img = image.clone();
    }

    cv::medianBlur(img, img, 3);
    if (gray_value > 50) cv::blur(img, img, cv::Size(3, 3));//亮度较高的杂波较多使用均值滤波，间隙中反光大，均值去掉部分边缘

    bool railtype = false;
    cv::Canny(img, img, gray_value - 10, (gray_value - 10) * 2);





    if (gray_value > 60) {//亮度高的情况下，白天，会出现间隙孔轮廓，影响直线查找 
        cv::Sobel(img, sobel_imgh, CV_8U, 1, 0, 3, 1, 0);
        cv::threshold(sobel_imgh, sobel_imgh, 250, 255, THRESH_BINARY);

        img = img - sobel_imgh;

    }


    gray_value < 60 ? railtype = false : railtype = true;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::dilate(img, img, element);

    if (image.rows < 2600) {
        auto himg = img::get_cols_roi(img, railtype, roi);
        img = himg;
    }

    // cv::imshow("1", img);
    //cv::waitKey();

















    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);


    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


    std::vector<linepoints> alllp;

    for (int i = 0; i < contours.size(); i++)
    {
        std::sort(contours[i].begin(), contours[i].end(), [](cv::Point &c1, cv::Point &c2) {
            return c1.x < c2.x;//从小到大
        });
    }




    cv::cvtColor(img, img, CV_GRAY2BGR);

    bool is_coincidence = false;
    std::vector<cv::Vec4f> lines;
    std::vector<double> ks;
    for (int i = 0; i < contours.size(); i++)
    {
        auto crect = cv::minAreaRect(contours[i]);
        if (contours[i].size() > 100)//crect.width / crect.height>3&&
        {
            cv::Vec4f line;
            cv::fitLine(contours[i], line, cv::DIST_L2, 0, 1e-2, 1e-2);


            double k = line[1] / line[0];



            double distance = 0.0;

            int maxdistancex = 0;
            std::vector<cv::Point> contour;
            //distance = fabs(k*(tpoints[0].x - line[2]) + line[3] - tpoints[0].y);
            for (int j = 0; j < contours[i].size(); j++) {
                double tempdistance = fabs(k*(contours[i][j].x - line[2]) + line[3] - contours[i][j].y);
                if (tempdistance > distance) {
                    distance = tempdistance;
                    maxdistancex = i;
                }
                if (tempdistance < 1) {//精确取值
                    contour.push_back(contours[i][j]);
                }
            }
            double kk = atan(k)*180.0 / 3.14159265;
            auto twidth = crect.size.width;
            auto theight = crect.size.height;

            kk > 0 ? crect.size.width = theight : crect.size.width = twidth;
            kk > 0 ? crect.size.height = twidth : crect.size.height = theight;
            cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0));
            if ((crect.size.width / crect.size.height > 5) && crect.size.width > 80 && abs(kk) > 8 && abs(kk) < 60 && crect.size.height < 50)//&&abs(kk)>10&& distance<3)//kk>10&&kk<60&& crect.width>100)
            {

                std::vector<cv::Point> contour;

                if (distance > 4) {

                    auto pnumber = contours[i].size() / 2;
                    maxdistancex < pnumber ? contour.assign(contours[i].begin() + pnumber, contours[i].end()) : contour.assign(contours[i].begin(), contours[i].begin() + pnumber);

                    cv::fitLine(contour, line, cv::DIST_L2, 0, 1e-2, 1e-2);
                }


                linepoints onelp;

                contour.size() > 0 ? onelp.points = contour : onelp.points = contours[i];
                onelp.lines = line;


                cv::Point2f point1, point2;
                point2.x = line[2];
                point2.y = line[3];
                double k = line[1] / line[0];
                point1.x = 0.0;
                point1.y = k * (0 - line[2]) + line[3];
                line[2] = point1.x;
                line[3] = point1.y;
                lines.push_back(line);

                alllp.push_back(onelp);

                // cv::line(img, point1, point2, cv::Scalar(0, 0, 255));

                double angle = atan(k)*180.0 / 3.14159265;


                ks.push_back(angle);







            }

        }

    }


    std::vector<int> positives;
    std::vector<int> negatives;

    for (int i = 0; i < ks.size(); i++) {
        if (ks[i] >= 0)positives.push_back(i);
        if (ks[i] < 0)negatives.push_back(i);
    }

    std::vector<linepoints> talllp;
    std::vector<cv::Vec4f>tlines;

    if (positives.size() > negatives.size() && negatives.size() > 0) {

        for (int i = 0; i < alllp.size(); i++) {
            bool tn = false;
            for (int j = 0; j < negatives.size(); j++) {
                if (negatives[j] == i) {
                    tn = true;
                    break;
                }


            }
            if (!tn) {
                talllp.push_back(alllp[i]);
                tlines.push_back(lines[i]);
            }
        }
        alllp = talllp;
        lines = tlines;


    }

    if (positives.size() < negatives.size() && positives.size() > 0) {
        for (int i = 0; i < alllp.size(); i++) {
            bool tn = false;
            for (int j = 0; j < positives.size(); j++) {
                if (positives[j] == i) {
                    tn = true;
                    break;
                }


            }
            if (!tn) {
                talllp.push_back(alllp[i]);
                tlines.push_back(lines[i]);
            }
        }


        alllp = talllp;
        lines = tlines;


    }
    if (image.rows > 3000) {


        std::vector<linepoints>::iterator itr = alllp.begin();
        std::vector<cv::Vec4f>::iterator itr2 = lines.begin();
        while (itr != alllp.end())
        {
            auto sum = std::accumulate(itr->points.begin(), itr->points.end(), 0, [](int  a, cv::Point c2) {return a + c2.y; });
            int mean = sum / itr->points.size();
            if (abs(mean - (img.rows / 2)) > 400) {

                itr = alllp.erase(itr);
                itr2 = lines.erase(itr2);
            }
            else {
                itr++;
                itr2++;
            }

        }
    }




    cv::Vec4f line;
    line[2] = 0;
    std::vector<double>distances;
    //for (int i = 0; i < lines.size(); i++) {
    //    if (line[2] == 0) { 
    //        line = lines[i];
    //        continue;
    //    }
    // //   double tempdistance = fabs(k*(tpoints[i].x - line[2]) + line[3] - tpoints[i].y);
    //    double tempdistance = fabs((lines[i][1] / lines[i][0]) * (line[2] - lines[i][2]) + lines[i][3] - line[3]);
    //    double tempdistance2 = fabs((line[1] / line[0]) * (lines[i][2] - line[2]) + line[3] - lines[i][3]);
    //    if (tempdistance > 2) {
    //        

    //        line[2] = 0;
    //        distances.push_back(tempdistance);
    //    }
    //    else { continue; }




    //}
    std::sort(lines.begin(), lines.end(), [](cv::Vec4f  &c1, cv::Vec4f &c2) {
        return c1[3] < c2[3];//从小到大
    });

    std::sort(alllp.begin(), alllp.end(), [](linepoints  &c1, linepoints &c2) {
        return c1.lines[3] < c2.lines[3];//从小到大
    });

    if (alllp.size() < 1)return rst;
    auto middley = (alllp.begin()->lines[3] + alllp.rbegin()->lines[3]) / 2;
    std::vector<cv::Point> points[2];
    std::vector< cv::Vec4f> vlines[2];
    for (int i = 0; i < alllp.size(); i++) {
        if (alllp[i].lines[3] > middley) {
            for (int j = 0; j < alllp[i].points.size(); j++) {
                points[1].push_back(alllp[i].points[j]);//35.bmp异常

            }
            vlines[1].push_back(alllp[i].lines);

        }
        else {
            for (int j = 0; j < alllp[i].points.size(); j++) {
                points[0].push_back(alllp[i].points[j]);

            }
            vlines[0].push_back(alllp[i].lines);
        }
    }

    if (points[1].size() < 1)return rst;
    auto rrc0 = cv::minAreaRect(points[0]);
    auto rrc1 = cv::minAreaRect(points[1]);
    cv::Point2f vertices[4], vertices1[4];
    rrc0.points(vertices);//获取矩形的四个点
    rrc1.points(vertices1);//获取矩形的四个点
    for (int i = 0; i < 4; i++) {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 0));
        cv::line(img, vertices1[i], vertices1[(i + 1) % 4], cv::Scalar(0, 255, 255));
        cv::line(palyimg, cv::Point2f(vertices[i].x + roi.x, vertices[i].y + roi.y), cv::Point2f(vertices[(i + 1) % 4].x + roi.x, vertices[(i + 1) % 4].y + roi.y), cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices1[i].x + roi.x, vertices1[i].y + roi.y), cv::Point2f(vertices1[(i + 1) % 4].x + roi.x, vertices1[(i + 1) % 4].y + roi.y), cv::Scalar(0, 255, 0));


    }

    //cv::imshow("1", palyimg);
    //cv::waitKey();

    if (lines.size() < 1)return rst;

    for (int i = 0; i < lines.size() - 1; i++) {
        double k = lines[i][1] / lines[i][0];
        double kk = atan(k)*180.0 / 3.14159265;
        double tempdistance = fabs((lines[i][1] / lines[i][0]) * (lines[i + 1][2] - lines[i][2]) + lines[i][3] - lines[i + 1][3]);

        //cv::putText(img, std::to_string(i), cv::Point(20*i, lines[i][3]), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

        ks.push_back(kk);

        distances.push_back(tempdistance);





    }

    float h1, h2;
    abs(rrc0.angle) < 45 ? h1 = rrc0.size.height : h1 = rrc0.size.width;
    abs(rrc1.angle) < 45 ? h2 = rrc1.size.height : h2 = rrc1.size.width;

    float hh1, hh2, hh11, hh22;

    hh1 = fabs(vlines[0][0][3] - vlines[0][vlines[0].size() - 1][3]) / sqrt(1 + pow(vlines[0][0][1] / vlines[0][0][0], 2));
    hh11 = fabs(vlines[0][0][3] - vlines[0][vlines[0].size() - 1][3]) / sqrt(1 + pow(vlines[0][vlines[0].size() - 1][1] / vlines[0][vlines[0].size() - 1][0], 2));
    hh2 = fabs(vlines[1][0][3] - vlines[1][vlines[1].size() - 1][3]) / sqrt(1 + pow(vlines[1][0][1] / vlines[1][0][0], 2));
    hh22 = fabs(vlines[1][0][3] - vlines[1][vlines[1].size() - 1][3]) / sqrt(1 + pow(vlines[1][vlines[1].size() - 1][1] / vlines[1][vlines[1].size() - 1][0], 2));

    float vk1 = vlines[0][vlines[0].size() - 1][1] / vlines[0][vlines[0].size() - 1][0];
    float vk2 = vlines[1][vlines[1].size() - 1][1] / vlines[1][vlines[1].size() - 1][0];

    hh11 = fabs(vk1* (points[0][0].x - vlines[0][vlines[0].size() - 1][2]) + vlines[0][vlines[0].size() - 1][3] - points[0][0].y) / sqrt(1 + pow(vk1, 2));
    hh22 = fabs(vk2* (points[1][0].x - vlines[1][vlines[1].size() - 1][2]) + vlines[1][vlines[1].size() - 1][3] - points[1][0].y) / sqrt(1 + pow(vk2, 2));

    double vangel = atan(vk1)*180.0 / 3.14159265;
    double vange2 = atan(vk1)*180.0 / 3.14159265;

    float  difference[2] = { 0 };
    vangel*rrc0.angle < 0 ? difference[0] = abs(abs(vangel - rrc0.angle) - 90) : difference[0] = abs(vangel - rrc0.angle);
    vange2*rrc1.angle < 0 ? difference[1] = abs(abs(vange2 - rrc1.angle) - 90) : difference[1] = abs(vange2 - rrc1.angle);


    if (difference[0] > 10) h1 = hh11;
    if (difference[1] > 10) h2 = hh22;
    //cv::resize(img, img, cv::Size(1024, 900));
    auto duration_capture = clock::now() - last;

    auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
    rst.time = ttime;
    rst.gap[0] = h1 * 0.75;
    rst.gap[1] = h2 * 0.75;
    rst.gap_position[0] = rrc0.center.y;
    rst.gap_position[1] = rrc1.center.y;
    cv::resize(img, img, cv::Size(img.cols, 1024));
    cv::resize(palyimg, palyimg, cv::Size(1024, 1024));
    /* cv::imshow("1", palyimg);

     cv::waitKey();*/
    cv::putText(img, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
    cv::putText(img, std::to_string(h1), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(h2), cv::Point(20, 110), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(rst.gap_position[0]), cv::Point(20, 150), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(rst.gap_position[1]), cv::Point(20, 190), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));



    cv::putText(palyimg, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
    cv::putText(palyimg, std::to_string(rst.gap[0]), cv::Point(20, 310), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap[1]), cv::Point(20, 350), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap_position[0]), cv::Point(20, 150), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap_position[1]), cv::Point(20, 190), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.shield_value[0]), cv::Point(20, 230), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.shield_value[1]), cv::Point(20, 270), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    //cv::putText(img, std::to_string(hh11), cv::Point(20, 230), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    //cv::putText(img, std::to_string(hh22), cv::Point(20, 270), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

    for (int i = 0; i < distances.size(); i++) {

        // cv::putText(img, std::to_string(i)+"==" + std::to_string(distances[i]), cv::Point(20, 70+i*40), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 255, 0));
         //cv::putText(img, std::to_string(i) + "==" + std::to_string(ks[i]), cv::Point(250, 70 + i * 40), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 255, 0));
    }


    //cv::imwrite("G:/datas/test/"+std::to_string((clock::now().time_since_epoch()).count())+".bmp", palyimg);
    cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);




    // cv::imshow("img", canyimg);
     //cv::waitKey();


    //img.empty() ? rst.imgtype = feature::none : rst.imgtype = feature::endbend;
    return rst;
#endif
}
result data_process::process_shield_img(unsigned char  *datas, cv::Rect &roi, cv::Mat &playimg, int size, int rails) {
    //1、查找定位点可以将图片y膨胀，消除中间锯齿状2、剪切时掐头去尾10像素点，去掉末端黏连，保证连通域内外分开
    numbers_++;
    result rst;
    if (size < 2048)
        return rst;
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();
    auto image = cv::Mat::Mat(size / 2048, 2048, CV_8UC1, datas);







#if CKAI

    if (!is_ysonnx_)return rst;
    std::vector<OutputSeg> output;
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    bool find = yvonnx_.OnnxDetect(image, output);
    std::vector<cv::Scalar> color;

    cv::Rect rects[2];

    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(0, 0, 255));
    }

    int size_gap = 0;
    if (find) {


        for (int k = 0; k < output.size(); k++) {
            if (output[k].id != 1)continue;


            if (size_gap < 2) {
                rects[size_gap] = output[k].box;
                size_gap++;
            }


        }







        auto duration_capture1 = clock::now() - last;



        float h1, h2;
        h1 = rects[0].height;
        h2 = rects[1].height;



        if (rects[0].y < rects[1].y) {
            auto th = h1;
            h1 = h2;
            h2 = th;

        }






        rst.shield_value[0] = h1 * 0.775;
        rst.shield_value[1] = h2 * 0.775;
        rst.shield_position[0] = rects[0].y;
        rst.shield_position[1] = rects[1].y;
        rst.shield = rects[0].x;

        auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
        rst.time = timeuse;
        std::string label = "TimeUse: " + std::to_string(timeuse) + "=" + std::to_string(rst.gap[0]) + "=" + std::to_string(rst.gap[1]);
        //DrawPred(image, output, yvonnx_._className, color);
        cv::rectangle(image, rects[0], cv::Scalar(0, 255, 0));
        cv::rectangle(image, rects[1], cv::Scalar(0, 255, 0));
        cv::putText(image, label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2, 8);
        cv::resize(image, image, cv::Size(1024, 1024));
        cv::imshow("result", image);
        cv::waitKey();

        return  rst;
    }

    else

        return  rst;








#endif




















    cv::Mat img, img2;
    img = image(cv::Rect(150, 0, 700, image.rows)).clone();
    img2 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();
    auto img7 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();
    playimg = image.clone();
    int miny, maxy, minx;
    miny = 0;
    maxy = 0;
    minx = 0;
    cv::medianBlur(img, img, 5);
    cv::threshold(img, img, 20, 255, cv::THRESH_BINARY);
    //show_mat(ui.procesed_label, img);
    auto kk = location_all_yy(img, minx, miny, maxy);
    if (maxy - miny > 1 && maxy - miny < img2.rows&&miny>0 && minx < img2.cols - 150) {
        int widthc = 200;
        int startx = 0;
        startx = minx + 150;

        image.rows > 3000 ? cv::medianBlur(img2, img2, 15) : cv::medianBlur(img2, img2, 3);


        int mean_value = static_cast<int>(cv::mean(img2).val[0]);
        unsigned int gray_value = 20;
        gray_value = mean_value / 2;
        if (gray_value > 20)gray_value = 18;
        //if (image.rows < 3000)gray_value = gray_value / 2;

       // cv::Canny(img2, img2, gray_value, gray_value * 2);
       // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
       // cv::dilate(img2, img2, element);


        img2.cols > widthc + startx ? widthc = 200 : widthc = img2.cols - startx;
        auto img4 = img2(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定
        int maxh, minh;






        auto img6 = img7(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定
        cv::medianBlur(img6, img6, 7);


        cv::Sobel(img6, img6, CV_8U, 1, 0, 5, 1, 0);
        //cv::imwrite("d:/img6c.bmp", img6);
        cv::threshold(img6, img6, 100, 255, cv::THRESH_BINARY);
        //img::threshold_bound(img6, img6, 100, 255);
    /*	cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
        cv::erode(img6, img6, element2);*/
        //cv::imwrite("d:/img7.bmp", img7);
        //cv::imwrite("d:/8c.bmp", img6);
        std::vector<std::vector<cv::Point>>contours;
        std::vector<cv::Vec4i>hierarchy;
        cv::findContours(img6, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
            return c1.size() > c2.size();
            //return cv::contourArea(c1) > cv::contourArea(c2); 
        });
        minh = -1;
        maxh = img6.rows;

        bool minb = false;
        bool maxb = false;



        int lastminx = 0;
        int lastmaxx = 0;
        for (int i = 0; i < contours.size(); i++) {
            auto rdrect = cv::boundingRect(contours[i]);
            auto rdarea = cv::contourArea(contours[i]);
            if (rdrect.height > 400)continue;

            if (rdrect.x > 5 && rdrect.x < 110 && rdrect.width < rdrect.height&&rdarea>300 && rdrect.width>12) {
                if (rdrect.y < img6.rows / 2 && rdrect.y>minh&&rdrect.x - lastminx > 0 && rdrect.y + rdrect.height < 320) {
                    minh = rdrect.y + rdrect.height;
                    lastminx = rdrect.x;
                    //lastminy = rdrect.y + rdrect.height;
                    minb = true;
                }
                if (rdrect.y > img6.rows / 2 && rdrect.y < maxh&&rdrect.x - lastmaxx>0 && img6.rows - rdrect.y < 320) {
                    lastmaxx = rdrect.x;
                    maxh = rdrect.y;
                    maxb = true;
                }


            }


        }

        for (int i = 0; i < contours.size() && i < 10; i++) {

            auto rdrect = cv::boundingRect(contours[i]);
            auto rdarea = cv::contourArea(contours[i]);
            if (rdrect.height > 400)continue;
            if (rdrect.x > 5 && rdrect.x < 110 && rdrect.width < rdrect.height&& rdrect.y + rdrect.height < 320) {
                if (rdrect.y < img6.rows / 2 && !minb&&rdrect.y < 5) {
                    minh = rdrect.y + rdrect.height;
                    minb = true;
                }
                if (rdrect.y > img6.rows / 2 && img6.rows - rdrect.y - rdrect.height < 5 && !maxb&& img6.rows - rdrect.y < 320) {
                    maxh = rdrect.y;
                    maxb = true;
                }


            }


        }
        //cv::cvtColor(img6, img6, CV_GRAY2BGR);







       // location_in_yy(img4, minh, maxh);


        //playimg = img2.clone();

        cv::cvtColor(playimg, playimg, CV_GRAY2BGR);
        cv::rectangle(playimg, cv::Rect(startx, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(playimg, cv::Rect(startx, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));
        cv::putText(playimg, std::to_string((minh + 10)*0.775), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        cv::putText(playimg, std::to_string((maxy - miny - maxh + 10)*0.775), cv::Point(20, 120), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        auto  dstImage = img4.clone();
        cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
        cv::rectangle(dstImage, cv::Rect(0, 0, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(dstImage, cv::Rect(0, maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string((minh + 10)*0.775), cv::Point(130, 70), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string((maxy - miny - maxh + 10)*0.775), cv::Point(130, 120), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string(numbers_), cv::Point(130, 170), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));

        auto duration_capture = clock::now() - last;
        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        //cv::imwrite("E:/datas/binary/" + std::to_string(numbers_) + ".bmp", dstImage);

        //cv::resize(playimg, playimg, cv::Size(1024, 1024));
        //cv::imshow("1", dstImage);
       // cv::waitKey();


        rst.time = ttime;
        rst.shield_value[0] = abs(minh + 10)*0.775;
        rst.shield_value[1] = abs(maxy - miny - maxh + 10) * 0.775;

        rst.shield_position[0] = miny;
        rst.shield_position[1] = miny + maxh;



        rst.shield = startx;
        rst.width = img4.cols;

        roi.x = minx + 500;
        roi.y = miny;
        roi.width = 1000;
        roi.height = maxy - miny;

        if (!kk) {
            roi.x = 400;
            roi.width = image.cols - 400;//600
            roi.y = 0;
            roi.height = image.rows;
        }
        roi.y = 0;
        roi.height = image.rows;

    }

    return rst;




























#if 0
    if (image.rows > 3000) {
        cv::blur(image, image, cv::Size(3, 3));
    }

    cv::Mat img, img2, canyimg;
    cv::Mat dstImage; //目标图
    cv::Mat normImage; //归一化后的图
    cv::Mat scaledImage; //线性变换后的八位无符号整形的图

    img = image.clone();
    img2 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();



    canyimg = img2.clone();
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    playimg = image;
    /*  cv::imshow("1", img2);
      cv::waitKey();*/
      //cv::Mat imageGamma;
      ////灰度归一化
      //image_.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);
      ////伽马变换
      //double gamma = 0.7;
      //pow(imageGamma, gamma, img2);//dist 要与imageGamma有相同的数据类型
      //img2.convertTo(img2, CV_8U, 255, 0);
      //show_mat(ui.procesed_label, img2);
      //cv::imshow("1", img2);
      //cv::waitKey();
      /*cv::Rect r1;
      auto himg = img::get_cols_roi(img2, false, r1);
      img2 = himg;*/

      //cv::blur(img2, img2, cv::Size(3, 3));

      /*cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 6));
      cv::erode(img2, img2, element2);*/


    int mean_value = static_cast<int>(cv::mean(img2).val[0]);
    unsigned int gray_value = 20;
    gray_value = mean_value / 2;
    if (gray_value > 20)gray_value = 18;
    cv::Canny(img2, img2, gray_value, gray_value * 2);

    //cv::Canny(canyimg, canyimg, gray_value, gray_value * 2);
 /*   cv::imshow("canyimg", canyimg);
    cv::waitKey();
    cv::imshow("1", img2);
    cv::waitKey();*/

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 15));
    cv::dilate(img2, img2, element);






    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(img2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
        return c1[0].x < c2[0].x;
        //return cv::contourArea(c1) > cv::contourArea(c2); 
    });
    cv::Rect shieldrect;
    for (int i = 0; i < contours.size(); i++) {
        shieldrect = cv::boundingRect(contours[i]);
        if (shieldrect.width > 50 && shieldrect.height > 500 && shieldrect.x > 100)
            break;
    }
    //

    if (shieldrect.x > 500) {//端部弯头处
        cv::Canny(canyimg, canyimg, gray_value / 2, gray_value);
        cv::dilate(canyimg, canyimg, element);
        img2 = canyimg.clone();


        contours.clear();
        hierarchy.clear();
        cv::findContours(img2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
            return c1[0].x < c2[0].x;
            //return cv::contourArea(c1) > cv::contourArea(c2); 
        });

        for (int i = 0; i < contours.size(); i++) {
            shieldrect = cv::boundingRect(contours[i]);
            if (shieldrect.width > 50 && shieldrect.height > 800 && shieldrect.x > 90)
                break;
        }
    }


    shieldrect.y = 0;
    shieldrect.height = img2.rows;
    auto twidth = shieldrect.width;
    shieldrect.width = 250;
    if (shieldrect.x + shieldrect.width > img2.cols)
        shieldrect.width = twidth;

    auto img3 = img2(shieldrect).clone();

    /* cv::imshow("1", img3);
     cv::waitKey();
 */
    dstImage = img2.clone();
    cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
    cv::drawContours(dstImage, contours, 0, cv::Scalar(0, 255, 0));
    std::vector<cv::Point> shieldpoints;

    for (int y = 0; y < img3.rows; y++) {
        int x;
        const  uchar* dptr = img3.ptr<uchar>(y);

        for (x = 0; x < img3.cols; x++) {
            if (dptr[x] > 0) {
                shieldpoints.push_back(cv::Point(x, y));
                //circle(dstImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                break;
            }
        }
    }



    std::vector<cv::Point3f> pkk;//先获取各点与前两点斜率，然后进行抑制排序，最终选取两点
    if (shieldpoints.size() < 6)return rst;

    for (int i = 0; i < shieldpoints.size() - 6; i++) {
        std::vector<cv::Point> ttpoints;
        ttpoints.assign(shieldpoints.begin() + i, shieldpoints.begin() + i + 5);
        cv::Vec4f line;
        // double sx = shieldpoints[i + 5].x - shieldpoints[i].x;
        // double sy = shieldpoints[i + 5].y - shieldpoints[i].y;
        cv::fitLine(ttpoints, line, cv::DIST_L2, 0, 1e-2, 1e-2);


        double kk = line[1] / line[0];
        cv::Point3f pf;
        pf.x = shieldpoints[i].x;
        pf.y = shieldpoints[i].y;
        pf.z = abs(kk);
        pkk.push_back(pf);


    }
    if (pkk.size() < 6)return rst;
    std::vector<cv::Point3f> pkks_nms;//y距离超过5就认为不在范围内，小于3取最小值
    std::vector<cv::Point3f> pkks_nms1;//

    for (int i = 0; i < pkk.size(); i++) {
        cv::Point3f pf;
        pf = pkk[i];
        int addj = 0;
        int addj2 = 0;
        for (int j = i + 1; j < pkk.size(); j++) {
            if ((pkk[j].y - pf.y) < 100) {
                if (pkk[j].z < pf.z) {
                    pf = pkk[j];
                    addj = j - i;
                }
                addj2 = j - i;
            }

            if (pkk[j].y - pf.y >= 100) {
                break;
            }
        }
        if (pf.z < 1) {
            //pf.z = i + addj;
            pkks_nms1.push_back(pf);

        }
        if (pf.z < 0.6) {

            pf.z = i + addj;
            pkks_nms.push_back(pf);

            circle(dstImage, cv::Point(pf.x, pf.y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
        }


        i = addj2 + i;


    }
    if (pkks_nms.size() < 2)return rst;
    std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.y < pf2.y;
    });

    float miny = 0;
    float maxy = 0;
    float minx, maxx;
    minx = 0;
    maxx = minx;
    bool findok = false;//是否膨胀接头
    float lhdistance = 9999.0;
    int smallline = -9999;
    std::vector<cv::Point2f> lhpoints;//符合条件的所有点
    if (image.rows > 3000) {
        for (int i = 0; i < pkks_nms.size(); i++) {
            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;
            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }


                int endpoint = posekk + 5;
                if (pkk.size() < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0; k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;

                if (pkks_nms[j].y - pkks_nms[i].y < 2000 && pkks_nms[j].y - pkks_nms[i].y>1900) {


                    findok = true;
                    //auto tdistance = abs((pkks_nms[j].y + pkks_nms[i].y - img3.rows) / 2);
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;
                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z; k++) {
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;
                        if (pkk[k].x == 0)zeropoint = true;
                    }

                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0; k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - 1920);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {//先或后与，无论满足哪一条件即可，中位或者大小。但如果两者都满足则优先选择此点
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;

                    }
                    circle(dstImage, cv::Point(pkks_nms[i].x, pkks_nms[i].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    circle(dstImage, cv::Point(pkks_nms[j].x, pkks_nms[j].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;
                    lhpoints.push_back(lhyp);



                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 2000)break;
            }
            //if (findok)break;
        }
    }
    else {
        int yrange = 1000;
        if (rails < 2)yrange = 700;



        for (int i = 0; i < pkks_nms.size() - 1; i++) {

            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;

            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }

                int endpoint = posekk + 5;
                if (pkk.size() - 1 < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0; k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;






                if (pkks_nms[j].y - pkks_nms[i].y < 1200 && pkks_nms[j].y - pkks_nms[i].y>600) {//半包嵌700、全包1000
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;

                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z; k++) {
                        if (pkk[k].x == 0)zeropoint = true;
                        if (pkk[k].x > pkks_nms[i].x)break;
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;

                    }

                    findok = true;
                    /***进一步确定搭接量位置，找到符合条件的上点***/


                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0; k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    //tdistance = matchs;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - yrange);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;

                    }
                    circle(dstImage, cv::Point(pkks_nms[i].x, pkks_nms[i].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    circle(dstImage, cv::Point(pkks_nms[j].x, pkks_nms[j].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;
                    lhpoints.push_back(lhyp);



                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 1200)break;
            }
            //if (findok)break;
        }
    }






    std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.z < pf2.z;
    });

    std::sort(pkks_nms1.begin(), pkks_nms1.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.z < pf2.z;
    });



    if ((pkks_nms.size() < 2 && pkks_nms1.size() > 1) || (!findok&&pkks_nms1.size() > 1)) {

        circle(dstImage, cv::Point(pkks_nms1[0].x, pkks_nms1[0].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
        circle(dstImage, cv::Point(pkks_nms1[1].x, pkks_nms1[1].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);


        if (pkks_nms1[0].y > pkks_nms1[1].y) {
            miny = pkks_nms1[1].y;
            maxy = pkks_nms1[0].y;
        }
        else {
            miny = pkks_nms1[0].y;
            maxy = pkks_nms1[1].y;
        }
    }



    if (maxy - miny > 1) {
        int widthc = 250;
        int startx = 9999;

        //img3= canyimg(shieldrect).clone()+img3;
        if (minx != maxx) {

            for (int i = 0; i < shieldpoints.size(); i++) {
                if (shieldpoints[i].y > maxy)break;
                if (shieldpoints[i].y > miny - 1 && startx > shieldpoints[i].x) {
                    startx = shieldpoints[i].x;

                }
            }


        }
        else { startx = 0; }

        startx = shieldrect.x + startx;






        img2.cols > widthc + startx ? widthc = 250 : widthc = img2.cols - startx;
        auto img4 = img2(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定


        int maxh, minh;
        img::get_shield_rows(img4, maxh, minh);

        cv::rectangle(dstImage, cv::Rect(shieldrect.x, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(dstImage, cv::Rect(shieldrect.x, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));




        auto duration_capture = clock::now() - last;

        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        rst.time = ttime;
        rst.shield_value[0] = abs(minh - 15)*0.75;
        rst.shield_value[1] = abs(maxy - miny - maxh - 15) * 0.75;
        rst.shield_position[0] = miny;
        rst.shield_position[1] = miny + maxh;

        cv::rectangle(image, cv::Rect(shieldrect.x, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(image, cv::Rect(shieldrect.x, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));

        rst.shield = shieldrect.x;
        rst.width = img4.cols;

        roi.x = shieldrect.x + 400;
        roi.y = miny;
        roi.width = 800;
        roi.height = maxy - miny;

        //cv::resize(image, image, cv::Size(1024, 1024));

        cv::putText(image, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
        cv::putText(image, std::to_string(rst.shield_position[0]), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        cv::putText(image, std::to_string(rst.shield_position[1]), cv::Point(20, 110), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

        imshow("1", image);
        cv::waitKey();






    }




    //circle(img2, cv::Point(pkks_nms[0].x, pkks_nms[0].y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);
    //circle(img2, cv::Point(pkks_nms[1].x, pkks_nms[1].y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);


    playimg = image;


    //cv::waitKey(2);
   // cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", image);
    return rst;



#endif



}
bool data_process::location_yy(const std::vector<cv::Point3f> &pkks_nms, const std::vector<cv::Point3f> &pkk, const int &rows, int&miny, int&maxy) {
    float minx, maxx;
    minx = 0;
    maxx = minx;
    bool findok = false;//是否膨胀接头
    float lhdistance = 9999.0;
    int smallline = -9999;
    if (pkks_nms.size() < 2 || pkk.size() < 2)return findok;

    if (rows > 3000) {
        for (int i = 0; i < pkks_nms.size(); i++) {
            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;
            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }


                int endpoint = posekk + 5;
                if (pkk.size() < posekk + 6)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0 && k < pkk.size(); k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;

                if (pkks_nms[j].y - pkks_nms[i].y < 2000 && pkks_nms[j].y - pkks_nms[i].y>1900) {



                    //auto tdistance = abs((pkks_nms[j].y + pkks_nms[i].y - img3.rows) / 2);
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;
                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z&& k < pkk.size(); k++) {
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;
                        if (pkk[k].x == 0)zeropoint = true;
                    }

                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0 && k < pkk.size(); k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - 1920);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        findok = true;
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;


                    }

                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;




                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 2000)break;
            }
            //if (findok)break;
        }
    }
    else {
        int yrange = 1000;
        if (2)yrange = 700;



        for (int i = 0; i < pkks_nms.size() - 1; i++) {

            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;

            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }

                int endpoint = posekk + 5;
                if (pkk.size() - 1 < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0 && k < pkk.size(); k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;






                if (pkks_nms[j].y - pkks_nms[i].y < 1200 && pkks_nms[j].y - pkks_nms[i].y>600) {//半包嵌700、全包1000
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;

                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z&&k < pkk.size(); k++) {
                        if (pkk[k].x == 0)zeropoint = true;
                        if (pkk[k].x > pkks_nms[i].x)break;
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;

                    }


                    /***进一步确定搭接量位置，找到符合条件的上点***/


                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0 && k < pkk.size(); k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    //tdistance = matchs;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - yrange);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        findok = true;
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;


                    }

                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;




                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 1200)break;
            }
            //if (findok)break;
        }
    }

    return findok;

}
bool data_process::location_all_yy(const cv::Mat &image, int &minx, int&miny, int&maxy) {
    auto img = image.clone();
    auto  dstImage = image.clone();
    cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
    cv::threshold(img, img, 20, 255, cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    cv::dilate(img, img, element);
    //cv::imshow("1", img);
    //cv::waitKey();

    std::vector<cv::Point> shieldpoints;
    minx = 0;
    for (int y = 0; y < img.rows; y++) {
        int x;
        const  uchar* dptr = img.ptr<uchar>(y);

        for (x = 0; x < img.cols; x++) {
            if (dptr[x] > 0) {
                shieldpoints.push_back(cv::Point(x, y));
                if (minx < x) {
                    minx = x;
                }
                //circle(dstImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                break;
            }
        }
    }



    std::vector<cv::Point3f> pkk;//先获取各点与前两点斜率，然后进行抑制排序，最终选取两点
    if (shieldpoints.size() < 6)return false;

    for (int i = 0; i < shieldpoints.size() - 6; i++) {
        std::vector<cv::Point> ttpoints;
        ttpoints.assign(shieldpoints.begin() + i, shieldpoints.begin() + i + 5);
        cv::Vec4f line;
        // double sx = shieldpoints[i + 5].x - shieldpoints[i].x;
        // double sy = shieldpoints[i + 5].y - shieldpoints[i].y;
        cv::fitLine(ttpoints, line, cv::DIST_L2, 0, 1e-2, 1e-2);


        double kk = line[1] / line[0];
        cv::Point3f pf;
        pf.x = shieldpoints[i].x;
        pf.y = shieldpoints[i].y;
        pf.z = abs(kk);
        pkk.push_back(pf);


    }
    if (pkk.size() < 6)return false;
    std::vector<cv::Point3f> pkks_nms;//y距离超过5就认为不在范围内，小于3取最小值
    std::vector<cv::Point3f> pkks_nms1;//

    for (int i = 0; i < pkk.size(); i++) {
        cv::Point3f pf;
        pf = pkk[i];
        int addj = 0;
        int addj2 = 0;
        for (int j = i + 1; j < pkk.size(); j++) {
            if ((pkk[j].y - pf.y) < 100) {
                if (pkk[j].z < pf.z) {
                    pf = pkk[j];
                    addj = j - i;
                }
                addj2 = j - i;
            }

            if (pkk[j].y - pf.y >= 100) {
                break;
            }
        }
        if (pf.z < 2) {
            //if (pf.z > 0.6)
               // circle(dstImage, cv::Point(120, pf.y), 10, cv::Scalar(255, 255, 0), 2, 8, 0);

            cv::Point3f tpf;
            tpf = pf;
            tpf.z = i + addj;
            pkks_nms1.push_back(tpf);




        }
        if (pf.z < 0.6) {

            pf.z = i + addj;
            pkks_nms.push_back(pf);
            //circle(dstImage, cv::Point(120, pf.y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);

        }


        i = addj2 + i;


    }
    if (pkks_nms1.size() < 2)return false;
    if (pkks_nms.size() > 1) {
        std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
            return pf1.y < pf2.y;
        });
    }


    bool is_location = location_yy(pkks_nms, pkk, img.rows, miny, maxy);

    if (pkks_nms.size() == 2 && !is_location) {
        if (img.rows > 3000 && (pkks_nms[1].y - pkks_nms[0].y < 2000 && pkks_nms[1].y - pkks_nms[0].y>1900)) {

            maxy = pkks_nms[1].y;
            miny = pkks_nms[0].y;
            is_location = true;

        }
        if (img.rows < 3000 && (pkks_nms[1].y - pkks_nms[0].y < 1000 && pkks_nms[1].y - pkks_nms[0].y>600)) {
            maxy = pkks_nms[1].y;
            miny = pkks_nms[0].y;
            is_location = true;
        }

    }


    if (!is_location)is_location = location_yy(pkks_nms1, pkk, img.rows, miny, maxy);



    for (int i = 0; i < shieldpoints.size(); i++) {
        if (shieldpoints[i].y > maxy)break;
        if (shieldpoints[i].y > miny - 1 && minx > shieldpoints[i].x) {
            minx = shieldpoints[i].x;

        }
    }


    //circle(dstImage, cv::Point(150, miny), 20, cv::Scalar(0, 255, 0), 2, 8, 0);
    //circle(dstImage, cv::Point(150, maxy), 20, cv::Scalar(0, 255, 0), 2, 8, 0);

    miny = miny + 2 + 10;
    maxy = maxy - 2 - 10;

    return is_location;
}
void data_process::location_in_yy(const cv::Mat &image, int&minh, int&maxh) {
    int threshhold = 30;
    int numberpoints = 80;
    int endnumber = 1;
    if (image.rows > 1500) {//膨胀接头
        threshhold = 20;
        numberpoints = 20;
        endnumber = 20;
    }


    int minnumber = 9999;
    int maxnumber = -9999;
    int minposition = image.rows / 2;
    std::vector<cv::Point> points;
    std::vector<cv::Point> points120;
    std::vector<uchar> gray;
    std::vector<cv::Point3i> shieldpoints;
    minh = 0;
    maxh = 0;
    for (int y = 0; y < image.rows; y++) {
        int x;
        const  uchar* dptr = image.ptr<uchar>(y);
        cv::Point pf, pfsmall;
        pf.y = -1;
        pf.x = 0;
        pfsmall.y = 0;
        pfsmall.x = 0;
        cv::Point3i spf;
        spf.y = y;
        spf.x = 0;
        spf.z = 0;
        for (x = 0; x < image.cols; x++) {
            if (dptr[x] > 0) {
                if (pf.y == -1) {
                    pf.y = x;
                }
                if (x < 165) {
                    pfsmall.y = x;
                    pfsmall.x++;
                }
                pf.x++;
                spf.x = x;


            }
        }
        spf.z = pf.x;
        points.push_back(pf);
        shieldpoints.push_back(spf);
        points120.push_back(pfsmall);
    }
    if (shieldpoints.size() < 6)return;
    bool isbignumber = false;
    for (int i = 0; i < shieldpoints.size() / 2; i++) {
        if (shieldpoints[i].x > 150)isbignumber = true;
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {
            minh = i;
            break;
        }
    }
    isbignumber = false;
    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x > 150)isbignumber = true;
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {
            maxh = i;
            break;
        }
    }



    isbignumber = false;
    int startp = 0;
    for (int i = 0; i < shieldpoints.size() / 2; i++) {
        if (shieldpoints[i].x > 60) {
            int bignumbers = 0;
            int endbigp = i;
            for (int j = i + 1; j < shieldpoints.size() / 2 && j < i + 70; j++) {
                if (shieldpoints[j].x > 60 && shieldpoints[j].z > 20) {
                    bignumbers++;
                    endbigp = j;
                }
            }
            if (bignumbers < 10 && shieldpoints[i].z > 10) {
                startp = i;
                isbignumber = true;
                minh = i;
            }
            if (bignumbers > 9) {
                i = endbigp;
            }

        }
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {

            break;
        }
    }



    isbignumber = false;
    startp = 0;
    maxnumber = -9999;
    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x - points[i].y > 100) {
            int bignumbers = 0;
            int endbigp = i;
            for (int j = i - 1; j > shieldpoints.size() / 2 && j > i - 70; j--) {
                if (shieldpoints[j].x - points[i].y > 100 && shieldpoints[j].z > threshhold) {
                    bignumbers++;
                    endbigp = j;
                }
                if (!isbignumber&&maxnumber < shieldpoints[j].z&&shieldpoints.size() - 1 - j > 30 && shieldpoints.size() - 1 - i > 30) {
                    maxnumber = shieldpoints[j].z;
                    maxh = j;

                }
            }
            if (bignumbers < 30 && shieldpoints[i].z > numberpoints &&  shieldpoints.size() - 1 - i > 5) {

                if (shieldpoints.size() - 1 - i < 30 && bignumbers>9) {
                    i = endbigp - 1;
                    continue;

                }


                startp = i;
                isbignumber = true;
                maxh = i;
                break;
            }
            if (bignumbers > 29 || i == shieldpoints.size() - 1) {//img.rows<3000  20
                i = endbigp - endnumber;
            }

        }
        if (!isbignumber&&maxnumber < shieldpoints[i].z&&shieldpoints.size() - 1 - i > 30) {
            maxnumber = shieldpoints[i].z;
            maxh = i;

        }
    }





#if 1

    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 30) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {

                if (shieldpoints[j].x > 160)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }

            if (maxbool&&numbers < 10 && numbers2 < 10 && i>20) {
                minh = i;
                break;
            }
        }




    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 30) {

            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int numbers = 0;
            int numbers2 = 0;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }
            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x > 160)numbers++;

            }

            if (minbool&&numbers < 10 && numbers2 < 10 && shieldpoints.size() - 1 - i>20) {
                maxh = i;
                break;
            }




        }


    }




    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            int threshgan = 120;
            //threshgan = threshgan + i * 80 / 600;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y > 40 && i - j > 5 && shieldpoints[j].x - points[j].y < threshgan) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 170)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;
                    break;
                }

            }

            if (minbool&&maxbool&&numbers < 10) {
                minh = i;
                break;
            }




        }
    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int threshgan = 120;
            //threshgan = threshgan + (shieldpoints.size() - 1 - i) * 80 / 600;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;
                    break;
                }

            }
            int numbers = 0;
            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x - points[j].y < threshgan &&points120[j].y - points[j].y > 40 && j - i > 5) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 170)numbers++;

            }

            if (minbool&&maxbool&&numbers < 10) {
                maxh = i;
                break;
            }




        }


    }



#endif














#if 0

    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            int threshgan = 120;
            //threshgan = threshgan + i * 80 / 600;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y > 40 && i - j > 5 && shieldpoints[j].x - points[j].y < threshgan) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }

            if (minbool&&maxbool&&numbers < 10 && numbers2 < 10) {
                minh = i;
                break;
            }




        }
    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int threshgan = 120;
            //threshgan = threshgan + (shieldpoints.size() - 1 - i) * 80 / 600;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;

            }
            int numbers = 0;

            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x - points[j].y < threshgan &&points120[j].y - points[j].y > 40 && j - i > 5) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers++;

            }

            if (minbool&&maxbool&&numbers < 10 && numbers2 < 10) {
                maxh = i;
                break;
            }




        }


    }

#endif












}
bool data_process::set_params(float  width, float height, float  realwidth, float real_height) {


    std::ofstream of("60.txt");
    float wvalue = realwidth - width;
    float hvalue = real_height - height;
    params_[2] = params_[2] + wvalue;
    params_[3] = params_[3] - hvalue;

    for (int i = 0; i < 8; i++) {
        of << params_[i] << std::endl;

    }
    of.close();

    return true;
}
#include <fstream>
#include <string> 

#include "data_process.h"
#include "img.h"
#include<chrono>
#include <numeric>
#define CKAI 1
//using namespace cv;
struct linepoints
{
    std::vector<cv::Point> points;
    cv::Vec4f lines;
};
data_process::data_process() {
    std::ifstream inf;
    inf.open("60.txt");
    std::string s;

    int line_number = 0;
    while (getline(inf, s) && line_number < 8) {
        params_[line_number] = atof(s.c_str());
        line_number++;
    }

    inf.close();
    is_ysonnx_ = yvonnx_.ReadModel("ck.par", false, 0, false);


}
result data_process::process_profile(bool left, float *profilex0, float *profiley0, int size0, float *profilex1, float *profiley1, int size1, int rails) {
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();


    frequency_ = frequency_ + 1;
    result rst;
    if (size0 < 100 || size1 < 100) {
        int pose0 = (frequency_ - 1) % 5;
        width_[pose0] = 0;
        height_[pose0] = 0;
        return rst;

    }

    std::vector<float> source_x;
    std::vector<float> source_y;
    std::vector<float> standard_x;
    std::vector<float> standard_y;
    std::vector<cv::Point2f> source0s, sources;
    std::vector<float> trans_x;
    std::vector<float> trans_y;


    for (int i = 0; i < size1; i++) {
        profilex1[i] = profilex1[i] * (-1);
        profiley1[i] = profiley1[i] - 500;

        cv::Point2f point, tpoint;
        point.y = profilex1[i];
        point.x = profiley1[i];

        tpoint.y = profilex1[i];
        tpoint.x = profiley1[i];

        profiley1[i] = tpoint.y;
        profilex1[i] = tpoint.x;
        sources.push_back(point);
    }



    for (int i = 0; i < size0; i++) {
        profilex0[i] = profilex0[i] * (-1);
        profiley0[i] = profiley0[i] - 500;

        cv::Point2f  tpoint;


        tpoint.y = profilex0[i];
        tpoint.x = profiley0[i];

        profiley0[i] = tpoint.y;
        profilex0[i] = tpoint.x;
        source0s.push_back(tpoint);
    }

    double angles[2];


    angles[0] = atan(params_[1] / params_[0]);
    angles[1] = atan(params_[5] / params_[4]);



    double x0 = params_[2];
    double y0 = params_[3];

    float icpx, icpy;

    double angle = 0;


    //开始校正
    double  mtheta = 0;
    double mx = 0;
    double my = 0;
    double minx = 0;
    double maxx = 0;
    icpx = 0;
    icpy = 0;

    std::vector<cv::Point2f> tsources, lines;
#if 1 //通过顶面的斜率校正角度，误差较大
    std::sort(sources.begin(), sources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y > b.y; });

    int minHessian = 400;



    for (int i = 0; i < sources.size(); i++) {
        if (sources[0].y - sources[i].y < 30.0)
            tsources.push_back(sources[i]);
        else break;

    }

    std::sort(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.x < b.x; });


    maxx = std::max_element(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.x < b.x; })->x - 26.5;
    minx = maxx - 20;

    for (auto point : tsources) {
        if (point.x > minx&&point.x < maxx) {
            lines.push_back(point);
        }
    }
    /*   cv::Vec4f line;

       cv::fitLine(lines, line, cv::DIST_L2, 0, 1e-2, 1e-2);
       auto linek = line[1] / line[0];
       double lineb = line[3] - linek * line[2];
       auto parallel = atan(linek)*180.0 / PI;
       mtheta = atan(linek);*/
       // return mtheta;


#endif



    //    cv::Mat  dynamicdlt = cv::Mat::zeros(8, 1, CV_32F);

    bool result1 = false;
    bool result2 = false;

    float dynamic_angle = 0;
    //找不到圆心会耗时加重
    //auto havecircle = img::find_dynamic_circle_points(sources, left,source_x, source_y, standard_x, standard_y);//圆心位置会有一定影响
    //if (havecircle) {
        //dynamic_angle = img::translate(1, source_x, source_y, standard_x, standard_y, trans_x, trans_y, dynamicdlt);
        //icpx = dynamicdlt.at<float>(2 + 1 * 4, 0);
       // icpy = dynamicdlt.at<float>(3 + 1 * 4, 0);
    //}


    if (abs(dynamic_angle) < 0.00000001) {
        dynamic_angle = angles[1];
        icpx = params_[6];
        icpy = params_[7];
    }

    std::vector<cv::Point2f> cstandards, csources, csource2s;

    /********************
    由于拟合使用的是轨腰部位400mm与20mm的圆心拟合，拟合算法为随机采样，角度一致性存在1度左右误差。
    所以校正时采用不同部位，如轨鄂和顶面中心的点进行，查找特定的两点，轨鄂与中心点，通过两点进行icp校正，并保证一致性。
    *******************/
    csources = sources;
    tsources.clear();
    source_x.clear();
    source_y.clear();
    standard_x.clear();
    standard_y.clear();
    std::vector<cv::Point2f> tsources2;


    /******************************************************************
   数据分段，校正后总共产生上下两部分，tsources，tsources2，当轮廓仪经过螺孔时
   数据拟合圆心会失败，因其螺孔周边产生了太多噪声
   *************************************************************************/


    for (int i = 0; i < sources.size(); i++) {
        csources[i].x = params_[4] * sources[i].x + params_[5] * sources[i].y + params_[6];
        csources[i].y = params_[5] * (-sources[i].x) + params_[4] * sources[i].y + params_[7];

    }
    auto heightpoint = *std::max_element(csources.begin(), csources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y < b.y; });
    auto hvalue = heightpoint.y - 60;

    for (int i = 0; i < csources.size(); i++) {
        if (csources[i].y > hvalue) {
            if (tsources.size() == 0) {
                tsources.push_back(csources[i]);
                continue;
            }
            if (tsources.size() != 0 && sqrt(pow(csources[i].x - tsources[tsources.size() - 1].x, 2) + pow(csources[i].y - tsources[tsources.size() - 1].y, 2)) < 5) { tsources.push_back(csources[i]); }//去除孤立点
        }

        if (csources[i].y < hvalue && csources[i].x>0) {
            tsources2.push_back(csources[i]);
        }

    }












    if (tsources.size() < 1)return  rst;
    auto  minpoint = *std::max_element(tsources.begin(), tsources.end(), [](cv::Point2f a, cv::Point2f b) {return a.y > b.y; });

    left ? standard_x.push_back(36.4264) : standard_x.push_back(-36.4264);
    standard_y.push_back(141.066);
    standard_x.push_back(0);
    standard_y.push_back(176);
    left ? standard_x.push_back(10.2) : standard_x.push_back(-10.2);
    standard_y.push_back(39.5701);

    double tdistance = sqrt(pow(standard_x[0], 2) + pow(176 - 141.066, 2));
    double tdistance2 = sqrt(pow(standard_x[0] - standard_x[2], 2) + pow(39.5701 - 141.066, 2));
    cv::Point2f maxpoint, maxpoint2;
    double mindistance = 10000;
    double mindistance2 = 10000;



    for (auto point : tsources) {
        auto distance = abs(sqrt(pow(point.x - minpoint.x, 2) + pow(point.y - minpoint.y, 2)) - tdistance);
        if (distance < mindistance) {
            maxpoint = point;
            mindistance = distance;
        }

    }
    for (auto point : tsources2) {
        auto distance = abs(sqrt(pow(point.x - minpoint.x, 2) + pow(point.y - minpoint.y, 2)) - tdistance2);
        if (distance < mindistance2) {
            maxpoint2 = point;
            mindistance2 = distance;
        }

    }


    source_x.push_back(minpoint.x);
    source_y.push_back(minpoint.y);
    source_x.push_back(maxpoint.x);
    source_y.push_back(maxpoint.y);
    source_x.push_back(maxpoint2.x);
    source_y.push_back(maxpoint2.y);
    cv::Mat dynamicdlt = cv::Mat::zeros(8, 1, CV_32F);
    auto check_angle = img::translate(1, source_x, source_y, standard_x, standard_y, trans_x, trans_y, dynamicdlt);
    angle = angles[1];
    if (abs(check_angle) > 0.00000001&&abs(check_angle) < 0.1) {
        auto ttt = dynamic_angle + check_angle;
        auto ticpx = icpx;
        auto ticpy = icpy;
        angle = dynamic_angle + check_angle;
        icpx = cos(check_angle)*ticpx + sin(check_angle)*ticpy + dynamicdlt.at<float>(2 + 1 * 4, 0);
        icpy = cos(check_angle)*ticpy - sin(check_angle)*ticpx + dynamicdlt.at<float>(3 + 1 * 4, 0);
    }

    //if (dynamic_angle + check_angle > 0.2) {
    //    angle = dynamic_angle + check_angle;
    //}



    angle = angle - angles[1];

    //   angles[1] = angle + angles[1];
    angles[0] += angle;
    angles[1] += angle;
    auto alpha = atan(y0 / x0) - angle;
    auto sl = sqrt(pow(x0, 2) + pow(y0, 2));
    x0 = sl * cos(alpha);
    y0 = sl * sin(alpha);


    for (int i = 0; i < size0; i++) {
        source0s[i].x = cos(angles[0])*profilex0[i] + sin(angles[0])*profiley0[i];
        source0s[i].y = -sin(angles[0])*profilex0[i] + cos(angles[0])*profiley0[i];
        profilex0[i] = source0s[i].x;
        profiley0[i] = source0s[i].y;

    }
    for (int i = 0; i < size1; i++) {
        sources[i].x = cos(angles[1])*profilex1[i] + sin(angles[1])*profiley1[i];
        sources[i].y = -sin(angles[1])*profilex1[i] + cos(angles[1])*profiley1[i];
        profilex1[i] = sources[i].x + x0;
        profiley1[i] = sources[i].y + y0;

    }









#if 0

    cv::Mat images = cv::Mat::zeros(1000, 1000, CV_8UC3);

    for (int i = 0; i < source0s.size(); i++) {
        float cvy = ((source0s[i].y + 600));
        float cvx = ((source0s[i].x + 600));
        if (0 < cvx&&cvx < images.cols && 0 < cvy&&cvy < images.rows)
            images.at<cv::Vec3b>(cvy, cvx) = cv::Vec3b(255, 255, 255);

    }




    cv::resize(images, images, cv::Size(1000, 1000));
    // cv::flip(images, images, 0);
    imshow("source0s", images);
    cv::waitKey(2);

#endif





    cv::Point2f fpoints[2];//高度与拉出由不同的点计算


    std::vector<cv::Point2f> underside_points;

    cv::Point2f abrasion;
    bool existance;
    bool expanding = false;
    fpoints[0] = img::dynamic_all_third_featch(source0s, underside_points, abrasion, existance, expanding, rails);
    fpoints[1] = img::get_featch_point(sources);
    rst.imgtype = feature::none;

    //拉出值为 a = L + x’t - xt
// 导高值为   b = yt + y’t - H

    rst.width = abs(fpoints[0].x) - abs(fpoints[1].x) + x0;
    rst.height = abs(fpoints[0].y) + abs(fpoints[1].y) - y0;

    int shield_size = 0;
    double distance = 0.0;
    auto joints = img::expansion_jionts(source0s, shield_size, distance);
    rst.shield = shield_size;







    int pose = (frequency_ - 1) % 5;
    width_[pose] = rst.width;
    height_[pose] = rst.height;
    shild_[pose] = shield_size;
    int hshild = 0;
    int hend = 0;
    if (frequency_ > 5) {
        double sumw = std::accumulate(width_, width_ + 5, 0) / 5.0;
        double sumh = std::accumulate(height_, height_ + 5, 0) / 5.0;
        double wm = 0;
        double wh = 0;
        double sw = 100;
        double sh = 100;
        double sw2 = width_[(frequency_ - 2) % 5];
        double sh2 = height_[(frequency_ - 2) % 5];
        int wi = 0;
        int hi = 0;
        for (int i = 0; i < 5; i++) {
            if (shild_[i] > 130)
                hshild++;

            if (height_[i] > 250)
                hend++;
            if (abs(sumw - width_[i]) > wm) {
                wm = abs(sumw - width_[i]);
                wi = i;

            }
            if (abs(sumh - height_[i]) > wh) {
                wh = abs(sumh - height_[i]);
                hi = i;

            }
            if (abs(width_[i] - 752.5) < sw) {
                sw2 = width_[i];
                sw = abs(width_[i] - 752.5);
            }
            if (abs(height_[i] - 200) < sh) {
                sh2 = height_[i];
                sh = abs(height_[i] - 200);
            }


        }
        //if (wi == pose && wm > 5) {
        if (abs(752.5 - width_[pose]) > 30 && width_[pose] != 0 && abs(752.5 - width_[pose]) < 146) {
            rst.width = width_[(frequency_ - 2) % 5];
            //rst.width = width_[hi];
            //rst.width = sw2;
            //width_[pose] = rst.width;

        }
        //if (hi == pose && wh > 5) {
        if (abs(sumh - height_[pose]) > 5 && height_[pose] != 0) {
            rst.height = height_[(frequency_ - 1) % 5];
            // rst.height = height_[hi];
            // rst.height = sh2;
             //height_[pose] = rst.height;

        }



    }



    // std::cout << "0=" << width_[0] << ":1=" << width_[1] << ":2=" << width_[2] << ":3=" << width_[3] << ":4=" << width_[4] << ":5=" << std::endl;







    bool  insulating = false;
    int sdistance0 = 0;
    if (source0s.size() > 2) {
        sdistance0 = abs(source0s[0].x - source0s[source0s.size() - 1].x);
    }

    if (shield_size < 91 && hend>3)rst.imgtype = feature::endbend;

    if (90 < shield_size&&shield_size < 131 && distance < 1.0&&rst.height < 290 && rst.height > 150 && rst.width < 800 && hshild < 1)insulating = true;


    // if (shield_size < 91 && height_[0] > 250 && height_[1] > 250 && height_[2] > 250 && height_[3] > 250 && height_[4] > 250)rst.imgtype = feature::endbend;

    if (rst.imgtype == feature::endbend&&rst.height < 290 && rst.height > 270 && distance < 1.0)insulating = true;
    if (insulating)rst.imgtype = feature::insulatorbracket;
    if (shield_size > 130 && sdistance0 > 50 && distance < 10 && source0s.size()>650 && height_[pose] > 150 && width_[pose] > 700 && hshild > 1)rst.imgtype = feature::expansion;
    //if(expanding&&distance>3&&distance<10)rst.imgtype = feature::expansion;

   // rst.gap[0] = distance;
    if (rst.width > 900 || rst.width < 500 || rst.height>320)rst.imgtype = feature::none;


    if (!existance) {
        auto duration_capture = clock::now() - last;
        rst.time = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        return rst;

    }





    cv::Vec4f line;
    cv::fitLine(underside_points, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    auto linek = line[1] / line[0];
    double lineb = line[3] - linek * line[2];
    rst.parallel = (atan(linek)*180.0 / PI);

    rst.parallel = rst.parallel - int(rst.parallel);

    rst.abrasion = abs(abrasion.y - linek * abrasion.x - lineb) / sqrt(1 + linek * linek);
    if (rails == 0) {
        rst.abrasion > 3 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 3 - rst.abrasion;

    }
    if (rails == 1) {
        rst.abrasion > 13 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 13 - rst.abrasion;
        rst.abrasion > 6 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : 1;
    }
    if (rails == 2) {
        rst.abrasion > 12.4 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : rst.abrasion = 12.4 - rst.abrasion;
        rst.abrasion > 5 ? rst.abrasion = rst.abrasion - int(rst.abrasion) : 1;
    }

    rst.pointdepth = -1;

    for (auto point : underside_points) {
        auto distance = abs(point.y - linek * point.x - lineb) / sqrt(1 + linek * linek);
        if (rst.pointdepth < distance)
            rst.pointdepth = distance;
    }











    //rst.parallel = angle * 180.0 / PI;//测试用
    auto duration_capture = clock::now() - last;

    rst.time = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
    //rst.parallel = angle;
    return rst;


}
result data_process::process_img(unsigned char  *datas, int size, int rails) {
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();
    result rst;
    if (size / 2048 < 1500)return rst;
    cv::Mat palyimg, img2, dstImage;
    auto image = cv::Mat::Mat(size / 2048, 2048, CV_8UC1, datas).clone();







    cv::Rect roi;
    if (rails % 2 != 0) {
        rst = process_shield_img(datas, roi, palyimg, size, rails);
        return  rst;
    }


    if (rails % 2 != 0 || image.cols < 1500) {
        // cv::imwrite("H:/third-rail/img-qingdao/testgap/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);
        return  rst;
    }


#if CKAI

    if (!is_ysonnx_)return rst;
    std::vector<OutputSeg> output, output2;
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    bool find = yvonnx_.OnnxDetect(image, output);
    std::vector<cv::Scalar> color;

    cv::RotatedRect rotate_rects[2];

    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(0, 0, 255));
    }

    if (find) {

        int size_gap = 0;
        for (int k = 0; k < output.size(); k++) {
            if (output[k].id != 0)continue;

            auto bmask = output[k].boxMask;
            std::vector<cv::Point>points;
            for (int i = 0; i < bmask.rows; i++)
            {
                const uchar* ph = bmask.ptr<uchar>(i);
                for (int j = 0; j < bmask.cols; j++)
                {
                    if (ph[j] > 0) {
                        //points.push_back(cv::Point(j,i));
                        points.push_back(cv::Point(j + output[k].box.x, i + output[k].box.y));//真实坐标
                    }
                }

            }
            if (points.size() > 1 && size_gap < 2) {
                rotate_rects[size_gap] = cv::minAreaRect(points);
                output2.push_back(output[k]);
                size_gap++;
            }


        }


















        float h1, h2;
        abs(rotate_rects[0].angle) < 45 ? h1 = rotate_rects[0].size.height : h1 = rotate_rects[0].size.width;
        abs(rotate_rects[1].angle) < 45 ? h2 = rotate_rects[1].size.height : h2 = rotate_rects[1].size.width;

        if (output2.size() > 1) {
            if (output2[0].box.y < output2[1].box.y) {
                auto th = h1;
                h1 = h2;
                h2 = th;
                auto sangle = rotate_rects[0].angle;
                rotate_rects[0].angle = rotate_rects[1].angle;
                rotate_rects[1].angle = sangle;
            }
        }


        auto duration_capture1 = clock::now() - last;
        float radian1 = abs(rotate_rects[0].angle) / 180 * CV_PI;
        float radian2 = abs(rotate_rects[1].angle) / 180 * CV_PI;
        if (abs(rotate_rects[0].angle) > 45) {
            radian1 = (90 - abs(rotate_rects[0].angle)) / 180 * CV_PI;


        }
        if (abs(rotate_rects[1].angle) > 45) {

            radian2 = (90 - abs(rotate_rects[1].angle)) / 180 * CV_PI;

        }


        rst.gap[0] = h1 * 0.775 / abs(cos(radian1));
        rst.gap[1] = h2 * 0.775 / abs(cos(radian2));
        auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
        rst.time = timeuse;









        /************************显示开始*************************/


        DrawPred(image, output, yvonnx_._className, color);
        cv::Point2f vertices[4], vertices1[4];
        rotate_rects[0].points(vertices);//获取矩形的四个点
        rotate_rects[1].points(vertices1);//获取矩形的四个点
        std::string label = "TimeUse: " + std::to_string(timeuse) + "=" + std::to_string(rst.gap[0]) + "=" + std::to_string(rst.gap[1]);

        for (int i = 0; i < 4; i++) {
            cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::line(image, vertices1[i], vertices1[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            // cv::line(palyimg, cv::Point2f(vertices[i].x + output[0].box.x, vertices[i].y + output[0].box.y), cv::Point2f(vertices[(i + 1) % 4].x + output[0].box.x, vertices[(i + 1) % 4].y + output[0].box.y), cv::Scalar(0, 255, 0));
             //cv::line(palyimg, cv::Point2f(vertices1[i].x + output[1].box.x, vertices1[i].y + output[1].box.y), cv::Point2f(vertices1[(i + 1) % 4].x + output[1].box.x, vertices1[(i + 1) % 4].y + output[1].box.y), cv::Scalar(0, 255, 0));

        }
        cv::putText(image, label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2, 8);
        cv::resize(image, image, cv::Size(1024, 1024));
        cv::imshow("result", image);
        std::cout << h1 << "=h1h2=" << h2 << std::endl;
        cv::waitKey();
        /************************显示结束*************************/


        return  rst;
    }

    else

        return  rst;








#endif

    roi.y = 0;
    roi.height = image.rows;
    if (image.rows < 2600 || roi.width<1000 || roi.height<1600 || roi.x + roi.width>image.cols || roi.y + roi.height>image.rows) {
        roi.x = 400;
        roi.width = image.cols - 400;//600
        roi.y = 0;
        roi.height = image.rows;
    }




    img2 = image(roi).clone();
    {

        int mean_value = static_cast<int>(cv::mean(img2).val[0]);
        unsigned int gray_value = 20;
        gray_value = mean_value / 2;
        cv::medianBlur(img2, img2, 11);
        cv::Canny(img2, img2, gray_value, (gray_value) * 2);
        // cv::imwrite("d:/61.bmp", img2);

      /*   img = img2.clone();
         cv::cvtColor(img, img, CV_GRAY2BGR);*/

        std::vector<std::vector<cv::Point>>contours, contoursfilter;
        std::vector<cv::Vec4i>hierarchy;
        cv::findContours(img2, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        bool is_coincidence = false;
        std::vector<cv::Vec4f> lines;
        std::vector<double> ks, negatives, positives, ppoints;
        std::vector<cv::Point> linepoints;
        for (int i = 0; i < contours.size(); i++)
        {
            for (auto contour : contours[i]) {
                linepoints.push_back(contour);
            }

            if (contours[i].size() > 100)//crect.width / crect.height>3&&
            {
                cv::Vec4f line;
                cv::fitLine(contours[i], line, cv::DIST_L2, 0, 1e-2, 1e-2);
                auto crect = cv::boundingRect(contours[i]);

                double k = line[1] / line[0];





                if (crect.width > crect.height) {
                    if (abs(k) > 0.2&& abs(k) < 0.4) {
                        k = atan(k)*180.0 / CV_PI;
                        k > 0 ? positives.push_back(k) : negatives.push_back(k);

                    }


                    contoursfilter.push_back(contours[i]);


                    float kk = 0.3;
                    k > 0 ? kk = 0.3 : kk = -0.3;
                    std::vector<float> bs;
                    for (int j = 0; j < contours[i].size(); j++) {
                        bs.push_back(contours[i][j].y - kk * contours[i][j].x);
                    }
                    cv::Mat meanb, stdb;
                    cv::meanStdDev(bs, meanb, stdb);
                    bs.size();

                    auto tmean = meanb.at<double>(0, 0);
                    auto tstdb = stdb.at<double>(0, 0);





                }




            }

        }
        int direction = 1;





        positives.size() > negatives.size() ? direction = -1 : direction = 1;
        direction > 0 ? ppoints = negatives : ppoints = positives;

        if (ppoints.size() < 2)return rst;
        cv::Mat pmean, pstd;
        cv::meanStdDev(ppoints, pmean, pstd);
        auto tpmean = pmean.at<double>(0, 0);
        double tpp = 1000;
        double anglel = 17.5;

        for (auto point : ppoints) {
            if (abs(point - tpmean) < tpp) {
                tpp = abs(point - tpmean);
                anglel = point;
            }

        }






        //tpmean < 0 ? anglel = anglel * -1 : anglel = anglel;


        float radian = (float)(-1) * anglel / 180 * CV_PI;

        cv::Mat outimg(2000, 2000, CV_8U, cv::Scalar(0));
        cv::cvtColor(outimg, outimg, CV_GRAY2BGR);//结果图片为彩色

        for (auto &point : linepoints) {
            auto tpoint = point;
            point.x = tpoint.x*cos(radian) - tpoint.y*sin(radian);
            point.y = tpoint.x*sin(radian) + tpoint.y*cos(radian);


        }


        cv::Mat hist_img(2000, 2000, CV_8U, cv::Scalar(255));


        cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色


        if (linepoints.size() < 2)return rst;


        std::sort(linepoints.begin(), linepoints.end(), [](cv::Point point1, cv::Point point2) {
            return point1.y < point2.y;
        });



        int poffset = 0;
        if (-linepoints[0].y > 0)poffset = -linepoints[0].y;
        int prow = linepoints[linepoints.size() - 1].y + poffset + 1;
        std::vector<cv::Point> anglepoints(prow);


        for (auto &point : linepoints) {
            anglepoints[point.y + poffset].x++;
            anglepoints[point.y + poffset].y = point.y + poffset;
        }


        std::vector<cv::Point3i> results;
        std::vector<cv::Point2i> points50;
        for (int i = 0; i < anglepoints.size(); i++) {

            int maxpoint = anglepoints[i].x;
            int maxp = i;

            cv::Point2i p2;
            p2.x = maxpoint;
            p2.y = maxp;
            if (anglepoints[i].x > 50)
                points50.push_back(p2);
            continue;


        }

        if (points50.size() < 4)return rst;
        std::sort(points50.begin(), points50.end(), [](cv::Point point1, cv::Point point2) {
            return point1.x > point2.x;
        });

        cv::Point frontp1, frontp, frontp2, behindp1, behindp, behindp2;

        bool fb = false;
        bool bb = false;

        for (int i = 1; i < points50.size(); i++) {
            auto distance = points50[i].y - points50[0].y;
            if (abs(distance) > 200) {
                for (int j = 1; j < points50.size(); j++) {
                    if (abs(points50[j].y - points50[0].y) < 190 && !fb&&abs(points50[j].y - points50[0].y) > 15) {

                        int py = points50[j].y;
                        if (points50[j].y < points50[0].y)  py = py - 20;
                        if (py < 0)py = 0;
                        int bignumber = 0;
                        for (int k = py; k < py + 20 && k < anglepoints.size(); k++) {
                            if (anglepoints[k].x > 50)bignumber++;
                        }
                        if (bignumber < 17) {

                            fb = true;
                            if (distance > 0) {
                                frontp1 = points50[j];
                                frontp2 = points50[0];
                            }
                            if (distance < 0) {
                                behindp1 = points50[j];
                                behindp2 = points50[0];
                            }
                        }










                    }
                    if (abs(points50[j].y - points50[i].y) < 190 && i != j && !bb&&abs(points50[j].y - points50[i].y) > 15) {
                        int py = points50[j].y;
                        if (points50[j].y < points50[i].y)  py = py - 20;
                        if (py < 0)py = 0;
                        int bignumber = 0;
                        for (int k = py; k < py + 20 && k < anglepoints.size(); k++) {
                            if (anglepoints[k].x > 50)bignumber++;
                        }
                        if (bignumber < 17) {




                            bb = true;
                            if (distance > 0) {
                                behindp1 = points50[j];
                                behindp2 = points50[i];
                            }
                            if (distance < 0) {
                                frontp1 = points50[j];
                                frontp2 = points50[i];
                            }
                        }

                    }
                }
                break;
            }


        }






        cv::putText(hist_img, std::to_string(frontp1.y - frontp2.y), cv::Point(20, 30 + 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
        cv::line(hist_img, cv::Point(frontp1.y, 1990 - frontp1.x), cv::Point(frontp2.y, 1990 - frontp2.x), cv::Scalar(255, 0, 0));



        auto duration_capture = clock::now() - last;

        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();

        rst.time = ttime;
        rst.gap[0] = abs(frontp1.y - frontp2.y) * 0.775 / abs(cos(-radian));
        rst.gap[1] = abs(behindp1.y - behindp2.y) * 0.775 / abs(cos(-radian));




#if 0

        cv::Mat playmat = image.clone();
        cv::cvtColor(playmat, playmat, cv::COLOR_GRAY2BGR);
        cv::Point vertices[2][4];
        vertices[0][0].x = 0;
        vertices[0][0].y = frontp1.y - poffset;
        vertices[0][1].x = 2000;
        vertices[0][1].y = frontp1.y - poffset;

        vertices[0][2].x = 0;
        vertices[0][2].y = frontp2.y - poffset;
        vertices[0][3].x = 2000;
        vertices[0][3].y = frontp2.y - poffset;

        vertices[1][0].x = 0;
        vertices[1][0].y = behindp1.y - poffset;
        vertices[1][1].x = 2000;
        vertices[1][1].y = behindp1.y - poffset;

        vertices[1][2].x = 0;
        vertices[1][2].y = behindp2.y - poffset;
        vertices[1][3].x = 2000;
        vertices[1][3].y = behindp2.y - poffset;

        for (int j = 0; j < 4; j++) {
            auto tpoint0 = vertices[0][j];
            auto tpoint1 = vertices[1][j];
            vertices[0][j].x = tpoint0.x*cos(-radian) - tpoint0.y*sin(-radian) + roi.x;
            vertices[0][j].y = tpoint0.x*sin(-radian) + tpoint0.y*cos(-radian) + roi.y;
            vertices[1][j].x = tpoint1.x*cos(-radian) - tpoint1.y*sin(-radian) + roi.x;
            vertices[1][j].y = tpoint1.x*sin(-radian) + tpoint1.y*cos(-radian) + roi.y;
        }
        for (int j = 0; j < 4; j = j + 2) {
            cv::line(playmat, vertices[0][j], vertices[0][(j + 1) % 4], cv::Scalar(0, 255, 0), 1, 16);
            cv::line(playmat, vertices[1][j], vertices[1][(j + 1) % 4], cv::Scalar(0, 255, 0), 1, 16);
        }
        cv::putText(playmat, std::to_string(rst.gap[0]), cv::Point(100, 100 + 30), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        cv::putText(playmat, std::to_string(rst.gap[1]), cv::Point(100, 100 + 230), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        cv::putText(playmat, std::to_string(anglel), cv::Point(100, 100 + 430), cv::FONT_HERSHEY_PLAIN, 10, cv::Scalar(255, 255, 255));
        //cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", playmat);
       // cv::imwrite("G:/datas/test/biaoji/" + std::to_string(numbers_) + ".bmp", playmat);
        cv::imshow("1", playmat);
        cv::waitKey();
        rst.gap_position[0] = vertices[0][1].y;
        rst.gap_position[1] = vertices[1][1].y;
#endif
        return rst;


    }

    int mean_value = static_cast<int>(cv::mean(img2).val[0]);
    unsigned int gray_value = 20;
    gray_value = mean_value / 2;
    if (gray_value > 20)gray_value = 18;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    //cv::adaptiveThreshold(dstImage, dstImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 4);
    //cv::dilate(dstImage, dstImage, element2);
    //morphologyEx(dstImage, dstImage, CV_MOP_OPEN, element, cv::Point(-1, -1), 3);
   // cv::medianBlur(dstImage, dstImage, 15);

    auto img3 = img2.clone();




    cv::threshold(img2, img2, mean_value / 2, 255, CV_THRESH_BINARY_INV);
    morphologyEx(img2, img2, CV_MOP_OPEN, element, cv::Point(-1, -1), 1);
    morphologyEx(img2, img2, CV_MOP_OPEN, element, cv::Point(-1, -1), 2);
    dstImage = img2(cv::Rect(500, 500, 400, 1000)).clone();
    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(dstImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //cv::imwrite("d:/img2.bmp", dstImage);
    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> points1, std::vector<cv::Point>points2)
    {return  points1.size() > points2.size(); });

    int updown = 0;
    cv::RotatedRect rrect[2];
    cv::Point2f vertices[2][4];

    for (int i = 0; i < contours.size(); i++) {

        std::sort(contours[i].begin(), contours[i].end(), [](cv::Point point1, cv::Point point2)
        {return  point1.x < point2.x; });

        auto hrect = cv::boundingRect(contours[i]);

        if (contours[i][0].x < 200 && updown < 2 && hrect.height / hrect.width < 3) {

            rrect[updown] = cv::minAreaRect(contours[i]);

            rrect[updown].points(vertices[updown]);//获取矩形的四个点

            updown++;
        }
        if (updown == 2)break;

    }
    if (updown < 2) {

        contours.clear();
        hierarchy.clear();
        cv::threshold(img3, img3, mean_value / 2, 255, CV_THRESH_BINARY_INV);
        //cv::dilate(img3, img3, element2);
        morphologyEx(img3, img3, CV_MOP_OPEN, element2, cv::Point(-1, -1), 1);
        dstImage = img3(cv::Rect(500, 500, 400, 1000)).clone();


        cv::findContours(dstImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> points1, std::vector<cv::Point>points2)
        {return  points1.size() > points2.size(); });

        int updown = 0;


        for (int i = 0; i < contours.size(); i++) {

            std::sort(contours[i].begin(), contours[i].end(), [](cv::Point point1, cv::Point point2)
            {return  point1.x < point2.x; });

            auto hrect = cv::boundingRect(contours[i]);

            if (contours[i][0].x < 200 && updown < 2 && hrect.height / hrect.width < 3) {

                rrect[updown] = cv::minAreaRect(contours[i]);

                rrect[updown].points(vertices[updown]);

                updown++;
            }
            if (updown == 2)break;

        }

    }





    cv::cvtColor(dstImage, dstImage, CV_GRAY2RGB);



    for (int i = 0; i < 4; i++) {
        cv::line(dstImage, vertices[0][i], vertices[0][(i + 1) % 4], cv::Scalar(0, 255, 0));
        cv::line(dstImage, vertices[1][i], vertices[1][(i + 1) % 4], cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices[0][i].x + roi.x + 500, vertices[0][i].y + roi.y + 500), cv::Point2f(vertices[0][(i + 1) % 4].x + roi.x + 500, vertices[0][(i + 1) % 4].y + roi.y + 500), cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices[1][i].x + roi.x + 500, vertices[1][i].y + roi.y + 500), cv::Point2f(vertices[1][(i + 1) % 4].x + roi.x + 500, vertices[1][(i + 1) % 4].y + roi.y + 500), cv::Scalar(0, 255, 0));


    }
    float h1, h2;
    abs(rrect[0].angle) < 45 ? h1 = rrect[0].size.height : h1 = rrect[0].size.width;
    abs(rrect[1].angle) < 45 ? h2 = rrect[1].size.height : h2 = rrect[1].size.width;

    auto duration_capture = clock::now() - last;

    auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();

    rst.time = ttime;
    rst.gap[0] = h1 * 0.75;
    rst.gap[1] = h2 * 0.75;

    //  cv::imshow("1", dstImage);
      //cv::waitKey();


      //rst.gap[0] = rrect[0].center.x + roi.x + 500;
     // rst.gap[1] = rrect[1].center.x + roi.x + 500;
    rst.gap_position[0] = rrect[0].center.y + roi.y + 500;
    rst.gap_position[1] = rrect[1].center.y + roi.y + 500;
    //rst.width = rrect[0].size.width;
    //rst.height = rrect[0].size.height;
    //rst.parallel = rrect[1].size.width;
    //rst.pointdepth= rrect[1].size.height;
    //rst.abrasion = rrect[0].angle;
    //rst.time=rrect[1].angle;

  // cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);
   // cv::resize(palyimg, palyimg, cv::Size(1024, 1024));



    return rst;



#if 0


    cv::Mat canyimg, img;

    //cv::medianBlur(img, img, 7);
    //cv::Canny(image, img, 30, 60);//125,280
//    cv::resize(canyimg, canyimg, cv::Size(1024, 1024));


    int p1, p2;
    cv::Rect r1;
    // auto himg = img::get_cols_roi(img,!rails, r1);

    int gray_value = static_cast<int>(cv::mean(image).val[0]);//根据整张图片的平均灰度值进行阈值的设定
    cv::Mat  sobel_imgh, sobel_imgw;

    if (image.rows > 3000 && !roi.empty() && image.rows > roi.y + roi.height&&image.cols > roi.x + roi.width) {

        img = image(roi).clone();
    }
    else {
        img = image.clone();
    }

    cv::medianBlur(img, img, 3);
    if (gray_value > 50) cv::blur(img, img, cv::Size(3, 3));//亮度较高的杂波较多使用均值滤波，间隙中反光大，均值去掉部分边缘

    bool railtype = false;
    cv::Canny(img, img, gray_value - 10, (gray_value - 10) * 2);





    if (gray_value > 60) {//亮度高的情况下，白天，会出现间隙孔轮廓，影响直线查找 
        cv::Sobel(img, sobel_imgh, CV_8U, 1, 0, 3, 1, 0);
        cv::threshold(sobel_imgh, sobel_imgh, 250, 255, THRESH_BINARY);

        img = img - sobel_imgh;

    }


    gray_value < 60 ? railtype = false : railtype = true;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::dilate(img, img, element);

    if (image.rows < 2600) {
        auto himg = img::get_cols_roi(img, railtype, roi);
        img = himg;
    }

    // cv::imshow("1", img);
    //cv::waitKey();

















    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);


    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


    std::vector<linepoints> alllp;

    for (int i = 0; i < contours.size(); i++)
    {
        std::sort(contours[i].begin(), contours[i].end(), [](cv::Point &c1, cv::Point &c2) {
            return c1.x < c2.x;//从小到大
        });
    }




    cv::cvtColor(img, img, CV_GRAY2BGR);

    bool is_coincidence = false;
    std::vector<cv::Vec4f> lines;
    std::vector<double> ks;
    for (int i = 0; i < contours.size(); i++)
    {
        auto crect = cv::minAreaRect(contours[i]);
        if (contours[i].size() > 100)//crect.width / crect.height>3&&
        {
            cv::Vec4f line;
            cv::fitLine(contours[i], line, cv::DIST_L2, 0, 1e-2, 1e-2);


            double k = line[1] / line[0];



            double distance = 0.0;

            int maxdistancex = 0;
            std::vector<cv::Point> contour;
            //distance = fabs(k*(tpoints[0].x - line[2]) + line[3] - tpoints[0].y);
            for (int j = 0; j < contours[i].size(); j++) {
                double tempdistance = fabs(k*(contours[i][j].x - line[2]) + line[3] - contours[i][j].y);
                if (tempdistance > distance) {
                    distance = tempdistance;
                    maxdistancex = i;
                }
                if (tempdistance < 1) {//精确取值
                    contour.push_back(contours[i][j]);
                }
            }
            double kk = atan(k)*180.0 / 3.14159265;
            auto twidth = crect.size.width;
            auto theight = crect.size.height;

            kk > 0 ? crect.size.width = theight : crect.size.width = twidth;
            kk > 0 ? crect.size.height = twidth : crect.size.height = theight;
            cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0));
            if ((crect.size.width / crect.size.height > 5) && crect.size.width > 80 && abs(kk) > 8 && abs(kk) < 60 && crect.size.height < 50)//&&abs(kk)>10&& distance<3)//kk>10&&kk<60&& crect.width>100)
            {

                std::vector<cv::Point> contour;

                if (distance > 4) {

                    auto pnumber = contours[i].size() / 2;
                    maxdistancex < pnumber ? contour.assign(contours[i].begin() + pnumber, contours[i].end()) : contour.assign(contours[i].begin(), contours[i].begin() + pnumber);

                    cv::fitLine(contour, line, cv::DIST_L2, 0, 1e-2, 1e-2);
                }


                linepoints onelp;

                contour.size() > 0 ? onelp.points = contour : onelp.points = contours[i];
                onelp.lines = line;


                cv::Point2f point1, point2;
                point2.x = line[2];
                point2.y = line[3];
                double k = line[1] / line[0];
                point1.x = 0.0;
                point1.y = k * (0 - line[2]) + line[3];
                line[2] = point1.x;
                line[3] = point1.y;
                lines.push_back(line);

                alllp.push_back(onelp);

                // cv::line(img, point1, point2, cv::Scalar(0, 0, 255));

                double angle = atan(k)*180.0 / 3.14159265;


                ks.push_back(angle);







            }

        }

    }


    std::vector<int> positives;
    std::vector<int> negatives;

    for (int i = 0; i < ks.size(); i++) {
        if (ks[i] >= 0)positives.push_back(i);
        if (ks[i] < 0)negatives.push_back(i);
    }

    std::vector<linepoints> talllp;
    std::vector<cv::Vec4f>tlines;

    if (positives.size() > negatives.size() && negatives.size() > 0) {

        for (int i = 0; i < alllp.size(); i++) {
            bool tn = false;
            for (int j = 0; j < negatives.size(); j++) {
                if (negatives[j] == i) {
                    tn = true;
                    break;
                }


            }
            if (!tn) {
                talllp.push_back(alllp[i]);
                tlines.push_back(lines[i]);
            }
        }
        alllp = talllp;
        lines = tlines;


    }

    if (positives.size() < negatives.size() && positives.size() > 0) {
        for (int i = 0; i < alllp.size(); i++) {
            bool tn = false;
            for (int j = 0; j < positives.size(); j++) {
                if (positives[j] == i) {
                    tn = true;
                    break;
                }


            }
            if (!tn) {
                talllp.push_back(alllp[i]);
                tlines.push_back(lines[i]);
            }
        }


        alllp = talllp;
        lines = tlines;


    }
    if (image.rows > 3000) {


        std::vector<linepoints>::iterator itr = alllp.begin();
        std::vector<cv::Vec4f>::iterator itr2 = lines.begin();
        while (itr != alllp.end())
        {
            auto sum = std::accumulate(itr->points.begin(), itr->points.end(), 0, [](int  a, cv::Point c2) {return a + c2.y; });
            int mean = sum / itr->points.size();
            if (abs(mean - (img.rows / 2)) > 400) {

                itr = alllp.erase(itr);
                itr2 = lines.erase(itr2);
            }
            else {
                itr++;
                itr2++;
            }

        }
    }




    cv::Vec4f line;
    line[2] = 0;
    std::vector<double>distances;
    //for (int i = 0; i < lines.size(); i++) {
    //    if (line[2] == 0) { 
    //        line = lines[i];
    //        continue;
    //    }
    // //   double tempdistance = fabs(k*(tpoints[i].x - line[2]) + line[3] - tpoints[i].y);
    //    double tempdistance = fabs((lines[i][1] / lines[i][0]) * (line[2] - lines[i][2]) + lines[i][3] - line[3]);
    //    double tempdistance2 = fabs((line[1] / line[0]) * (lines[i][2] - line[2]) + line[3] - lines[i][3]);
    //    if (tempdistance > 2) {
    //        

    //        line[2] = 0;
    //        distances.push_back(tempdistance);
    //    }
    //    else { continue; }




    //}
    std::sort(lines.begin(), lines.end(), [](cv::Vec4f  &c1, cv::Vec4f &c2) {
        return c1[3] < c2[3];//从小到大
    });

    std::sort(alllp.begin(), alllp.end(), [](linepoints  &c1, linepoints &c2) {
        return c1.lines[3] < c2.lines[3];//从小到大
    });

    if (alllp.size() < 1)return rst;
    auto middley = (alllp.begin()->lines[3] + alllp.rbegin()->lines[3]) / 2;
    std::vector<cv::Point> points[2];
    std::vector< cv::Vec4f> vlines[2];
    for (int i = 0; i < alllp.size(); i++) {
        if (alllp[i].lines[3] > middley) {
            for (int j = 0; j < alllp[i].points.size(); j++) {
                points[1].push_back(alllp[i].points[j]);//35.bmp异常

            }
            vlines[1].push_back(alllp[i].lines);

        }
        else {
            for (int j = 0; j < alllp[i].points.size(); j++) {
                points[0].push_back(alllp[i].points[j]);

            }
            vlines[0].push_back(alllp[i].lines);
        }
    }

    if (points[1].size() < 1)return rst;
    auto rrc0 = cv::minAreaRect(points[0]);
    auto rrc1 = cv::minAreaRect(points[1]);
    cv::Point2f vertices[4], vertices1[4];
    rrc0.points(vertices);//获取矩形的四个点
    rrc1.points(vertices1);//获取矩形的四个点
    for (int i = 0; i < 4; i++) {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 0));
        cv::line(img, vertices1[i], vertices1[(i + 1) % 4], cv::Scalar(0, 255, 255));
        cv::line(palyimg, cv::Point2f(vertices[i].x + roi.x, vertices[i].y + roi.y), cv::Point2f(vertices[(i + 1) % 4].x + roi.x, vertices[(i + 1) % 4].y + roi.y), cv::Scalar(0, 255, 0));
        cv::line(palyimg, cv::Point2f(vertices1[i].x + roi.x, vertices1[i].y + roi.y), cv::Point2f(vertices1[(i + 1) % 4].x + roi.x, vertices1[(i + 1) % 4].y + roi.y), cv::Scalar(0, 255, 0));


    }

    //cv::imshow("1", palyimg);
    //cv::waitKey();

    if (lines.size() < 1)return rst;

    for (int i = 0; i < lines.size() - 1; i++) {
        double k = lines[i][1] / lines[i][0];
        double kk = atan(k)*180.0 / 3.14159265;
        double tempdistance = fabs((lines[i][1] / lines[i][0]) * (lines[i + 1][2] - lines[i][2]) + lines[i][3] - lines[i + 1][3]);

        //cv::putText(img, std::to_string(i), cv::Point(20*i, lines[i][3]), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

        ks.push_back(kk);

        distances.push_back(tempdistance);





    }

    float h1, h2;
    abs(rrc0.angle) < 45 ? h1 = rrc0.size.height : h1 = rrc0.size.width;
    abs(rrc1.angle) < 45 ? h2 = rrc1.size.height : h2 = rrc1.size.width;

    float hh1, hh2, hh11, hh22;

    hh1 = fabs(vlines[0][0][3] - vlines[0][vlines[0].size() - 1][3]) / sqrt(1 + pow(vlines[0][0][1] / vlines[0][0][0], 2));
    hh11 = fabs(vlines[0][0][3] - vlines[0][vlines[0].size() - 1][3]) / sqrt(1 + pow(vlines[0][vlines[0].size() - 1][1] / vlines[0][vlines[0].size() - 1][0], 2));
    hh2 = fabs(vlines[1][0][3] - vlines[1][vlines[1].size() - 1][3]) / sqrt(1 + pow(vlines[1][0][1] / vlines[1][0][0], 2));
    hh22 = fabs(vlines[1][0][3] - vlines[1][vlines[1].size() - 1][3]) / sqrt(1 + pow(vlines[1][vlines[1].size() - 1][1] / vlines[1][vlines[1].size() - 1][0], 2));

    float vk1 = vlines[0][vlines[0].size() - 1][1] / vlines[0][vlines[0].size() - 1][0];
    float vk2 = vlines[1][vlines[1].size() - 1][1] / vlines[1][vlines[1].size() - 1][0];

    hh11 = fabs(vk1* (points[0][0].x - vlines[0][vlines[0].size() - 1][2]) + vlines[0][vlines[0].size() - 1][3] - points[0][0].y) / sqrt(1 + pow(vk1, 2));
    hh22 = fabs(vk2* (points[1][0].x - vlines[1][vlines[1].size() - 1][2]) + vlines[1][vlines[1].size() - 1][3] - points[1][0].y) / sqrt(1 + pow(vk2, 2));

    double vangel = atan(vk1)*180.0 / 3.14159265;
    double vange2 = atan(vk1)*180.0 / 3.14159265;

    float  difference[2] = { 0 };
    vangel*rrc0.angle < 0 ? difference[0] = abs(abs(vangel - rrc0.angle) - 90) : difference[0] = abs(vangel - rrc0.angle);
    vange2*rrc1.angle < 0 ? difference[1] = abs(abs(vange2 - rrc1.angle) - 90) : difference[1] = abs(vange2 - rrc1.angle);


    if (difference[0] > 10) h1 = hh11;
    if (difference[1] > 10) h2 = hh22;
    //cv::resize(img, img, cv::Size(1024, 900));
    auto duration_capture = clock::now() - last;

    auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
    rst.time = ttime;
    rst.gap[0] = h1 * 0.75;
    rst.gap[1] = h2 * 0.75;
    rst.gap_position[0] = rrc0.center.y;
    rst.gap_position[1] = rrc1.center.y;
    cv::resize(img, img, cv::Size(img.cols, 1024));
    cv::resize(palyimg, palyimg, cv::Size(1024, 1024));
    /* cv::imshow("1", palyimg);

     cv::waitKey();*/
    cv::putText(img, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
    cv::putText(img, std::to_string(h1), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(h2), cv::Point(20, 110), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(rst.gap_position[0]), cv::Point(20, 150), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(img, std::to_string(rst.gap_position[1]), cv::Point(20, 190), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));



    cv::putText(palyimg, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
    cv::putText(palyimg, std::to_string(rst.gap[0]), cv::Point(20, 310), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap[1]), cv::Point(20, 350), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap_position[0]), cv::Point(20, 150), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.gap_position[1]), cv::Point(20, 190), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.shield_value[0]), cv::Point(20, 230), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    cv::putText(palyimg, std::to_string(rst.shield_value[1]), cv::Point(20, 270), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    //cv::putText(img, std::to_string(hh11), cv::Point(20, 230), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
    //cv::putText(img, std::to_string(hh22), cv::Point(20, 270), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

    for (int i = 0; i < distances.size(); i++) {

        // cv::putText(img, std::to_string(i)+"==" + std::to_string(distances[i]), cv::Point(20, 70+i*40), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 255, 0));
         //cv::putText(img, std::to_string(i) + "==" + std::to_string(ks[i]), cv::Point(250, 70 + i * 40), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 255, 0));
    }


    //cv::imwrite("G:/datas/test/"+std::to_string((clock::now().time_since_epoch()).count())+".bmp", palyimg);
    cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", palyimg);




    // cv::imshow("img", canyimg);
     //cv::waitKey();


    //img.empty() ? rst.imgtype = feature::none : rst.imgtype = feature::endbend;
    return rst;
#endif
}
result data_process::process_shield_img(unsigned char  *datas, cv::Rect &roi, cv::Mat &playimg, int size, int rails) {
    //1、查找定位点可以将图片y膨胀，消除中间锯齿状2、剪切时掐头去尾10像素点，去掉末端黏连，保证连通域内外分开
    numbers_++;
    result rst;
    if (size < 2048)
        return rst;
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();
    auto image = cv::Mat::Mat(size / 2048, 2048, CV_8UC1, datas);




#include "ys_onnx.h"
    using namespace std;
    using namespace cv;
    using namespace cv::dnn;
    using namespace Ort;

    bool ysOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
        if (_batchSize < 1) _batchSize = 1;
        try
        {
            std::vector<std::string> available_providers = GetAvailableProviders();
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

                //auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);


#endif
            }
            else
            {
                std::cout << "************* Infer model on CPU! *************" << std::endl;
            }
            //
            _OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            std::wstring model_path(modelPath.begin(), modelPath.end());
            _OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
            _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

            Ort::AllocatorWithDefaultOptions allocator;
            //init input
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
            _inputTensorShape = input_tensor_info.GetShape();

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
            _outputMaskNodeDataType = tensor_info_output1.GetElementType(); //the same as output0
            _outputMaskTensorShape = tensor_info_output1.GetShape();
            if (_outputTensorShape[0] == -1)
            {
            	_outputTensorShape[0] = _batchSize;
            	_outputMaskTensorShape[0] = _batchSize;
            }
            if (_outputMaskTensorShape[2] == -1) {
            	size_t ouput_rows = 0;
            	for (int i = 0; i < _strideSize; ++i) {
            		ouput_rows += 3 * (_netWidth / _netStride[i]) * _netHeight / _netStride[i];
            	}
            	_outputTensorShape[1] = ouput_rows;

            	_outputMaskTensorShape[2] = _segHeight;
            	_outputMaskTensorShape[3] = _segWidth;
            }
            warm up
            /**/
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
   
    */
    int ysOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
        outSrcImgs.clear();
        Size input_size = Size(_netWidth, _netHeight);
        for (int i = 0; i < srcImgs.size(); ++i) {
            Mat temp_img = srcImgs[i];
            Vec4d temp_param = { 1,1,0,0 };
            if (temp_img.size() != input_size) {
                Mat borderImg;
                LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
                //cout << borderImg.size() << endl;
                outSrcImgs.push_back(borderImg);
                params.push_back(temp_param);
            }
            else {
                outSrcImgs.push_back(temp_img);
                params.push_back(temp_param);
            }
        }

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
        //preprocessing
        Preprocessing(srcImgs, input_images, params);
        cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

        int64_t input_tensor_length = VectorProduct(_inputTensorShape);
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

        output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
            _inputNodeNames.data(),
            input_tensors.data(),
            _inputNodeNames.size(),
            _outputNodeNames.data(),
            _outputNodeNames.size()
        );

        //post-process

        int net_width = _className.size() + 4 + _segChannels;
        float* all_data = output_tensors[0].GetTensorMutableData<float>();
        _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        _outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
        int mask_protos_length = VectorProduct(mask_protos_shape);
        int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
        for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
            Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
            all_data += one_output_length;
            float* pdata = (float*)output0.data;
            int rows = output0.rows;
            std::vector<int> class_ids;//\BD\E1\B9\FBid\CA\FD\D7\E9
            std::vector<float> confidences;//\BD\E1\B9\FB?\B8\F6id\B6\D4?\D6\C3\D0?\C8\CA\FD\D7\E9
            std::vector<cv::Rect> boxes;//?\B8\F6id\BE\D8\D0ο\F2
            std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
            for (int r = 0; r < rows; ++r) {    //stride
                cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
                Point classIdPoint;
                double max_class_socre;
                minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                max_class_socre = (float)max_class_socre;
                if (max_class_socre >= _classThreshold) {
                    vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
                    picked_proposals.push_back(temp_proto);
                    //rect [x,y,w,h]
                    float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
                    float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
                    float w = pdata[2] / params[img_index][0];  //w
                    float h = pdata[3] / params[img_index][1];  //h
                    int left = MAX(int(x - 0.5 * w + 0.5), 0);
                    int top = MAX(int(y - 0.5 * h + 0.5), 0);
                    class_ids.push_back(classIdPoint.x);
                    confidences.push_back(max_class_socre);
                    boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
                }
                pdata += net_width;//\CF\C2?\D0\D0
            }

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
                temp_mask_proposals.push_back(picked_proposals[idx]);
                temp_output.push_back(result);
            }

            MaskParams mask_params;
            mask_params.params = params[img_index];
            mask_params.srcImgShape = srcImgs[img_index].size();
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


#if CKAI


    if (!is_ysonnx_)return rst;
    std::vector<OutputSeg> output;
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    bool find = yvonnx_.OnnxDetect(image, output);
    std::vector<cv::Scalar> color;

    cv::Rect rects[2];

    for (int i = 0; i < 80; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(0, 0, 255));
    }

    int size_gap = 0;
    if (find) {


        for (int k = 0; k < output.size(); k++) {
            if (output[k].id != 1)continue;


            if (size_gap < 2) {
                rects[size_gap] = output[k].box;
                size_gap++;
            }


        }







        auto duration_capture1 = clock::now() - last;



        float h1, h2;
        h1 = rects[0].height;
        h2 = rects[1].height;



        if (rects[0].y < rects[1].y) {
            auto th = h1;
            h1 = h2;
            h2 = th;

        }






        rst.shield_value[0] = h1 * 0.775;
        rst.shield_value[1] = h2 * 0.775;
        rst.shield_position[0] = rects[0].y;
        rst.shield_position[1] = rects[1].y;
        rst.shield = rects[0].x;

        auto timeuse = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture1).count();
        rst.time = timeuse;
        std::string label = "TimeUse: " + std::to_string(timeuse) + "=" + std::to_string(rst.gap[0]) + "=" + std::to_string(rst.gap[1]);
        //DrawPred(image, output, yvonnx_._className, color);
        cv::rectangle(image, rects[0], cv::Scalar(0, 255, 0));
        cv::rectangle(image, rects[1], cv::Scalar(0, 255, 0));
        cv::putText(image, label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2, 8);
        cv::resize(image, image, cv::Size(1024, 1024));
        cv::imshow("result", image);
        cv::waitKey();

        return  rst;
    }

    else

        return  rst;








#endif




















    cv::Mat img, img2;
    img = image(cv::Rect(150, 0, 700, image.rows)).clone();
    img2 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();
    auto img7 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();
    playimg = image.clone();
    int miny, maxy, minx;
    miny = 0;
    maxy = 0;
    minx = 0;
    cv::medianBlur(img, img, 5);
    cv::threshold(img, img, 20, 255, cv::THRESH_BINARY);
    //show_mat(ui.procesed_label, img);
    auto kk = location_all_yy(img, minx, miny, maxy);
    if (maxy - miny > 1 && maxy - miny < img2.rows&&miny>0 && minx < img2.cols - 150) {
        int widthc = 200;
        int startx = 0;
        startx = minx + 150;

        image.rows > 3000 ? cv::medianBlur(img2, img2, 15) : cv::medianBlur(img2, img2, 3);


        int mean_value = static_cast<int>(cv::mean(img2).val[0]);
        unsigned int gray_value = 20;
        gray_value = mean_value / 2;
        if (gray_value > 20)gray_value = 18;
        //if (image.rows < 3000)gray_value = gray_value / 2;

       // cv::Canny(img2, img2, gray_value, gray_value * 2);
       // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
       // cv::dilate(img2, img2, element);


        img2.cols > widthc + startx ? widthc = 200 : widthc = img2.cols - startx;
        auto img4 = img2(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定
        int maxh, minh;






        auto img6 = img7(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定
        cv::medianBlur(img6, img6, 7);


        cv::Sobel(img6, img6, CV_8U, 1, 0, 5, 1, 0);
        //cv::imwrite("d:/img6c.bmp", img6);
        cv::threshold(img6, img6, 100, 255, cv::THRESH_BINARY);
        //img::threshold_bound(img6, img6, 100, 255);
    /*	cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
        cv::erode(img6, img6, element2);*/
        //cv::imwrite("d:/img7.bmp", img7);
        //cv::imwrite("d:/8c.bmp", img6);
        std::vector<std::vector<cv::Point>>contours;
        std::vector<cv::Vec4i>hierarchy;
        cv::findContours(img6, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
            return c1.size() > c2.size();
            //return cv::contourArea(c1) > cv::contourArea(c2); 
        });
        minh = -1;
        maxh = img6.rows;

        bool minb = false;
        bool maxb = false;



        int lastminx = 0;
        int lastmaxx = 0;
        for (int i = 0; i < contours.size(); i++) {
            auto rdrect = cv::boundingRect(contours[i]);
            auto rdarea = cv::contourArea(contours[i]);
            if (rdrect.height > 400)continue;

            if (rdrect.x > 5 && rdrect.x < 110 && rdrect.width < rdrect.height&&rdarea>300 && rdrect.width>12) {
                if (rdrect.y < img6.rows / 2 && rdrect.y>minh&&rdrect.x - lastminx > 0 && rdrect.y + rdrect.height < 320) {
                    minh = rdrect.y + rdrect.height;
                    lastminx = rdrect.x;
                    //lastminy = rdrect.y + rdrect.height;
                    minb = true;
                }
                if (rdrect.y > img6.rows / 2 && rdrect.y < maxh&&rdrect.x - lastmaxx>0 && img6.rows - rdrect.y < 320) {
                    lastmaxx = rdrect.x;
                    maxh = rdrect.y;
                    maxb = true;
                }


            }


        }

        for (int i = 0; i < contours.size() && i < 10; i++) {

            auto rdrect = cv::boundingRect(contours[i]);
            auto rdarea = cv::contourArea(contours[i]);
            if (rdrect.height > 400)continue;
            if (rdrect.x > 5 && rdrect.x < 110 && rdrect.width < rdrect.height&& rdrect.y + rdrect.height < 320) {
                if (rdrect.y < img6.rows / 2 && !minb&&rdrect.y < 5) {
                    minh = rdrect.y + rdrect.height;
                    minb = true;
                }
                if (rdrect.y > img6.rows / 2 && img6.rows - rdrect.y - rdrect.height < 5 && !maxb&& img6.rows - rdrect.y < 320) {
                    maxh = rdrect.y;
                    maxb = true;
                }


            }


        }
        //cv::cvtColor(img6, img6, CV_GRAY2BGR);







       // location_in_yy(img4, minh, maxh);


        //playimg = img2.clone();

        cv::cvtColor(playimg, playimg, CV_GRAY2BGR);
        cv::rectangle(playimg, cv::Rect(startx, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(playimg, cv::Rect(startx, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));
        cv::putText(playimg, std::to_string((minh + 10)*0.775), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        cv::putText(playimg, std::to_string((maxy - miny - maxh + 10)*0.775), cv::Point(20, 120), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        auto  dstImage = img4.clone();
        cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
        cv::rectangle(dstImage, cv::Rect(0, 0, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(dstImage, cv::Rect(0, maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string((minh + 10)*0.775), cv::Point(130, 70), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string((maxy - miny - maxh + 10)*0.775), cv::Point(130, 120), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::putText(dstImage, std::to_string(numbers_), cv::Point(130, 170), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));

        auto duration_capture = clock::now() - last;
        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        //cv::imwrite("E:/datas/binary/" + std::to_string(numbers_) + ".bmp", dstImage);

        //cv::resize(playimg, playimg, cv::Size(1024, 1024));
        //cv::imshow("1", dstImage);
       // cv::waitKey();


        rst.time = ttime;
        rst.shield_value[0] = abs(minh + 10)*0.775;
        rst.shield_value[1] = abs(maxy - miny - maxh + 10) * 0.775;

        rst.shield_position[0] = miny;
        rst.shield_position[1] = miny + maxh;



        rst.shield = startx;
        rst.width = img4.cols;

        roi.x = minx + 500;
        roi.y = miny;
        roi.width = 1000;
        roi.height = maxy - miny;

        if (!kk) {
            roi.x = 400;
            roi.width = image.cols - 400;//600
            roi.y = 0;
            roi.height = image.rows;
        }
        roi.y = 0;
        roi.height = image.rows;

    }

    return rst;




























#if 0
    if (image.rows > 3000) {
        cv::blur(image, image, cv::Size(3, 3));
    }

    cv::Mat img, img2, canyimg;
    cv::Mat dstImage; //目标图
    cv::Mat normImage; //归一化后的图
    cv::Mat scaledImage; //线性变换后的八位无符号整形的图

    img = image.clone();
    img2 = image(cv::Range::all(), cv::Range(0, image.cols / 2)).clone();



    canyimg = img2.clone();
    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    playimg = image;
    /*  cv::imshow("1", img2);
      cv::waitKey();*/
      //cv::Mat imageGamma;
      ////灰度归一化
      //image_.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);
      ////伽马变换
      //double gamma = 0.7;
      //pow(imageGamma, gamma, img2);//dist 要与imageGamma有相同的数据类型
      //img2.convertTo(img2, CV_8U, 255, 0);
      //show_mat(ui.procesed_label, img2);
      //cv::imshow("1", img2);
      //cv::waitKey();
      /*cv::Rect r1;
      auto himg = img::get_cols_roi(img2, false, r1);
      img2 = himg;*/

      //cv::blur(img2, img2, cv::Size(3, 3));

      /*cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 6));
      cv::erode(img2, img2, element2);*/


    int mean_value = static_cast<int>(cv::mean(img2).val[0]);
    unsigned int gray_value = 20;
    gray_value = mean_value / 2;
    if (gray_value > 20)gray_value = 18;
    cv::Canny(img2, img2, gray_value, gray_value * 2);

    //cv::Canny(canyimg, canyimg, gray_value, gray_value * 2);
 /*   cv::imshow("canyimg", canyimg);
    cv::waitKey();
    cv::imshow("1", img2);
    cv::waitKey();*/

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 15));
    cv::dilate(img2, img2, element);






    std::vector<std::vector<cv::Point>>contours;
    std::vector<cv::Vec4i>hierarchy;
    cv::findContours(img2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
        return c1[0].x < c2[0].x;
        //return cv::contourArea(c1) > cv::contourArea(c2); 
    });
    cv::Rect shieldrect;
    for (int i = 0; i < contours.size(); i++) {
        shieldrect = cv::boundingRect(contours[i]);
        if (shieldrect.width > 50 && shieldrect.height > 500 && shieldrect.x > 100)
            break;
    }
    //

    if (shieldrect.x > 500) {//端部弯头处
        cv::Canny(canyimg, canyimg, gray_value / 2, gray_value);
        cv::dilate(canyimg, canyimg, element);
        img2 = canyimg.clone();


        contours.clear();
        hierarchy.clear();
        cv::findContours(img2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> &c1, std::vector<cv::Point> &c2) {
            return c1[0].x < c2[0].x;
            //return cv::contourArea(c1) > cv::contourArea(c2); 
        });

        for (int i = 0; i < contours.size(); i++) {
            shieldrect = cv::boundingRect(contours[i]);
            if (shieldrect.width > 50 && shieldrect.height > 800 && shieldrect.x > 90)
                break;
        }
    }


    shieldrect.y = 0;
    shieldrect.height = img2.rows;
    auto twidth = shieldrect.width;
    shieldrect.width = 250;
    if (shieldrect.x + shieldrect.width > img2.cols)
        shieldrect.width = twidth;

    auto img3 = img2(shieldrect).clone();

    /* cv::imshow("1", img3);
     cv::waitKey();
 */
    dstImage = img2.clone();
    cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
    cv::drawContours(dstImage, contours, 0, cv::Scalar(0, 255, 0));
    std::vector<cv::Point> shieldpoints;

    for (int y = 0; y < img3.rows; y++) {
        int x;
        const  uchar* dptr = img3.ptr<uchar>(y);

        for (x = 0; x < img3.cols; x++) {
            if (dptr[x] > 0) {
                shieldpoints.push_back(cv::Point(x, y));
                //circle(dstImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                break;
            }
        }
    }



    std::vector<cv::Point3f> pkk;//先获取各点与前两点斜率，然后进行抑制排序，最终选取两点
    if (shieldpoints.size() < 6)return rst;

    for (int i = 0; i < shieldpoints.size() - 6; i++) {
        std::vector<cv::Point> ttpoints;
        ttpoints.assign(shieldpoints.begin() + i, shieldpoints.begin() + i + 5);
        cv::Vec4f line;
        // double sx = shieldpoints[i + 5].x - shieldpoints[i].x;
        // double sy = shieldpoints[i + 5].y - shieldpoints[i].y;
        cv::fitLine(ttpoints, line, cv::DIST_L2, 0, 1e-2, 1e-2);


        double kk = line[1] / line[0];
        cv::Point3f pf;
        pf.x = shieldpoints[i].x;
        pf.y = shieldpoints[i].y;
        pf.z = abs(kk);
        pkk.push_back(pf);


    }
    if (pkk.size() < 6)return rst;
    std::vector<cv::Point3f> pkks_nms;//y距离超过5就认为不在范围内，小于3取最小值
    std::vector<cv::Point3f> pkks_nms1;//

    for (int i = 0; i < pkk.size(); i++) {
        cv::Point3f pf;
        pf = pkk[i];
        int addj = 0;
        int addj2 = 0;
        for (int j = i + 1; j < pkk.size(); j++) {
            if ((pkk[j].y - pf.y) < 100) {
                if (pkk[j].z < pf.z) {
                    pf = pkk[j];
                    addj = j - i;
                }
                addj2 = j - i;
            }

            if (pkk[j].y - pf.y >= 100) {
                break;
            }
        }
        if (pf.z < 1) {
            //pf.z = i + addj;
            pkks_nms1.push_back(pf);

        }
        if (pf.z < 0.6) {

            pf.z = i + addj;
            pkks_nms.push_back(pf);

            circle(dstImage, cv::Point(pf.x, pf.y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
        }


        i = addj2 + i;


    }
    if (pkks_nms.size() < 2)return rst;
    std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.y < pf2.y;
    });

    float miny = 0;
    float maxy = 0;
    float minx, maxx;
    minx = 0;
    maxx = minx;
    bool findok = false;//是否膨胀接头
    float lhdistance = 9999.0;
    int smallline = -9999;
    std::vector<cv::Point2f> lhpoints;//符合条件的所有点
    if (image.rows > 3000) {
        for (int i = 0; i < pkks_nms.size(); i++) {
            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;
            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }


                int endpoint = posekk + 5;
                if (pkk.size() < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0; k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;

                if (pkks_nms[j].y - pkks_nms[i].y < 2000 && pkks_nms[j].y - pkks_nms[i].y>1900) {


                    findok = true;
                    //auto tdistance = abs((pkks_nms[j].y + pkks_nms[i].y - img3.rows) / 2);
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;
                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z; k++) {
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;
                        if (pkk[k].x == 0)zeropoint = true;
                    }

                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0; k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - 1920);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {//先或后与，无论满足哪一条件即可，中位或者大小。但如果两者都满足则优先选择此点
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;

                    }
                    circle(dstImage, cv::Point(pkks_nms[i].x, pkks_nms[i].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    circle(dstImage, cv::Point(pkks_nms[j].x, pkks_nms[j].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;
                    lhpoints.push_back(lhyp);



                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 2000)break;
            }
            //if (findok)break;
        }
    }
    else {
        int yrange = 1000;
        if (rails < 2)yrange = 700;



        for (int i = 0; i < pkks_nms.size() - 1; i++) {

            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;

            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }

                int endpoint = posekk + 5;
                if (pkk.size() - 1 < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0; k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;






                if (pkks_nms[j].y - pkks_nms[i].y < 1200 && pkks_nms[j].y - pkks_nms[i].y>600) {//半包嵌700、全包1000
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;

                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z; k++) {
                        if (pkk[k].x == 0)zeropoint = true;
                        if (pkk[k].x > pkks_nms[i].x)break;
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;

                    }

                    findok = true;
                    /***进一步确定搭接量位置，找到符合条件的上点***/


                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0; k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    //tdistance = matchs;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - yrange);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;

                    }
                    circle(dstImage, cv::Point(pkks_nms[i].x, pkks_nms[i].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    circle(dstImage, cv::Point(pkks_nms[j].x, pkks_nms[j].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;
                    lhpoints.push_back(lhyp);



                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 1200)break;
            }
            //if (findok)break;
        }
    }






    std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.z < pf2.z;
    });

    std::sort(pkks_nms1.begin(), pkks_nms1.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
        return pf1.z < pf2.z;
    });



    if ((pkks_nms.size() < 2 && pkks_nms1.size() > 1) || (!findok&&pkks_nms1.size() > 1)) {

        circle(dstImage, cv::Point(pkks_nms1[0].x, pkks_nms1[0].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
        circle(dstImage, cv::Point(pkks_nms1[1].x, pkks_nms1[1].y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);


        if (pkks_nms1[0].y > pkks_nms1[1].y) {
            miny = pkks_nms1[1].y;
            maxy = pkks_nms1[0].y;
        }
        else {
            miny = pkks_nms1[0].y;
            maxy = pkks_nms1[1].y;
        }
    }



    if (maxy - miny > 1) {
        int widthc = 250;
        int startx = 9999;

        //img3= canyimg(shieldrect).clone()+img3;
        if (minx != maxx) {

            for (int i = 0; i < shieldpoints.size(); i++) {
                if (shieldpoints[i].y > maxy)break;
                if (shieldpoints[i].y > miny - 1 && startx > shieldpoints[i].x) {
                    startx = shieldpoints[i].x;

                }
            }


        }
        else { startx = 0; }

        startx = shieldrect.x + startx;






        img2.cols > widthc + startx ? widthc = 250 : widthc = img2.cols - startx;
        auto img4 = img2(cv::Range(miny, maxy), cv::Range(startx, widthc + startx)).clone();//宽度需要重新界定


        int maxh, minh;
        img::get_shield_rows(img4, maxh, minh);

        cv::rectangle(dstImage, cv::Rect(shieldrect.x, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(dstImage, cv::Rect(shieldrect.x, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));




        auto duration_capture = clock::now() - last;

        auto  ttime = std::chrono::duration_cast<std::chrono::milliseconds>(duration_capture).count();
        rst.time = ttime;
        rst.shield_value[0] = abs(minh - 15)*0.75;
        rst.shield_value[1] = abs(maxy - miny - maxh - 15) * 0.75;
        rst.shield_position[0] = miny;
        rst.shield_position[1] = miny + maxh;

        cv::rectangle(image, cv::Rect(shieldrect.x, miny, img4.cols, minh), cv::Scalar(0, 255, 0));
        cv::rectangle(image, cv::Rect(shieldrect.x, miny + maxh, img4.cols, maxy - miny - maxh), cv::Scalar(0, 255, 0));

        rst.shield = shieldrect.x;
        rst.width = img4.cols;

        roi.x = shieldrect.x + 400;
        roi.y = miny;
        roi.width = 800;
        roi.height = maxy - miny;

        //cv::resize(image, image, cv::Size(1024, 1024));

        cv::putText(image, "ttime==" + std::to_string(ttime), cv::Point(20, 30), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(0, 0, 255));
        cv::putText(image, std::to_string(rst.shield_position[0]), cv::Point(20, 70), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));
        cv::putText(image, std::to_string(rst.shield_position[1]), cv::Point(20, 110), cv::FONT_HERSHEY_PLAIN, 1.7, cv::Scalar(255, 255, 0));

        imshow("1", image);
        cv::waitKey();






    }




    //circle(img2, cv::Point(pkks_nms[0].x, pkks_nms[0].y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);
    //circle(img2, cv::Point(pkks_nms[1].x, pkks_nms[1].y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);


    playimg = image;


    //cv::waitKey(2);
   // cv::imwrite("G:/datas/test/biaoji/" + std::to_string((clock::now().time_since_epoch()).count()) + ".bmp", image);
    return rst;



#endif



}
bool data_process::location_yy(const std::vector<cv::Point3f> &pkks_nms, const std::vector<cv::Point3f> &pkk, const int &rows, int&miny, int&maxy) {
    float minx, maxx;
    minx = 0;
    maxx = minx;
    bool findok = false;//是否膨胀接头
    float lhdistance = 9999.0;
    int smallline = -9999;
    if (pkks_nms.size() < 2 || pkk.size() < 2)return findok;

    if (rows > 3000) {
        for (int i = 0; i < pkks_nms.size(); i++) {
            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;
            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }


                int endpoint = posekk + 5;
                if (pkk.size() < posekk + 6)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0 && k < pkk.size(); k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;

                if (pkks_nms[j].y - pkks_nms[i].y < 2000 && pkks_nms[j].y - pkks_nms[i].y>1900) {



                    //auto tdistance = abs((pkks_nms[j].y + pkks_nms[i].y - img3.rows) / 2);
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;
                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z&& k < pkk.size(); k++) {
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;
                        if (pkk[k].x == 0)zeropoint = true;
                    }

                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0 && k < pkk.size(); k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - 1920);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        findok = true;
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;


                    }

                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;




                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 2000)break;
            }
            //if (findok)break;
        }
    }
    else {
        int yrange = 1000;
        if (2)yrange = 700;



        for (int i = 0; i < pkks_nms.size() - 1; i++) {

            auto smallmatchs = 0;
            for (int k = pkks_nms[i].z; k < pkk.size(); k++) {

                if (k - pkks_nms[i].z > 100)break;
                if (pkk[k].x < pkks_nms[i].x)smallmatchs++;
            }
            if (smallmatchs < 20)continue;

            for (int j = i + 1; j < pkks_nms.size(); j++) {
                auto bigmatchs = 0;
                auto posekk = pkks_nms[j].z;
                int  lowx = pkks_nms[j].x;
                for (int k = posekk; k < pkk.size(); k++) {
                    if (pkk[k].z > 1 || k - posekk > 10)break;
                    lowx = pkk[k].x;

                }

                int endpoint = posekk + 5;
                if (pkk.size() - 1 < posekk + 5)endpoint = pkk.size() - 1;
                lowx = pkk[endpoint].x;
                for (int k = pkks_nms[j].z; k > 0 && k < pkk.size(); k--) {

                    if (pkks_nms[j].z - k > 100)break;
                    if (pkk[k].x < lowx)bigmatchs++;
                }


                if (bigmatchs < 20 || pkk[endpoint].x < pkks_nms[j].x)continue;






                if (pkks_nms[j].y - pkks_nms[i].y < 1200 && pkks_nms[j].y - pkks_nms[i].y>600) {//半包嵌700、全包1000
                    int matchs = 0;
                    int tdistance = 0;
                    int meanvalue = 0;
                    int tennumber = 0;

                    bool zeropoint = false;
                    for (int k = pkks_nms[i].z; k < pkks_nms[j].z&&k < pkk.size(); k++) {
                        if (pkk[k].x == 0)zeropoint = true;
                        if (pkk[k].x > pkks_nms[i].x)break;
                        if (pkk[k].x < pkks_nms[i].x)matchs++;
                        if (tennumber > 4 && tennumber < 11)meanvalue = meanvalue + pkk[k].x;
                        tennumber++;

                    }


                    /***进一步确定搭接量位置，找到符合条件的上点***/


                    matchs = 0;
                    for (int k = pkks_nms[i].z; k > 0 && k < pkk.size(); k--) {
                        if (pkk[k].z > 2)break;
                        matchs++;

                    }
                    for (int k = pkks_nms[i].z; k < pkk.size(); k++) {
                        if (pkk[k].z > 2)break;
                        matchs++;
                    }

                    meanvalue = meanvalue / 6;
                    //tdistance = matchs;
                    tdistance = abs(pkks_nms[j].y - pkks_nms[i].y - yrange);




                    if (lhdistance > tdistance&&pkks_nms[i].z != 0) {
                        findok = true;
                        lhdistance = tdistance;
                        smallline = matchs;
                        miny = pkks_nms[i].y;
                        maxy = pkks_nms[j].y;
                        minx = pkks_nms[i].x;
                        maxx = pkks_nms[j].x;


                    }

                    cv::Point2f lhyp;
                    lhyp.x = pkks_nms[i].y;
                    lhyp.y = pkks_nms[j].y;




                    //break;
                }

                if (pkks_nms[j].y - pkks_nms[i].y > 1200)break;
            }
            //if (findok)break;
        }
    }

    return findok;

}
bool data_process::location_all_yy(const cv::Mat &image, int &minx, int&miny, int&maxy) {
    auto img = image.clone();
    auto  dstImage = image.clone();
    cv::cvtColor(dstImage, dstImage, CV_GRAY2BGR);
    cv::threshold(img, img, 20, 255, cv::THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));
    cv::dilate(img, img, element);
    //cv::imshow("1", img);
    //cv::waitKey();

    std::vector<cv::Point> shieldpoints;
    minx = 0;
    for (int y = 0; y < img.rows; y++) {
        int x;
        const  uchar* dptr = img.ptr<uchar>(y);

        for (x = 0; x < img.cols; x++) {
            if (dptr[x] > 0) {
                shieldpoints.push_back(cv::Point(x, y));
                if (minx < x) {
                    minx = x;
                }
                //circle(dstImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
                break;
            }
        }
    }



    std::vector<cv::Point3f> pkk;//先获取各点与前两点斜率，然后进行抑制排序，最终选取两点
    if (shieldpoints.size() < 6)return false;

    for (int i = 0; i < shieldpoints.size() - 6; i++) {
        std::vector<cv::Point> ttpoints;
        ttpoints.assign(shieldpoints.begin() + i, shieldpoints.begin() + i + 5);
        cv::Vec4f line;
        // double sx = shieldpoints[i + 5].x - shieldpoints[i].x;
        // double sy = shieldpoints[i + 5].y - shieldpoints[i].y;
        cv::fitLine(ttpoints, line, cv::DIST_L2, 0, 1e-2, 1e-2);


        double kk = line[1] / line[0];
        cv::Point3f pf;
        pf.x = shieldpoints[i].x;
        pf.y = shieldpoints[i].y;
        pf.z = abs(kk);
        pkk.push_back(pf);


    }
    if (pkk.size() < 6)return false;
    std::vector<cv::Point3f> pkks_nms;//y距离超过5就认为不在范围内，小于3取最小值
    std::vector<cv::Point3f> pkks_nms1;//

    for (int i = 0; i < pkk.size(); i++) {
        cv::Point3f pf;
        pf = pkk[i];
        int addj = 0;
        int addj2 = 0;
        for (int j = i + 1; j < pkk.size(); j++) {
            if ((pkk[j].y - pf.y) < 100) {
                if (pkk[j].z < pf.z) {
                    pf = pkk[j];
                    addj = j - i;
                }
                addj2 = j - i;
            }

            if (pkk[j].y - pf.y >= 100) {
                break;
            }
        }
        if (pf.z < 2) {
            //if (pf.z > 0.6)
               // circle(dstImage, cv::Point(120, pf.y), 10, cv::Scalar(255, 255, 0), 2, 8, 0);

            cv::Point3f tpf;
            tpf = pf;
            tpf.z = i + addj;
            pkks_nms1.push_back(tpf);




        }
        if (pf.z < 0.6) {

            pf.z = i + addj;
            pkks_nms.push_back(pf);
            //circle(dstImage, cv::Point(120, pf.y), 10, cv::Scalar(0, 255, 0), 2, 8, 0);

        }


        i = addj2 + i;


    }
    if (pkks_nms1.size() < 2)return false;
    if (pkks_nms.size() > 1) {
        std::sort(pkks_nms.begin(), pkks_nms.end(), [](cv::Point3f pf1, cv::Point3f pf2) {
            return pf1.y < pf2.y;
        });
    }


    bool is_location = location_yy(pkks_nms, pkk, img.rows, miny, maxy);

    if (pkks_nms.size() == 2 && !is_location) {
        if (img.rows > 3000 && (pkks_nms[1].y - pkks_nms[0].y < 2000 && pkks_nms[1].y - pkks_nms[0].y>1900)) {

            maxy = pkks_nms[1].y;
            miny = pkks_nms[0].y;
            is_location = true;

        }
        if (img.rows < 3000 && (pkks_nms[1].y - pkks_nms[0].y < 1000 && pkks_nms[1].y - pkks_nms[0].y>600)) {
            maxy = pkks_nms[1].y;
            miny = pkks_nms[0].y;
            is_location = true;
        }

    }


    if (!is_location)is_location = location_yy(pkks_nms1, pkk, img.rows, miny, maxy);



    for (int i = 0; i < shieldpoints.size(); i++) {
        if (shieldpoints[i].y > maxy)break;
        if (shieldpoints[i].y > miny - 1 && minx > shieldpoints[i].x) {
            minx = shieldpoints[i].x;

        }
    }


    //circle(dstImage, cv::Point(150, miny), 20, cv::Scalar(0, 255, 0), 2, 8, 0);
    //circle(dstImage, cv::Point(150, maxy), 20, cv::Scalar(0, 255, 0), 2, 8, 0);

    miny = miny + 2 + 10;
    maxy = maxy - 2 - 10;

    return is_location;
}
void data_process::location_in_yy(const cv::Mat &image, int&minh, int&maxh) {
    int threshhold = 30;
    int numberpoints = 80;
    int endnumber = 1;
    if (image.rows > 1500) {//膨胀接头
        threshhold = 20;
        numberpoints = 20;
        endnumber = 20;
    }


    int minnumber = 9999;
    int maxnumber = -9999;
    int minposition = image.rows / 2;
    std::vector<cv::Point> points;
    std::vector<cv::Point> points120;
    std::vector<uchar> gray;
    std::vector<cv::Point3i> shieldpoints;
    minh = 0;
    maxh = 0;
    for (int y = 0; y < image.rows; y++) {
        int x;
        const  uchar* dptr = image.ptr<uchar>(y);
        cv::Point pf, pfsmall;
        pf.y = -1;
        pf.x = 0;
        pfsmall.y = 0;
        pfsmall.x = 0;
        cv::Point3i spf;
        spf.y = y;
        spf.x = 0;
        spf.z = 0;
        for (x = 0; x < image.cols; x++) {
            if (dptr[x] > 0) {
                if (pf.y == -1) {
                    pf.y = x;
                }
                if (x < 165) {
                    pfsmall.y = x;
                    pfsmall.x++;
                }
                pf.x++;
                spf.x = x;


            }
        }
        spf.z = pf.x;
        points.push_back(pf);
        shieldpoints.push_back(spf);
        points120.push_back(pfsmall);
    }
    if (shieldpoints.size() < 6)return;
    bool isbignumber = false;
    for (int i = 0; i < shieldpoints.size() / 2; i++) {
        if (shieldpoints[i].x > 150)isbignumber = true;
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {
            minh = i;
            break;
        }
    }
    isbignumber = false;
    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x > 150)isbignumber = true;
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {
            maxh = i;
            break;
        }
    }



    isbignumber = false;
    int startp = 0;
    for (int i = 0; i < shieldpoints.size() / 2; i++) {
        if (shieldpoints[i].x > 60) {
            int bignumbers = 0;
            int endbigp = i;
            for (int j = i + 1; j < shieldpoints.size() / 2 && j < i + 70; j++) {
                if (shieldpoints[j].x > 60 && shieldpoints[j].z > 20) {
                    bignumbers++;
                    endbigp = j;
                }
            }
            if (bignumbers < 10 && shieldpoints[i].z > 10) {
                startp = i;
                isbignumber = true;
                minh = i;
            }
            if (bignumbers > 9) {
                i = endbigp;
            }

        }
        if (shieldpoints[i].z < 5 && shieldpoints[i].x - points[i].y < 35 && isbignumber) {

            break;
        }
    }



    isbignumber = false;
    startp = 0;
    maxnumber = -9999;
    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x - points[i].y > 100) {
            int bignumbers = 0;
            int endbigp = i;
            for (int j = i - 1; j > shieldpoints.size() / 2 && j > i - 70; j--) {
                if (shieldpoints[j].x - points[i].y > 100 && shieldpoints[j].z > threshhold) {
                    bignumbers++;
                    endbigp = j;
                }
                if (!isbignumber&&maxnumber < shieldpoints[j].z&&shieldpoints.size() - 1 - j > 30 && shieldpoints.size() - 1 - i > 30) {
                    maxnumber = shieldpoints[j].z;
                    maxh = j;

                }
            }
            if (bignumbers < 30 && shieldpoints[i].z > numberpoints &&  shieldpoints.size() - 1 - i > 5) {

                if (shieldpoints.size() - 1 - i < 30 && bignumbers>9) {
                    i = endbigp - 1;
                    continue;

                }


                startp = i;
                isbignumber = true;
                maxh = i;
                break;
            }
            if (bignumbers > 29 || i == shieldpoints.size() - 1) {//img.rows<3000  20
                i = endbigp - endnumber;
            }

        }
        if (!isbignumber&&maxnumber < shieldpoints[i].z&&shieldpoints.size() - 1 - i > 30) {
            maxnumber = shieldpoints[i].z;
            maxh = i;

        }
    }





#if 1

    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 30) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {

                if (shieldpoints[j].x > 160)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }

            if (maxbool&&numbers < 10 && numbers2 < 10 && i>20) {
                minh = i;
                break;
            }
        }




    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {
        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 30) {

            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int numbers = 0;
            int numbers2 = 0;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }
            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x > 160)numbers++;

            }

            if (minbool&&numbers < 10 && numbers2 < 10 && shieldpoints.size() - 1 - i>20) {
                maxh = i;
                break;
            }




        }


    }




    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            int threshgan = 120;
            //threshgan = threshgan + i * 80 / 600;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y > 40 && i - j > 5 && shieldpoints[j].x - points[j].y < threshgan) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 170)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;
                    break;
                }

            }

            if (minbool&&maxbool&&numbers < 10) {
                minh = i;
                break;
            }




        }
    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int threshgan = 120;
            //threshgan = threshgan + (shieldpoints.size() - 1 - i) * 80 / 600;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;
                    break;
                }

            }
            int numbers = 0;
            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x - points[j].y < threshgan &&points120[j].y - points[j].y > 40 && j - i > 5) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 170)numbers++;

            }

            if (minbool&&maxbool&&numbers < 10) {
                maxh = i;
                break;
            }




        }


    }



#endif














#if 0

    for (int i = 0; i < shieldpoints.size() / 2; i++) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            int threshgan = 120;
            //threshgan = threshgan + i * 80 / 600;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers = 0;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y > 40 && i - j > 5 && shieldpoints[j].x - points[j].y < threshgan) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers++;

            }
            for (int j = i; j < maxi; j++) {
                if (points120[j].y - points[j].y < 40) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;
            }

            if (minbool&&maxbool&&numbers < 10 && numbers2 < 10) {
                minh = i;
                break;
            }




        }
    }

    for (int i = shieldpoints.size() - 1; i > shieldpoints.size() / 2; i--) {

        if (shieldpoints[i].x - points[i].y > 100 && shieldpoints[i].z > 50) {
            int mini = 0;
            int maxi = 0;
            bool minbool = false;
            bool maxbool = false;
            int threshgan = 120;
            //threshgan = threshgan + (shieldpoints.size() - 1 - i) * 80 / 600;
            i - 20 < 0 ? mini = 0 : mini = i - 20;
            i + 20 > shieldpoints.size() ? maxi = shieldpoints.size() : maxi = i + 20;
            int numbers2 = 0;
            for (int j = mini; j < i; j++) {
                if (points120[j].y - points[j].y < 40) {
                    minbool = true;

                }
                if (shieldpoints[j].x > 160)numbers2++;

            }
            int numbers = 0;

            for (int j = i; j < maxi; j++) {
                if (shieldpoints[j].x - points[j].y < threshgan &&points120[j].y - points[j].y > 40 && j - i > 5) {
                    maxbool = true;

                }
                if (shieldpoints[j].x > 160)numbers++;

            }

            if (minbool&&maxbool&&numbers < 10 && numbers2 < 10) {
                maxh = i;
                break;
            }




        }


    }

#endif












}
bool data_process::set_params(float  width, float height, float  realwidth, float real_height) {


    std::ofstream of("60.txt");
    float wvalue = realwidth - width;
    float hvalue = real_height - height;
    params_[2] = params_[2] + wvalue;
    params_[3] = params_[3] - hvalue;

    for (int i = 0; i < 8; i++) {
        of << params_[i] << std::endl;

    }
    of.close();

    return true;
}