#include "img.h"



cv::Mat img::get_sleeper_img(const cv::Mat & image, cv::Mat &sleeper_image, int &position){
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    cv::Mat threshold_img;
    cv::threshold(image, threshold_img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = 0;
    int continuation_position = 0;
    int value = image.cols / 4;
    for (int i = 0; i < threshold_img.rows; i++){

        int count = 0;
        const uchar* ptr = threshold_img.ptr<uchar>(i);
        for (int h = 0; h < threshold_img.cols; h++){
            if (ptr[h] == 0){
                count++;
            }

        }

        if (count > value){
         
            if (max_contiuation < continuation){
                max_contiuation = continuation;
                continuation_position = i;

            }
            continuation = 0;

        }
        else{
            position = i;
            continuation++;
        }
        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));

    }
    if (max_contiuation < continuation){
        max_contiuation = continuation;
        position = position - max_contiuation;
    }
    else{ position = continuation_position - max_contiuation; }

    sleeper_image = image(cv::Rect(0, position, image.cols, max_contiuation));
    return hist_img;

}


/**
* @brief 带通二值化
* @params [in] src   输入图像
* @params [out] dst   输出图像
* @params [in] lbound  低阈值
* @params [in] ubound 高阈值
*/
void img::threshold_bound(const cv::Mat& src, cv::Mat& dst, uchar lbound, uchar ubound)
{
    cv::Mat dstTempImage1, dstTempImage2;
    // 小阈值对源灰度图像进行阈值化操作
    cv::threshold(src, dstTempImage1,
        lbound, 255, cv::THRESH_BINARY);
    // 大阈值对源灰度图像进行阈值化操作
    cv::threshold(src, dstTempImage2,
        ubound, 255, cv::THRESH_BINARY_INV);
 
    cv::bitwise_and(dstTempImage1, dstTempImage2, dst);

    /*uchar tab[256];
    for (int i = 0; i < 256; ++i)
    tab[i] = i >= lbound && i <= ubound ? 255 : 0;

    cv::Size size = src.size();
    dst.create(size, CV_8UC1);
    if (src.isContinuous() && dst.isContinuous()) {
    size.width *= size.height;
    size.height = 1;
    }

    for (int y = 0; y < size.height; y++) {
    int x;
    uchar* dptr = dst.ptr<uchar>(y);
    const uchar* sptr = src.ptr<uchar>(y);
    for (x = 0; x <= size.width; x += 4) {
    dptr[x] = tab[sptr[x]];
    dptr[x + 1] = tab[sptr[x + 1]];
    dptr[x + 2] = tab[sptr[x + 2]];
    dptr[x + 3] = tab[sptr[x + 3]];
    }
    for (; x < size.width; x++)
    dptr[x] = tab[sptr[x]];
    }*/
}



void img::zero_image(cv::Mat &image){
    cv::Size size = image.size();
    for (int y = 0; y < size.height; y++) {
        cv::Vec3b* dptr = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < size.width; x ++) {
            dptr[x][0] = 0;
            dptr[x][1] = 0;
            dptr[x][2] = 0;
            
        }
    }
}





cv::Mat img::threshold_bound(const cv::Mat& src, uchar lbound, uchar ubound)
{
    uchar tab[256];
    cv::Mat dst;
    for (int i = 0; i < 256; ++i)
        tab[i] = i >= lbound && i <= ubound ? 255 : 0;

    cv::Size size = src.size();
    dst.create(size, CV_8UC1);
    if (src.isContinuous() && dst.isContinuous()) {
        size.width *= size.height;
        size.height = 1;
    }

    for (int y = 0; y < size.height; y++) {
        int x;
        uchar* dptr = dst.ptr<uchar>(y);
        const uchar* sptr = src.ptr<uchar>(y);
        for (x = 0; x <= size.width; x += 4) {
            dptr[x] = tab[sptr[x]];
            dptr[x + 1] = tab[sptr[x + 1]];
            dptr[x + 2] = tab[sptr[x + 2]];
            dptr[x + 3] = tab[sptr[x + 3]];
        }
        for (; x < size.width; x++)
            dptr[x] = tab[sptr[x]];
    }
    return dst;
}




cv::Mat img::get_rail_img(const cv::Mat & image, cv::Mat &rail_img, cv::Rect &position){
    cv::Mat hist_img(image.rows, image.cols, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    cv::Mat threshold_img;
    cv::threshold(image, threshold_img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position.x = 0;
    position.y = 0;
    int threshvalue = threshold_img.rows / 2;


    int continuation_position = 0;
    int state = 0;
    for (int i = 0; i < threshold_img.cols; i++){
        cv::Mat colmat = threshold_img(cv::Range::all(), cv::Range(i, i + 1));
        int  count_zero = threshold_img.rows - cv::countNonZero(colmat);
        if (count_zero > threshvalue){

            if (max_contiuation < continuation){
                max_contiuation = continuation;
                continuation_position = i;

            }
            continuation = 0;

        }
        else{
            position.x = i;
            continuation++;
        }

        //cv::line(hist_img, cv::Point(i, threshold_img.rows), cv::Point(i, threshold_img.rows - count_zero), cv::Scalar(0, 0, 0));




    }
    if (max_contiuation < continuation){
        max_contiuation = continuation;
        position.x = position.x - max_contiuation;
    }
    else{ position.x = continuation_position - max_contiuation; }
    position.y = 0;
    position.width = max_contiuation;
    position.height = image.rows;

    rail_img = image(cv::Rect(position.x, 0, max_contiuation, image.rows));
    return hist_img;
}

cv::Mat img::get_cols_histogram(const cv::Mat & image, int &position1,int &position2){
    cv::Mat hist_img(image.rows, image.cols, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    position1 = image.cols;
    position2 = 0;
    int temp_position = 0;
    int temp_position2= 0;

    int position_low = image.cols;
    int continuation_position = 0;
    int state = 0;
    for (int i = 0; i < image.cols - 5; i++){
        cv::Mat colmat = image(cv::Range::all(), cv::Range(i, i+1));
        int  count_zero = image.rows - cv::countNonZero(colmat);
        if ( count_zero > 60 && state == 0){
            position2 = i;
            state++;

        }
        cv::Mat rectimg = image(cv::Rect(i, 0, 5, 5));
        cv::Mat rectimg2 = image(cv::Rect(i, image.rows-5, 5, 5));
        cv::Scalar smean = cv::mean(rectimg);
        cv::Scalar smean2 = cv::mean(rectimg2);

        if (smean2.val[0] < 10 && continuation == 0){
            position_low = i;
            cv::line(hist_img, cv::Point(i, image.rows), cv::Point(i, image.rows - count_zero), cv::Scalar(0, 0, 255));
            continuation++;
        }

        if (smean.val[0]<10 && state == 1){
            position1 = i;
            cv::line(hist_img, cv::Point(i, image.rows), cv::Point(i, image.rows - count_zero), cv::Scalar(0, 0, 255));
            state++;
        }
        else{
            cv::line(hist_img, cv::Point(i, image.rows), cv::Point(i, image.rows - count_zero), cv::Scalar(0, 0, 0));
        }

    

    }
    temp_position = position1;
    temp_position2 = position2;

    if (position1 - position2 < 270){
        position1 = position2 + 295;
        
        position2 = -1;
        return hist_img;

    }
    if (position1==image.cols){
        position1 = temp_position2 + 295;
    }

    if (position_low - temp_position2 < 270){
    
        position2 = -2;

    }
    return hist_img;
}
cv::Mat img::get_binary_histogram(const cv::Mat & image, int &position){
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = 0;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++){

        int count = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++){
            if (ptr[h] == 0){
                count++;
            }

        }

        if (count > 60){

            if (max_contiuation < continuation){
                max_contiuation = continuation;
                continuation_position = i;

            }
            continuation = 0;

        }
        else{
            position = i;
            continuation++;
        }
        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));
 
    }
    if (max_contiuation < continuation){
        max_contiuation = continuation;
        position = position - max_contiuation / 2;
    }
    else{ position = continuation_position - max_contiuation / 2; }

    return hist_img;
}
void img::get_next_minloc(cv::Mat &result, cv::Point minloc, int maxvalue, int templatw, int templath){
    // 先将第一个最小值点附近两倍模板宽度和高度的都设置为最大值防止产生干扰  
    int startX = minloc.x - templatw / 2;
    int startY = minloc.y - templath / 2;
    int endX = minloc.x + templatw / 2;
    int endY = minloc.y + templath / 2;
    if (startX < 0 || startY < 0)
    {
        startX = 0;
        startY = 0;
    }
    if (endX > result.cols || endY > result.rows)
    {
        endX = result.cols;
        endY = result.rows;
    }
    int y, x;
    for (y = startY; y < endY; y++)
    {
        float* ptr = result.ptr<float>(y);
        for (x = startX; x < endX; x++)
        {
            ptr[x] = static_cast<float>(maxvalue);

        }
    }
}
cv::Mat img::get_histogram_image(const cv::Mat &image, int width){
    int channel[1];
    int hist_size[1];
    const float* ranges[1];
    float hrange[2];
    channel[0] = 0;
    hist_size[0] = 256;
    hrange[0] = 0;
    hrange[1] = 255.0;
    ranges[0] = hrange;
    cv::MatND hist;
    cv::calcHist(&image, 1, channel, cv::Mat(), hist, 1, hist_size, ranges);
    double max_value, min_value;
    cv::minMaxLoc(hist, &min_value, &max_value, 0, 0);
    cv::Mat hist_img(width, hist_size[0], CV_8U, cv::Scalar(255));
    int hpt = static_cast<int>(0.9*hist_size[0]);
    for (int h = 0; h < width; h++){
        float binvalue = hist.at<float>(h);
        int intensity = static_cast<int>(binvalue*hpt / max_value);
        cv::line(hist_img, cv::Point(h, hist_size[0]), cv::Point(h, hist_size[0] - intensity), cv::Scalar::all(0));

    }
    return hist_img;


    
}
cv::Mat img::get_gray_histogram(const cv::Mat & image, int &position){
    cv::Mat hist_img(255, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = 0;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++){
        int maxvalue = 0;
        int minvalue = 255;
        int difvalue = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++){
            if (ptr[h] < minvalue){
                minvalue = ptr[h];
            }
            if (ptr[h] > maxvalue){
                maxvalue = ptr[h];
            }
        }
        difvalue = maxvalue - minvalue;
        if (difvalue < 200){
            if (max_contiuation < continuation){
                max_contiuation = continuation;
                continuation_position = i;

            }
            continuation = 0;
            cv::line(hist_img, cv::Point(i, 255), cv::Point(i, 255 - difvalue), cv::Scalar::all(0));

        }
        else{
            if (max_difvalue < difvalue){
                max_difvalue = difvalue;
                position = i;
            }

            continuation++;
            cv::line(hist_img, cv::Point(i, 255), cv::Point(i, 255 - difvalue), cv::Scalar(0, 0, 255));

        }


    }
    position = continuation_position - max_contiuation / 2;
    max_contiuation < continuation ? max_contiuation = continuation : 0;


    return hist_img;
}
void img::get_templet_position(const cv::Mat &image, const cv::Mat &tempat_image, int &position){
    cv::Mat result;
    cv::matchTemplate(image, tempat_image, result, cv::TM_SQDIFF_NORMED);
    double new_minVaule, new_maxValue;
    cv::Point new_minLoc, new_maxLoc;
    cv::minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc);
    new_minLoc.y - 45>0 ? position = new_minLoc.y - 45 : position = 0;
}
void img::get_templet_position(const cv::Mat &image, const cv::Mat &tempat_image, cv::Point &position1, cv::Point &position2,std::vector<cv::Mat> &images){

    cv::Mat result, merge_img, left_img, right_img;
    merge_img = image.clone();
    right_img = merge_img(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));
    cv::flip(right_img, right_img, 1);

    cv::matchTemplate(merge_img, tempat_image, result, cv::TM_CCOEFF_NORMED);
    double new_minVaule, new_maxValue;
    cv::Point new_minLoc, new_maxLoc;
    cv::minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc);
    position1 = new_maxLoc;
   // new_minLoc.y - 45>0 ? position1.y = new_minLoc.y - 45 : position1.y = 0;
    cv::Rect rect1;
    rect1.x = new_maxLoc.x;
    rect1.y = position1.y;
    
    rect1.x + 260 > merge_img.cols ? rect1.width = merge_img.cols - rect1.x : rect1.width = 260;
    rect1.y + 230 > merge_img.rows ? rect1.height = merge_img.rows - rect1.y : rect1.height = 230;
    images.push_back(merge_img(rect1));

    if (rect1.x > image.cols / 2){
        position1.x = image.cols * 3 / 2 - new_maxLoc.x - 260;
        position2.x = position1.x - 630;
        position2.y = position1.y;


    }
    else{
        position1.x = new_maxLoc.x;
        position2.x = position1.x + 630;
        position2.y = position1.y;



    }
    cv::Rect rect2;
    rect2 = rect1;
    rect2.x = position2.x;
  
    if (position2.x < merge_img.cols&&position2.x>0){
        rect2.x + 260 > merge_img.cols ? rect2.width = merge_img.cols - rect2.x : rect2.width = 260;
        images.push_back(merge_img(rect2));
     
    }

   
    





#if 0




    new_minLoc.y - 45>0 ? position1.y = new_minLoc.y - 45 : position1.y = 0;
    new_minLoc.x > image.cols / 2 ? position1.x = image.cols * 3 / 2 - new_minLoc.x - 260 : position1.x = new_minLoc.x;
    get_next_minloc(result, new_minLoc, new_maxValue, tempat_image.cols, tempat_image.rows);
    cv::minMaxLoc(result, &new_minVaule, &new_maxValue, &new_minLoc, &new_maxLoc);

    rect1.x = new_minLoc.x;
    rect1.y = new_minLoc.y;
    rect1.width = 260;

    rect1.y + 230 > merge_img.rows ? rect1.height = merge_img.rows - rect1.y : rect1.height = 230;
    images.push_back(merge_img(rect1));

    new_minLoc.y - 45>0 ? position2.y = new_minLoc.y - 45 : position2.y = 0;
    if(abs(position1.y - position2.y) > 10) {
        position2.y = position1.y;
        position1.x > image.cols / 2 ? position2.x = position1.x - 630 : position2.x = position1.x + 630;

    } 
    else { new_minLoc.x > image.cols / 2 ? position2.x = image.cols * 3 / 2 - new_minLoc.x - 260 : position2.x = new_minLoc.x; }
#endif

}
cv::Rect img::get_combine_rect(const cv::Rect &rect1, const cv::Rect &rect2){
    int xmin, xmax, ymin, ymax;
    xmin = rect1.x;
    if (xmin > rect2.x)xmin = rect2.x;
    xmax = rect1.width + rect1.x;
    if (xmax < (rect2.width + rect2.x))xmax = rect2.width + rect2.x;
    ymin = rect1.y;
    if (ymin > rect2.y)ymin = rect2.y;
    ymax = rect1.y + rect1.height;
    if (ymax < (rect2.y + rect2.height))ymax = rect2.y + rect2.height;
    cv::Rect rect_dress = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    return rect_dress;
}
void img::delete_image_line(cv::Mat &image){

    for (int i = 0; i < image.rows; i++){
        int sum = 0;
        uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++){
            sum += ptr[h];
        }
        if (sum>image.cols*128){
            for (int h = 0; h < image.cols; h++){
                ptr[h] = 0;
            }

        }

    }


}
cv::Mat img::get_fastener_position(const cv::Mat & image, int &position) {
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = -1;
    bool is_bigcount = false;
    bool is_fastener = false;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++) {

        int count = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++) {
            if (ptr[h] == 0) {
                count++;
            }

        }

        if (count<100) {
            if (continuation > 5 && !is_bigcount)
                position = i;
            if (continuation > 5 && is_bigcount&&i-position < 250)
                is_fastener = true;
            //if (continuation > 5 && is_bigcount&&max_difvalue > 400) {
            //    position = i;
            //    is_bigcount = false;
            //    max_difvalue = 0;
            //}

            continuation++;
            max_contiuation = 0;
           // continue;

        }
      else if (count > 250) {
          if (max_contiuation > 100 && position > -1)
              is_bigcount = true;

          if (max_difvalue < max_contiuation)
              max_difvalue = max_contiuation;
          max_contiuation++;
          continuation = 0;
          //continue;

        }
      else {

            max_contiuation = 0;
            continuation = 0;
        }











#if 0
        if (count > 200) {

            //if (continuation > 100) {
    
            //    position = i;
            //    

            //}
            continuation ++;

        }
        else {
            if (continuation > 100){
                position = i - continuation;
                break;
            }
                
            continuation = 0;
        }
#endif


        
        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));

    }
    //if (max_contiuation < continuation) {
    //    max_contiuation = continuation;
    //    position = position - max_contiuation / 2;
    //}
    //else { position = continuation_position - max_contiuation / 2; }

    if (!is_fastener || max_difvalue > 400) {
        position = -1;
    }


    return hist_img;
}

cv::Mat img::get_fastener_position2(const cv::Mat & image, int &position) {
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = -1;
    bool is_bigcount = true;
    bool is_fastener = false;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++) {

        int count = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++) {
            if (ptr[h] == 0) {
                count++;
            }

        }

        if (count <130) {
            if(position==-1) 
                position = i;
            if (continuation > 180)
                is_fastener = true;
   

            continuation++;
            if (max_difvalue < continuation)
                max_difvalue = continuation;
            // continue;

        }
        else {
            if (continuation > 330|| max_difvalue < 180) {//226、186
                is_fastener = false;
                max_difvalue = 0;
                position = -1;

            }
                
   
            continuation = 0;
        }


        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));
       // cv::putText(hist_img,  std::to_string(count), cv::Point(i, image.cols - count), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 0));

    }


    if (!is_fastener || continuation > 330 || position == 0) {
        position = -1;
    }
    else {
        position = position + (max_difvalue - 210) / 2;
    }
    if (position + 250 > image.rows)
        position = -1;
    return hist_img;
}
cv::Mat img::adjust_fastener_position(const cv::Mat & image, int &position) {
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    position = -1;
    bool is_bigcount = true;
    bool is_fastener = false;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++) {

        int count = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++) {
            if (ptr[h] == 0) {
                count++;
            }

        }

        if (count < 60) {
            if (position == -1)
                position = i;
            if (continuation > 8)
                is_fastener = true;


            continuation++;
            if (max_difvalue < continuation)
                max_difvalue = continuation;
            // continue;

        }
        else {
            if (continuation > 60 || max_difvalue < 5) {//226、186
                is_fastener = false;
                max_difvalue = 0;
                position = -1;

            }


            continuation = 0;
        }


        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));

    }


    if (!is_fastener || max_difvalue > 60) {
        position = 0;
    }
 else {
        if (position == 0)
            position = max_difvalue;

        else if ((position + max_difvalue) == image.rows)
            position = -max_difvalue;
        else position = 0;

       
    }
    return hist_img;
}

cv::Mat img::get_fastener_positions(const cv::Mat & image, std::vector<int>&positions) {
    cv::Mat hist_img(image.cols, image.rows, CV_8U, cv::Scalar(255));
    cv::cvtColor(hist_img, hist_img, CV_GRAY2BGR);//结果图片为彩色
    int continuation = 0;
    int max_contiuation = 0;
    int max_difvalue = 0;
    std::vector<int>continuations;
    bool is_bigcount = true;
    bool is_fastener = false;
    int continuation_position = 0;
    for (int i = 0; i < image.rows; i++) {

        int count = 0;
        const uchar* ptr = image.ptr<uchar>(i);
        for (int h = 0; h < image.cols; h++) {
            if (ptr[h] == 0) {
                count++;
            }

        }

        if (count < 100) {
            continuation++;
            if (max_difvalue < continuation)
                max_difvalue = continuation;

        }
        else {
            if (continuation < 330 && continuation>100) {
                if (i - continuation == 0 && continuation < 100) {
                    continuation = 0;
                    continue;
                }
                continuation > 260 ? positions.push_back(i - (continuation + 210) / 2) : positions.push_back(i - continuation);
                
                continuations.push_back(continuation);
               //cv::putText(hist_img, "continuation=" + std::to_string(continuation), cv::Point(50, 15*(2* positions.size()+1)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 0));
                //cv::putText(hist_img, "position=" + std::to_string(i - continuation), cv::Point(50, 15 * (2 * positions.size() + 2)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 0));

            }


            continuation = 0;
        }


        cv::line(hist_img, cv::Point(i, image.cols), cv::Point(i, image.cols - count), cv::Scalar::all(0));
        // if(count>120)
         // cv::putText(hist_img,  std::to_string(count), cv::Point(i, image.cols - count), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 0));

    }



    return hist_img;
}
// 骨架化函数
cv::Mat img::skeletonize(const cv::Mat& src) {
    cv::Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    cv::Mat img = src.clone();

    bool done;
    do {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);

        done = (cv::countNonZero(img) == 0);
    } while (!done);

    return skel;
}

void img::calculate_crack_dimensions(const cv::Mat& mask, const cv::Mat& skeleton,float &length,float &width) {

    if (skeleton.empty()) {
      
        return;
    }



    // 计算裂纹长度
    length = cv::countNonZero(skeleton);

    // 距离变换
    cv::Mat dist_transform;
    cv::distanceTransform(mask, dist_transform, cv::DIST_L2, 5);//（通过像素的灰度值表示点到边缘（0）的距离,

    // 计算裂纹宽度
    double minVal, maxVal;
    cv::minMaxLoc(dist_transform, &minVal, &maxVal);
    double maxWidth = maxVal * 2;  // 最大距离乘以2即为最大宽度

    // 计算平均宽度
    cv::Mat skeleton_dist;
    dist_transform.copyTo(skeleton_dist, skeleton);  // 只保留骨架上的距离值（灰度像素）
    double meanWidth = cv::mean(skeleton_dist, skeleton)[0] * 2;  // 骨架（像素宽度为1）上的距离（灰度值）求平均，即是该裂隙宽度中心到边缘的平均值，再*2就是宽度
   

        // 计算不为0的最小宽度
    double minWidth = std::numeric_limits<double>::max();
    for (int y = 0; y < skeleton_dist.rows; ++y) {
        for (int x = 0; x < skeleton_dist.cols; ++x) {
            double val = skeleton_dist.at<float>(y, x);
            if (val > 0 && val * 2 < minWidth) {
                minWidth = val * 2;
            }
        }
    }

    if (minWidth == std::numeric_limits<double>::max()) {
        minWidth = 0;  // 如果没有找到非零的最小值，则设置为0
    }

     width = meanWidth;  // 最大距离乘以2即为宽度

    
}
/*
void img::calculate_crack_dimensions(const cv::Mat& mask, const cv::Mat& skeleton, float &length, float &width) {
    if (skeleton.empty()) {
        return;
    }

    // 计算裂纹长度
    length = cv::countNonZero(skeleton);

    // 距离变换
    cv::Mat dist_transform;
    cv::distanceTransform(mask, dist_transform, cv::DIST_L2, 5);

    // 使用亚像素精度计算裂纹宽度
    cv::Mat skeleton_dist;
    dist_transform.copyTo(skeleton_dist, skeleton);  // 只保留骨架上的距离值

    // 计算平均宽度
    double meanWidth = cv::mean(skeleton_dist, skeleton)[0] * 2;  // 平均距离乘以2即为平均宽度

    // 计算不为0的最小宽度
    double minWidth = std::numeric_limits<double>::max();
    for (int y = 0; y < skeleton_dist.rows; ++y) {
        for (int x = 0; x < skeleton_dist.cols; ++x) {
            float val = skeleton_dist.at<float>(y, x);
            if (val > 0 && val * 2 < minWidth) {
                minWidth = val * 2;
            }
        }
    }

    if (minWidth == std::numeric_limits<double>::max()) {
        minWidth = 0;  // 如果没有找到非零的最小值，则设置为0
    }

    // 使用亚像素精度计算最大宽度
    double maxWidth = 0;
    for (int y = 0; y < dist_transform.rows; ++y) {
        for (int x = 0; x < dist_transform.cols; ++x) {
            float val = dist_transform.at<float>(y, x);
            if (val > maxWidth) {
                maxWidth = val;
            }
        }
    }
    maxWidth *= 2;  // 最大距离乘以2即为最大宽度

    width = meanWidth;  // 使用平均宽度作为最终宽度
}
double calculateCrackWidth(const Mat& image) {
    // 二值化图像
    Mat binary;
    threshold(image, binary, 128, 255, THRESH_BINARY_INV);

    // 获取骨架图像
    Mat skeleton = getSkeleton(binary.clone());

    // 查找骨架图像中的非零点
    vector<Point> skeletonPoints;
    findNonZero(skeleton, skeletonPoints);

    // 计算裂缝宽度
    double totalWidth = 0.0;
    int count = 0;
    for (const Point& p : skeletonPoints) {
        // 获取骨架点的邻域
        Rect roi(p.x - 1, p.y - 1, 3, 3);
        roi &= Rect(0, 0, binary.cols, binary.rows); // 确保ROI在图像范围内

        // 计算邻域内的非零点数
        Mat neighborhood = binary(roi);
        int nonZeroCount = countNonZero(neighborhood);

        // 裂缝宽度为邻域内非零点数
        totalWidth += nonZeroCount;
        count++;
    }

    double averageWidth = totalWidth / count;
    return averageWidth;
}

*/
double calculateCrackWidth(const cv::Mat& image) {
    // 二值化图像
    cv::Mat binary;
    cv::threshold(image, binary, 128, 255, cv::THRESH_BINARY_INV);

    // 获取骨架图像
    cv::Mat skeleton = img::skeletonize(binary.clone());

    // 查找骨架图像中的非零点
    std::vector<cv::Point> skeletonPoints;
    cv::findNonZero(skeleton, skeletonPoints);

    // 计算裂缝宽度
    double totalWidth = 0.0;
    int count = 0;
    for (const cv::Point& p : skeletonPoints) {
        // 获取骨架点的邻域
        cv::Rect roi(p.x - 1, p.y - 1, 3, 3);
        roi &= cv::Rect(0, 0, binary.cols, binary.rows); // 确保ROI在图像范围内

        // 计算邻域内的非零点数
        cv::Mat neighborhood = binary(roi);
        int nonZeroCount = cv::countNonZero(neighborhood);

        // 裂缝宽度为邻域内非零点数
        totalWidth += nonZeroCount;
        count++;
    }

    double averageWidth = totalWidth / count;
    return averageWidth;
}
cv::Mat img::removeSmallComponents(const cv::Mat& image, int minSize) {
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(image, labels, stats, centroids);

    cv::Mat cleanedImage = cv::Mat::zeros(image.size(), CV_8UC1);

    for (int i = 1; i < numComponents; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minSize) {
            cleanedImage.setTo(255, labels == i);
        }
    }

    return cleanedImage;
}

bool img::isValid(int x, int y, int rows, int cols) {
    return x >= 0 && x < rows && y >= 0 && y < cols;
}

int img::countNonZeroNeighbors(const cv::Mat& img, int x, int y) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i != 0 || j != 0) {
                if (isValid(x + i, y + j, img.rows, img.cols) && img.at<uchar>(x + i, y + j) > 0) {
                    count++;
                }
            }
        }
    }
    return count;
}

int img::countTransitions(const cv::Mat& img, int x, int y) {
    int transitions = 0;
    std::vector<std::pair<int, int>> neighbors = {
        {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}
    };
    for (size_t i = 0; i < neighbors.size() - 1; ++i) {
        int x1 = x + neighbors[i].first;
        int y1 = y + neighbors[i].second;
        int x2 = x + neighbors[i + 1].first;
        int y2 = y + neighbors[i + 1].second;
        if (isValid(x1, y1, img.rows, img.cols) && isValid(x2, y2, img.rows, img.cols) &&
            img.at<uchar>(x1, y1) == 0 && img.at<uchar>(x2, y2) > 0) {
            transitions++;
        }
    }
    return transitions;
}

cv::Mat img::zhangSuenThinning(const cv::Mat& binaryImage) {
    cv::Mat img = binaryImage.clone();
    img /= 255; // Convert to binary (0, 1)
    bool hasChanged;
    do {
        hasChanged = false;
        std::vector<std::pair<int, int>> toRemove;
        for (int i = 1; i < img.rows - 1; ++i) {
            for (int j = 1; j < img.cols - 1; ++j) {
                if (img.at<uchar>(i, j) == 1) {
                    int neighbors = countNonZeroNeighbors(img, i, j);
                    int transitions = countTransitions(img, i, j);
                    if (neighbors >= 2 && neighbors <= 6 && transitions == 1 &&
                        img.at<uchar>(i - 1, j) * img.at<uchar>(i, j + 1) * img.at<uchar>(i + 1, j) == 0 &&
                        img.at<uchar>(i, j + 1) * img.at<uchar>(i + 1, j) * img.at<uchar>(i, j - 1) == 0) {
                        toRemove.push_back({ i, j });
                    }
                }
            }
        }
        for (const auto& p : toRemove) {
            img.at<uchar>(p.first, p.second) = 0;
            hasChanged = true;
        }
        toRemove.clear();
        for (int i = 1; i < img.rows - 1; ++i) {
            for (int j = 1; j < img.cols - 1; ++j) {
                if (img.at<uchar>(i, j) == 1) {
                    int neighbors = countNonZeroNeighbors(img, i, j);
                    int transitions = countTransitions(img, i, j);
                    if (neighbors >= 2 && neighbors <= 6 && transitions == 1 &&
                        img.at<uchar>(i - 1, j) * img.at<uchar>(i, j + 1) * img.at<uchar>(i, j - 1) == 0 &&
                        img.at<uchar>(i - 1, j) * img.at<uchar>(i + 1, j) * img.at<uchar>(i, j - 1) == 0) {
                        toRemove.push_back({ i, j });
                    }
                }
            }
        }
        for (const auto& p : toRemove) {
            img.at<uchar>(p.first, p.second) = 0;
            hasChanged = true;
        }
    } while (hasChanged);
    img *= 255; // Convert back to 0, 255
    return img;
}
