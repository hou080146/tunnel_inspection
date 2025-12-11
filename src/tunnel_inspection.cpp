#include "tunnel_inspection.h"
#include <QFileDialog>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include<thread>
#include<qtextcodec.h>

#include<QDebug>
#include "AppConfig.h"
#pragma execution_character_set("utf-8")
void tunnel_inspection::create_word_title(std::string files_name) 
{/*
里程类型:0
线名:4
上行下行:上行
操作员:1
里程加减:-
日期:2014/12/31/16/26/17
当前里程:30.459400
小车方向:3.000000
当前脉冲:0
轨道类型:60
*/

    std::string linename = "4_up_-_ck";// 固定线路名，用于生成文件名等









    // 获取结果文件夹路径（GB2312编码转码）
    QTextCodec *code = QTextCodec::codecForName("GB2312");
    std::string result_files_name = code->fromUnicode(ui->lineSaveResultPath->text()).data();

    using clock = std::chrono::high_resolution_clock;
    clock::time_point now, last;
    clock::duration duration_alg, duration_capture, duration_capture1, duration_capture2;
    // 获取当前系统时间，格式化为“年.月.日”
    auto tNow = std::chrono::system_clock::now();
    auto tmNow = std::chrono::system_clock::to_time_t(tNow);
    auto locNow = std::localtime(&tmNow);
    std::ostringstream oss;
    oss << std::put_time(locNow, "%Y.%m.%d");
    std::string monthday = "_" + oss.str() + "_";
    // 拼接csv文件名，格式：结果路径 + 线路名 + 时间 + result.csv
    std::string csvname = result_files_name + linename + monthday + "result.csv";

    auto  qcsvname = QString::fromStdString(csvname);

    qDebug() << "csv名称: " << qcsvname;






    // 打开里程信息文件，路径为传入的files_name目录下的"Milepost1231.mp"
    std::ifstream inputFile(files_name+"/Milepost1231.mp");  // 打开文件


    std::vector<std::string> contentsAfterColon;// 用来存放每行“:”后面的内容
    std::string line;
    int lineCount = 0;
    const int maxLines = 10;  // 只处理前十行
    // 逐行读取文件，最多10行
    while (std::getline(inputFile, line) && lineCount < maxLines) {
        size_t colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            // 提取":"之后的字符串
            std::string content = line.substr(colonPos + 1);
            contentsAfterColon.push_back(content);
        }
        lineCount++;
    }
    if (contentsAfterColon.size() < 10)// 如果没读满10行，直接返回（文件格式不对或缺失）
        return;
    inputFile.close();

    // 以下开始使用word_对象操作Word文档，设置排版格式、字体、插入内容
    word_.setPageOrientation(0);// 设置页面方向（0可能是纵向）
    word_.setWordPageView(3);// 设置页面视图模式（具体数字含义依实现）
    word_.setParagraphAlignment(0);// 段落左对齐
    word_.setFontSize(30);// 字体大小30
    word_.setFontBold(true);// 加粗
    word_.insertText(QObject::tr("地铁隧道巡检报告"));// 插入主标题文本
    word_.setFontBold(false);// 取消加粗

    // 插入多次换行（空行）
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.setParagraphAlignment(1); // 居中对齐
    //QDateTime time = QDateTime::currentDateTime();
    //QString tsme = time.toString("yyyy-MM-dd hh:mm:ss dddd");
    //QString tsme_one = "巡检时间：" + tsme;


    word_.setFontSize(15);// 插入巡检信息，字体大小15
    //word_.insertText(tsme_one);
    word_.insertMoveDown();
    word_.setFontSize(15);
    word_.insertText("巡检线路："+ QString::fromStdString(contentsAfterColon[1]));
    word_.insertMoveDown();
    word_.setFontSize(15);
    word_.insertText("巡检员：" + QString::fromStdString(contentsAfterColon[3]));
    word_.insertMoveDown();
    word_.setFontSize(15);

    //QString tempupdown = QString::fromLocal8Bit(contentsAfterColon[2].c_str());
    // 上下行，使用 fromLocal8Bit 解决编码问题（中文）
    word_.insertText("上下行："+ QString::fromLocal8Bit(contentsAfterColon[2].c_str()));
    word_.insertMoveDown();
    word_.setFontSize(15);
    word_.insertText("里程加减：" + QString::fromStdString(contentsAfterColon[4]));
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.insertMoveDown();
    word_.setFontSize(15);
    word_.insertText("当前里程：" + QString::fromStdString(contentsAfterColon[6]));
    word_.insertMoveDown();
    word_.setFontSize(15);
    word_.insertText("归档日期：" + QString::fromStdString(contentsAfterColon[5]));

    //插入分页符
    word_.InsertBreak();

    //word_.insertMoveDown();
    // 第二页开始，设置格式，插入章节标题“1.背景”
    word_.setPageOrientation(0);
    word_.setWordPageView(3);

    word_.setParagraphAlignment(1); // 居中
    word_.setFontSize(15);
    word_.setFontBold(true);
    word_.insertText(QObject::tr("1.背景"));
    word_.setFontBold(false);
    word_.insertMoveDown();

    // 继续设置格式，插入正文内容，段落间距和字号设置
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(10);
    //word_.setFirstLineIndent(20);
    word_.setLineSpacing(1);
    word_.setFontBold(false);
    word_.insertText(QObject::tr("    近年来，地铁巡检迎来了新的黄金发展时期。由于地下环境阴暗潮湿、地铁负荷量大、制动频繁等因素的影响，裂纹、渗漏水、掉块等隧道表面会造成损伤，导致地铁易发生故障而严重危害公共交通安全。这对地铁列车的安全巡检提出了更高的要求传统的地铁车辆巡检方式以人工为主，具有效率低、成本高、结果不稳定等不足之处。因此合肥超科电子公司针对现代化的背景以及地铁的所需要求，研发出了此地铁隧道巡检小车。"));
    word_.setFontBold(false);
    word_.insertMoveDown(); // 

    // 插入第二个章节标题“2.实现功能”
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(15);
    word_.setFontBold(true);
    word_.insertText(QObject::tr("2.实现功能"));
    word_.setFontBold(false);
    word_.insertMoveDown(); // 

    // 插入实现功能的详细条目
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(10);
    word_.setFontBold(false);
    word_.insertText(QObject::tr("    （1）实现对隧道270°的同步采集；"));
    word_.insertMoveDown(); // 
    word_.insertText(QObject::tr("    （2）最大巡检时速为15km/h，覆盖范围超过12m，像素精度0.2mm；"));
    word_.insertMoveDown(); // 
    word_.insertText(QObject::tr("    （3）缺陷检测：裂纹、渗漏水、掉块等；"));
    word_.setFontBold(false);
    word_.insertMoveDown(); //

    // 插入第三个章节标题“3.硬件需求”
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(15);
    word_.setFontBold(true);
    word_.insertText(QObject::tr("3.硬件需求"));
    word_.setFontBold(false);
    word_.insertMoveDown();

    // 插入硬件需求详细内容（多条）
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(10);
    word_.setFontBold(false);
    word_.insertText(QObject::tr("    （1）巡检小车应能在轨道上双向行驶，可随时启停，行驶方 向和行驶速度可调节操作台、电控箱等电器设备外壳防护等级达到GB4208 规定的IP65 级，供货时提供第三方检测报告；"));
    word_.insertMoveDown();
    word_.insertText(QObject::tr("    （2）重量：≤300kg，拆卸后单个部件重量≤60kg；"));
    word_.insertMoveDown();
    word_.insertText(QObject::tr("    （3）行驶速度：≤15km/h ，续航里程：≥60km；"));
    word_.insertMoveDown();
    word_.insertText(QObject::tr("    （4）爬坡能力：≥40‰；"));
    word_.insertMoveDown();
    word_.insertText(QObject::tr("    （5）刹车距离： ≤30m  (长直轨道上) ；"));
    word_.insertMoveDown();
    word_.setFontBold(false);

    // 插入第四章节标题“4.检测结果文件”
    word_.setPageOrientation(0);
    word_.setWordPageView(3);
    word_.setParagraphAlignment(1);
    word_.setFontSize(15);
    word_.setFontBold(true);
    word_.insertText(QObject::tr("4.检测结果文件"));
    word_.setFontBold(false);
    word_.insertMoveDown(); // 插入回车

    word_.setFontSize(10);
    word_.insertText("输出报表：");
    word_.insertMoveDown();

  

    // 插入超链接到生成的csv文件路径
    QAxObject *hyperlinks = word_.getDocument()->querySubObject("Hyperlinks");
    QAxObject* range = word_.getDocument()->querySubObject("Content");

    QAxObject* paragraphs = range->querySubObject("Paragraphs");

    int paragraphCount = paragraphs->property("Count").toInt();


    // 获取最后一段的范围
    QAxObject* lastParagraph = paragraphs->querySubObject("Item(int)", paragraphCount);


    QAxObject* lastRange = lastParagraph->querySubObject("Range");


    // 添加超链接，链接地址和显示文本均为csv文件路径
    hyperlinks->dynamicCall("Add(QAxObject*, QString, QVariant, QVariant, QString)",
    lastRange->asVariant(), qcsvname, QVariant(), QVariant(), qcsvname);



    // 关闭Word文档，保存退出
    word_.close();

    //QAxObject *content = word_.getDocument();





}
tunnel_inspection::tunnel_inspection(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::tunnel_inspectionClass)
{
    ui->setupUi(this);
    //初始化界面
    init();

	// 算法线程初始化，加载模型，设置回调函数
	alg_thread_.init(
        [this](const result& ret, const cv::Mat& frame) {
		    if (frame.empty()) {
			    ui->progress_bar->setMaximum(ret.bar_value);
			    return;
		    }
            //显示处理耗时（毫秒）和当前进度值到time_label和signals_bar。
		    ui->time_label->setText(QString::number(ret.proc_time)+"==="+ QString::number(ret.bar_value));
		    signals_bar(ret.bar_value+2);

		    //if (!ui->radio_button->isChecked())return;

		    cv::Mat tframe = frame.clone();
            cv::cvtColor(tframe, tframe, CV_BGR2RGB);//QT是RGB格式
            //将 OpenCV 图像转换为 Qt 可显示的 QImage
		    QImage qimage = QImage((uchar*)tframe.data, tframe.cols, tframe.rows,
			    tframe.cols * tframe.channels(), QImage::Format_RGB888);
		    ui->oringinal_label->setPixmap(QPixmap::fromImage(qimage));//显示图像
	    });
    
}

tunnel_inspection::~tunnel_inspection()
{
    delete ui;
}

void tunnel_inspection::init()
{
    //显示图像label
    ui->oringinal_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->oringinal_label->setScaledContents(true);

    

    //设置标题
    this->setWindowTitle("隧道巡检回放系统 v1.0");
    this->setWindowIcon(QIcon(":/mainwindow/logo.PNG"));

    //加载qss
    QFile file("Ubuntu.qss");    // 如果 exe 同目录
    if (file.open(QFile::ReadOnly)) {
        QString style = QLatin1String(file.readAll());
        qApp->setStyleSheet(style);
    }

    //加载配置
    //qDebug() << "CameraPath_1: " << AppConfig::CameraPath_1;
    ui->lineCameraPath_1->setText(AppConfig::CameraPath_1);
    ui->lineCameraPath_2->setText(AppConfig::CameraPath_2);
    ui->lineCameraPath_3->setText(AppConfig::CameraPath_3);
    ui->lineCameraPath_4->setText(AppConfig::CameraPath_4);
    ui->lineCameraPath_5->setText(AppConfig::CameraPath_5);
    ui->lineCameraPath_6->setText(AppConfig::CameraPath_6);
    ui->lineSavePicturePath->setText(AppConfig::SavePicturePath);
    ui->lineSaveResultPath->setText(AppConfig::SaveResultPath);
    ui->line_mileage->setText(QString::number(AppConfig::Mileage));
    m_mileage = AppConfig::Mileage;

    //保存原图cbx控件
    if (ui->lineSavePicturePath->text() != nullptr)
    {
        ui->save_radio_button->setEnabled(true);
    }
    else
    {
        ui->save_radio_button->setEnabled(false);
    }
    ui->save_radio_button->setChecked(false);
    connect(ui->save_radio_button, &QCheckBox::stateChanged,
        this, &tunnel_inspection::save_raw_picture_ckb);
    
    //链接控件事件到配置保存函数
    connect(ui->lineCameraPath_1, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineCameraPath_2, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineCameraPath_3, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineCameraPath_4, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineCameraPath_5, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineCameraPath_6, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineSavePicturePath, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->lineSaveResultPath, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);
    connect(ui->line_mileage, &QLineEdit::textChanged,
        this, &tunnel_inspection::saveConfig);

    //链接进度条事件
    connect(this, &tunnel_inspection::signals_bar, ui->progress_bar, [=](int value) {
        ui->progress_bar->setValue(value); // 更新进度条的值  
        });
}

void tunnel_inspection::saveConfig()
{
    AppConfig::CameraPath_1 = ui->lineCameraPath_1->text();
    AppConfig::CameraPath_2 = ui->lineCameraPath_2->text();
    AppConfig::CameraPath_3 = ui->lineCameraPath_3->text();
    AppConfig::CameraPath_4 = ui->lineCameraPath_4->text();
    AppConfig::CameraPath_5 = ui->lineCameraPath_5->text();
    AppConfig::CameraPath_6 = ui->lineCameraPath_6->text();
    AppConfig::SavePicturePath = ui->lineSavePicturePath->text();
    AppConfig::SaveResultPath = ui->lineSaveResultPath->text();
    AppConfig::Mileage = ui->line_mileage->text().toDouble();

    AppConfig::writeConfig();
}




//start按键进行裂纹检测和渗水掉块
void tunnel_inspection::on_start_pushbutton_clicked() {
    //初始化摄像头-1
    camera_id_ = -1;
    // 设置文本编码为GB2312，用于从Qt字符串转换为std::string（主要处理中文路径）
    QTextCodec *code = QTextCodec::codecForName("GB2312");
    // 从界面输入的路径编辑框获取路径，转换为std::string（GB2312编码）

    std::vector<QString> files_name(6);
    //两个读图的line
    files_name[0] = ui->lineCameraPath_1->text();
    files_name[1] = ui->lineCameraPath_2->text();
    files_name[2] = ui->lineCameraPath_3->text();
    files_name[3] = ui->lineCameraPath_4->text();
    files_name[4] = ui->lineCameraPath_5->text();
    files_name[5] = ui->lineCameraPath_6->text();
   
    //两个保存路径的line
    std::string store_files_name = code->fromUnicode(ui->lineSavePicturePath->text()).data();
    std::string result_files_name = code->fromUnicode(ui->lineSaveResultPath->text()).data();

    // 打开并检测每个摄像头数据是否正常，将正确打开的摄像头文件存入有效数组中
	auto  maxbar_value = alg_thread_.set_data_name(files_name, store_files_name, result_files_name);//拼接摄像头的二进制文件名
    // 设置界面进度条的最大值，进度条范围 [0, maxbar_value]
	ui->progress_bar->setMaximum(maxbar_value);
	
    // 准备存放所有摄像头数据文件名的容器
	std::vector<std::string>files_names;
    // 根据摄像头编号，组合对应的文件路径
	for (int i = 0; i < CAMERANUMBER; i++) {
        files_names.push_back((files_name[i] + +"/Recv.img").toStdString());
	}

    // 初始化并启动每个摄像头对应的 file_data 实例（负责读取文件和裂纹检测）
	for (int i = 0; i < CAMERANUMBER; i++) {
        //回调函数
		file_datas_[i].init(
            [this](file_data::frame& frame) {
			    alg_thread_.push_frame(frame.clone());// 1. 将克隆的帧数据传递给算法线程进行渗水和掉块检测
                if (camera_id_ == -1) {
                    camera_id_ = frame.camera_id; // 2. 如果camera_id_是-1（还没赋值），就设置为当前帧的相机ID
                }
                if (camera_id_ == frame.camera_id&&ui->save_radio_button->isChecked()) {
                    signals_bar(frame.frame_number + 2); // 3. 如果当前帧是选中的摄像头帧且保存选项被勾选，更新进度条
                }
            },
            files_names[i],// 文件路径
            i);// 摄像头ID
        //开始裂纹检测
        qDebug() << "开始裂纹检测";
		file_datas_[i].start();
        
        // 设置文件存储路径、结果路径及是否保存文件的标志
		file_datas_[i].set_params(store_files_name+"/"+std::to_string(i+1)+"/",// 存储路径（分摄像头子文件夹）
            result_files_name,// 结果保存路径
            ui->save_radio_button->isChecked()&&ui->lineSavePicturePath->text()!=nullptr);// 是否保存标志
		
	}

	//开始渗水和掉块检测
    qDebug() << "开始渗水掉块检测";
	alg_thread_.start();
    
    // 获取当前系统时间，用于生成带日期的文件名
    auto tNow = std::chrono::system_clock::now();
    auto tmNow = std::chrono::system_clock::to_time_t(tNow);
    auto locNow = std::localtime(&tmNow);
    std::ostringstream oss;
    //格式化日期
    oss << std::put_time(locNow, "%Y.%m.%d");
    std::string monthday = oss.str() + "_";
    // 拼接Word文件名，存放检测结果
    std::string wordname = result_files_name + monthday + "result.doc";
    
    qDebug() << "拼接报表名称: " << QString::fromStdString(wordname);
    if (!word_.createWord(QString::fromStdString(wordname))) {
        QString error = QObject::tr("导出失败,") + word_.getStrErrorInfo();
        qDebug() << error;

    }
    // 创建Word文档标题，传入第一个文件夹路径作为标题内容
    create_word_title(files_name[0].toStdString());

}

void tunnel_inspection::update_bar() {


}

//加载相机1
void tunnel_inspection::on_btnCameraPath_1_clicked() {
    //文件夹路径
    QString path = ui->lineCameraPath_1->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_1->setText(src_dirpath);

}
//加载相机2
void tunnel_inspection::on_btnCameraPath_2_clicked() {
    QString path = ui->lineCameraPath_2->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_2->setText(src_dirpath);

}
//加载相机3
void tunnel_inspection::on_btnCameraPath_3_clicked() {
    QString path = ui->lineCameraPath_3->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_3->setText(src_dirpath);

}
//加载相机4
void tunnel_inspection::on_btnCameraPath_4_clicked() {
    QString path = ui->lineCameraPath_4->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_4->setText(src_dirpath);

}
//加载相机5
void tunnel_inspection::on_btnCameraPath_5_clicked() {
    QString path = ui->lineCameraPath_5->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_5->setText(src_dirpath);

}
//加载相机6
void tunnel_inspection::on_btnCameraPath_6_clicked() {
    QString path = ui->lineCameraPath_6->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineCameraPath_6->setText(src_dirpath);

}

//保存原图路径按键
void tunnel_inspection::on_btnSavePicturePath_clicked() {
    QString path = ui->lineSavePicturePath->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineSavePicturePath->setText(src_dirpath);

    for (int i = 1; i <= CAMERANUMBER; ++i) {
        QString folderName = QString("%1").arg(i);
        QString fullPath = src_dirpath + "/" + folderName;
        bool result = QDir().mkpath(fullPath);
    }
    if (ui->lineSavePicturePath->text() != nullptr)
    {
        ui->save_radio_button->setEnabled(true);
    }
    else
    {
        ui->save_radio_button->setEnabled(false);
    }
}
void tunnel_inspection::save_raw_picture_ckb(int ischecked)
{
    qDebug() << "cbx is checked! ";
    if (ischecked)
    {
        QString src_dirpath = ui->lineSavePicturePath->text();
        for (int i = 1; i <= CAMERANUMBER; ++i) {
            QString folderName = QString("%1").arg(i);
            QString fullPath = src_dirpath + "/" + folderName;
            bool result = QDir().mkpath(fullPath);
        }
    }
    
}

//保存结果路径
void tunnel_inspection::on_btnSaveResultPath_clicked() {
    QString path = ui->lineSaveResultPath->text();
    QDir dir(path);
    if (!dir.exists()) {
        path = "/";
    }

    auto  src_dirpath = QFileDialog::getExistingDirectory(
        this, "选择文件夹",
        path);
    ui->lineSaveResultPath->setText(src_dirpath);

}