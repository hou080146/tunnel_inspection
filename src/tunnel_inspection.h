#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_tunnel_inspection.h"
#include"alg_thread.h" 
#include"result.h"
#include "file_data.h"
#include"qword.h"

class tunnel_inspection : public QMainWindow
{
    Q_OBJECT 
             
public:
    tunnel_inspection(QWidget *parent = nullptr);
    ~tunnel_inspection();
    void create_word_title(std::string files_name);
private:
    void init();
    
public slots:
    //加载文件路径
    void on_btnCameraPath_1_clicked();
    void on_btnCameraPath_2_clicked();
    void on_btnCameraPath_3_clicked();
    void on_btnCameraPath_4_clicked();
    void on_btnCameraPath_5_clicked();
    void on_btnCameraPath_6_clicked();

    void on_btnSavePicturePath_clicked();
    void on_btnSaveResultPath_clicked();
    void save_raw_picture_ckb(int ischecked);
    void on_start_pushbutton_clicked();
    void update_bar();
    void saveConfig();
signals:
    void signals_bar(int value);
private:
    Ui::tunnel_inspectionClass *ui;
	alg_thread alg_thread_;
	file_data file_datas_[8];
    int camera_id_ = -1;
    QWord word_;
    double m_mileage;
};