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
public slots:
    void on_load_pushbutton_clicked();
    void on_load_pushbutton_2_clicked(); 
    void on_path_pushbutton_clicked();
    void on_result_path_pushbutton_clicked();
    void on_save_pushbutton_clicked();
    void on_start_pushbutton_clicked();
    void on_picture_pushbutton_clicked();
    void update_bar();
signals:
    void signals_bar(int value);
private:
    Ui::tunnel_inspectionClass ui;
	alg_thread alg_thread_;
	file_data file_datas_[8];
    int camera_id_;
    QWord word_;
};