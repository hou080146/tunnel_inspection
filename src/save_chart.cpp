

#include"save_chart.h"

#include "qword.h"

#include <QDebug>
#include <QObject>

#include <QDir>
#include <QCoreApplication>
void save_chart::save_excel(vector<string>qx_label, vector<string>lc, vector<string> qx_path, string save_path, string qslc, int sxx) {
    return;
	
}
//
//



#pragma execution_character_set("utf-8")
//Word保存
void save_chart::save_word(vector<string>qx_label, vector<string>lc, vector<string> qx_path, int tt_lj, string save_path, string qslc, int sxx) {
	// 将string类型的save_path转换为QString类型
        QString qSavePath = QString::fromStdString(save_path);

        // 获取当前可执行文件所在的目录路径作为相对路径的基准
        QString currentDir = QCoreApplication::applicationDirPath();

        // 拼接完整的相对保存路径
        QString fullSavePath = currentDir + QDir::separator() + qSavePath;

        // 确认保存路径是否可访问，如果不可访问尝试创建目录
        QDir saveDir(fullSavePath);
        if (!saveDir.exists()) {
            if (!saveDir.mkpath(fullSavePath)) {
                qDebug() << "无法创建保存目录: " << fullSavePath;
                return;
            }
        }

        QString save_word_lj = fullSavePath + "hfdt.docx";
        qDebug() << "word_lj:" << save_word_lj;

        QWord word;
        if (!word.createWord(save_word_lj)) {
            QString error = QObject::tr("导出失败,") + word.getStrErrorInfo();
            qDebug() << error;
            return;
        }
	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(0);
	word.setFontSize(30);
	word.setFontBold(true);
	word.insertText(QObject::tr("合肥地铁巡检报告"));
	word.setFontBold(false);
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.setParagraphAlignment(1);
	//QDateTime time = QDateTime::currentDateTime();
	//QString tsme = time.toString("yyyy-MM-dd hh:mm:ss dddd");
	//QString tsme_one = "巡检时间：" + tsme;
	word.setFontSize(15);
	//word.insertText(tsme_one);
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("巡检线路：");
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("巡检员：");
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("记录检查员：");
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("巡检线路：");
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("归档编号：");
	word.insertMoveDown();
	word.setFontSize(15);
	word.insertText("归档日期：");
	//插入分页符
	word.InsertBreak();

	//word.insertMoveDown();

	word.setPageOrientation(0);
	word.setWordPageView(3);

	word.setParagraphAlignment(1);
	word.setFontSize(15);
	word.setFontBold(true);
	word.insertText(QObject::tr("1.背景"));
	word.setFontBold(false);
	word.insertMoveDown();

	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(10);
	//word.setFirstLineIndent(20);
	word.setLineSpacing(1);
	word.setFontBold(false);
	word.insertText(QObject::tr("    近年来，地铁巡检迎来了新的黄金发展时期。由于地下环境阴暗潮湿、地铁负荷量大、制动频繁等因素的影响，钢轨、扣件、轨枕、道床等轨道部件易造成损伤，导致地铁易发生故障而严重危害公共交通安全。这对地铁列车的安全巡检提出了更高的要求传统的地铁车辆巡检方式以人工为主，具有效率低、成本高、结果不稳定等不足之处。因此合肥超科电子公司针对现代化的背景以及北京地铁的所需要求，研发出了此北京地铁巡检小车。"));
	word.setFontBold(false);
	word.insertMoveDown(); // 

	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(15);
	word.setFontBold(true);
	word.insertText(QObject::tr("2.实现功能"));
	word.setFontBold(false);
	word.insertMoveDown(); // 

	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(10);
	word.setFontBold(false);
	word.insertText(QObject::tr("    （1）实现对钢轨、扣件、轨枕、道床等轨道部件的同步采集；"));
	word.insertMoveDown(); // 
	word.insertText(QObject::tr("    （2）最大巡检时速为15km/h，覆盖范围超过3m，测量精度1mm，行频5k以上；"));
	word.insertMoveDown(); // 
	word.insertText(QObject::tr("    （3）所需缺陷检测：弹条缺失、弹条偏移、螺丝帽断裂、轨距块断裂、轨距块丢失、胶垫串出、道床异物、螺栓浮起(4mm)；"));
	word.setFontBold(false);
	word.insertMoveDown(); //

	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(15);
	word.setFontBold(true);
	word.insertText(QObject::tr("3.硬件需求"));
	word.setFontBold(false);
	word.insertMoveDown();

	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(10);
	word.setFontBold(false);
	word.insertText(QObject::tr("    （1）巡检小车应能在轨道上双向行驶，可随时启停，行驶方 向和行驶速度可调节操作台、电控箱等电器设备外壳防护等级达到GB4208 规定的IP65 级，供货时提供第三方检测报告；"));
	word.insertMoveDown();
	word.insertText(QObject::tr("    （2）重量：≤300kg，拆卸后单个部件重量≤60kg；"));
	word.insertMoveDown();
	word.insertText(QObject::tr("    （3）行驶速度：≤15km/h ，续航里程：≥60km；"));
	word.insertMoveDown();
	word.insertText(QObject::tr("    （4）爬坡能力：≥40‰；"));
	word.insertMoveDown();
	word.insertText(QObject::tr("    （5）刹车距离： ≤30m  (长直轨道上) ；"));
	word.insertMoveDown();
	word.setFontBold(false);


	word.setPageOrientation(0);
	word.setWordPageView(3);
	word.setParagraphAlignment(1);
	word.setFontSize(15);
	word.setFontBold(true);
	word.insertText(QObject::tr("4.检测结果"));
	word.setFontBold(false);
	word.insertMoveDown(); // 插入回车

	QStringList load_list;
	word.setFontSize(12);
	int table_column = 2 + load_list.length();// 
	int kn_table_cols = table_column % 2;
	table_column += kn_table_cols; // 
	//QDateTime time = QDateTime::currentDateTime();
	//QString tsme = time.toString("yyyy-MM-dd hh:mm:ss dddd");
	int nums = qx_label.size();
	//cout << "Numeber of images: " << tt_lj << "," << qx_path.size() << std::endl;
	string get_mileage = to_string(tt_lj * 0.0015);

	QString distance = QString::number(nums);
	//QString tsme_two = "    今日巡检时间为" + tsme + ",巡检人员为尚宏伟，随海亮等，巡检距离共" + QString::fromStdString(get_mileage) + "km，" + "共发现缺陷" + distance + "个。";
	//word.insertText(tsme_two);
	word.insertMoveDown();

	//插入一个几行几列表格
	word.insertMoveDown(); // 插入回车

	//cout << "行数：" << nums << std::endl;
	QAxObject* table = word.intsertTable(nums + 1, 4);

	word.setCellString(table, 1, 1, QObject::tr("序号"));
	word.setColumnWidth(1, 35);    //设置；l
	word.setCellString(table, 1, 2, QObject::tr("异常类型"));
	word.setColumnWidth(2, 65);    //设置；l
	word.setCellString(table, 1, 3, QObject::tr("异常位置"));
	word.setColumnWidth(3, 65);    //设置；l
	word.setCellString(table, 1, 4, QObject::tr("异常图像"));
	word.setColumnWidth(4, 250);    //设置；l


	for (int i = 0; i < nums; ++i)
	{
		//for (const auto& qx : qx_path) {
		//word.setColumnWidth(i+2,4);
		//word.setParagraphAlignment(1);
		word.setCellString(table, i + 2, 1, QString::number(i));
		word.setCellString(table, i + 2, 2, QString::fromStdString(qx_label[i]));
		string lc_shw;
		if (sxx == 0) {
			lc_shw = qslc + "-" + lc[i];
		}
		else {
			lc_shw = qslc + "+" + lc[i];
		}
		word.setCellString(table, i + 2, 3, QString::fromStdString(lc_shw));
		qDebug() << "SL:" << QString::fromStdString(qx_path[i]) << endl;
		word.insertCellPic(table, i + 2, 4, QString::fromStdString(qx_path[i]));
	}


	word.setVisible(false);
	word.saveAs(save_word_lj);

	// 检查文件是否确实被保存成功
	//QFileInfo savedFile(save_word_lj);
	//QMessageBox::information(nullptr, "保存成功", "Word已成功保存！");

	// 关闭Word相关资源
	word.close();

}


//void save_chart::save_excel(vector<string>excel_list, string save_path) {
//	QXlsx::Document xlsx; 
//	std::string image_path =""; string defect_category = ""; string defect_coordinate = "";
//	xlsx.write(1, 1, "序号"); xlsx.write(1, 2, "轨道"); xlsx.write(1, 3, "异常位置"); xlsx.write(1, 4, "异常类型"); xlsx.write(1, 5, "异常图像"); 
//	xlsx.setColumnWidth("C1", 15);
//	xlsx.setColumnWidth("E1", 90);
//	//i为列数
//	QString cr_zg;	int nums = excel_list.size();
//	//std::cout << qx << std::endl;
//	for (int i = 0; i < nums; ++i)
//	{
//		//std::cout << "xs__________________________________：" << excel_list[i] << std::endl;
//		//split(zg_whole.all[j], image_path, defect_category, defect_coordinate);
//		split(excel_list[i], image_path, defect_category, defect_coordinate);
//		
//		cr_zg = QString::fromStdString(image_path);
//		QImage image(cr_zg);
//		//cout << "qx_label[i]：" << image_path <<"，"<< defect_category << "," << defect_coordinate << std::endl;
//		//cout << "lc[i]：" << lc[i] << std::endl;
//		xlsx.setRowHeight(i + 2, 150);
//		//cout << "i + 1：" << i + 1 << std::endl;
//		xlsx.write(i + 2, 1, i + 1);  xlsx.write(i + 2, 2, "左轨"); xlsx.write(i + 2, 3, atoi(defect_coordinate.c_str())); xlsx.write(i + 2, 4, atoi(defect_category.c_str()));
//		xlsx.insertImage(i + 1, 4, image.scaled(600, 200, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
//
//	}
//	
//	QString save_excel_lj = QString::fromStdString(save_path + "合肥地铁巡检.xlsx");
//	qDebug() << save_excel_lj << endl;
//	xlsx.saveAs(save_excel_lj);
//}


void save_chart::split(string zfc, string image_path, string defect_category, string defect_coordinate) {
#pragma execution_character_set("utf-8")
	//std::string image_path, defect_category, defect_coordinate;
	size_t dot_bmp_index = zfc.find(".bmp");
	if (dot_bmp_index != std::string::npos) {
		std::cout << "dot_bmp_index：" << dot_bmp_index << std::endl;
		// 提取图像路径  一个中文路径
		image_path = zfc.substr(10, dot_bmp_index - 4);

		// 找到缺陷类别和缺陷坐标的分隔符
		size_t separator_index = zfc.find("缺陷类别", dot_bmp_index);
		size_t qx_index = zfc.find("缺陷坐标", dot_bmp_index);
		if (separator_index != std::string::npos) {
			// 提取缺陷类别
			defect_category = zfc.substr(separator_index + 10, qx_index - separator_index - 10);
			std::cout << "dot_bmp_index：" << separator_index + 10 << "," << qx_index << std::endl;
			// 提取缺陷坐标
			defect_coordinate = zfc.substr(qx_index + 10);
			std::cout << "图像路径：" << image_path << std::endl;
			std::cout << "缺陷类别：" << defect_category << std::endl;
			std::cout << "缺陷坐标：" << defect_coordinate << std::endl;
		}
	}

}





