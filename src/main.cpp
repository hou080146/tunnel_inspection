#include "tunnel_inspection.h"
#include <QtWidgets/QApplication>
#include"img.h"
#include <QScreen>
#include "AppConfig.h"


int main(int argc, char *argv[])
{     
    QApplication a(argc, argv);
    AppConfig::ConfigFile = QString("Config.ini");
    AppConfig::readConfig();
    tunnel_inspection w;

    // 获取屏幕的几何信息
    QRect screenRect = QGuiApplication::primaryScreen()->geometry();
    // 计算弹窗的中心位置
    int x = (screenRect.width() - w.width()) / 2;
    int y = (screenRect.height() - w.height()) / 2 - 15;
    // 设置弹窗的位置
    w.move(x, y);
    
    w.show();     
    return a.exec();
}
