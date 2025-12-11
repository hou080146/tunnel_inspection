#include "tunnel_inspection.h"
#include <QtWidgets/QApplication>
#include"img.h"
#include "AppConfig.h"


int main(int argc, char *argv[])
{     
    QApplication a(argc, argv);
    AppConfig::ConfigFile = QString("Config.ini");
    AppConfig::readConfig();
    tunnel_inspection w;
    w.show();     
    return a.exec();
}
