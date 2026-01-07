#include "AppConfig.h"
#include <qDebug>



QString AppConfig::ConfigFile = "Config.ini";
QString AppConfig::CameraPath_1 = "G:";
QString AppConfig::CameraPath_2 = "G:";
QString AppConfig::CameraPath_3 = "G:";
QString AppConfig::CameraPath_4 = "G:";
QString AppConfig::CameraPath_5 = "G:";
QString AppConfig::CameraPath_6 = "G:";
QString AppConfig::SavePicturePath = "D:Vedio";
QString AppConfig::SaveResultPath = "D:Vedio";
double AppConfig::Mileage = 0.0;
int AppConfig::MileageDown = 1;
bool AppConfig::WriteFlag = false;
bool AppConfig::Style = false;


void AppConfig::readConfig()
{
    QSettings set(AppConfig::ConfigFile, QSettings::IniFormat);

    set.beginGroup("AppConfig");
    AppConfig::WriteFlag = set.value("WriteFlag").toBool();
    AppConfig::Style = set.value("Style").toBool();
    AppConfig::CameraPath_1 = set.value("CameraPath_1").toString();
    AppConfig::CameraPath_2 = set.value("CameraPath_2").toString();
    AppConfig::CameraPath_3 = set.value("CameraPath_3").toString();
    AppConfig::CameraPath_4 = set.value("CameraPath_4").toString();
    AppConfig::CameraPath_5 = set.value("CameraPath_5").toString();
    AppConfig::CameraPath_6 = set.value("CameraPath_6").toString();
    AppConfig::SavePicturePath = set.value("SavePicturePath").toString();
    AppConfig::SaveResultPath = set.value("SaveResultPath").toString();
    AppConfig::Mileage = set.value("Mileage").toDouble();
    //AppConfig::MileageDown = set.value("Mileage").toInt();
    set.endGroup();

    if (!AppConfig::checkIniFile(AppConfig::ConfigFile))
    {
        writeConfig();
        return;
    }
}

void AppConfig::writeConfig()
{
    QSettings set(AppConfig::ConfigFile, QSettings::IniFormat);

    //qDebug() << " in writeConfig";
    set.beginGroup("AppConfig");
    set.setValue("WriteFlag", AppConfig::WriteFlag);
    set.setValue("Style", AppConfig::Style);
    set.setValue("CameraPath_1", AppConfig::CameraPath_1);
    set.setValue("CameraPath_2", AppConfig::CameraPath_2);
    set.setValue("CameraPath_3", AppConfig::CameraPath_3);
    set.setValue("CameraPath_4", AppConfig::CameraPath_4);
    set.setValue("CameraPath_5", AppConfig::CameraPath_5);
    set.setValue("CameraPath_6", AppConfig::CameraPath_6);
    set.setValue("SavePicturePath", AppConfig::SavePicturePath);
    set.setValue("SaveResultPath", AppConfig::SaveResultPath);
    set.setValue("Mileage", AppConfig::Mileage);
    //set.setValue("MileageDown", AppConfig::MileageDown);
    set.endGroup();
}



bool AppConfig::checkIniFile(const QString &iniFile)
{
    QFile file(iniFile);
    if (file.size() == 0)
    {
        return false;
    }

    if (file.open(QFile::ReadOnly))
    {
        bool ok = true;
        while (!file.atEnd())
        {
            QString line = file.readLine();
            line.replace("\r", "");
            line.replace("\n", "");
            QStringList list = line.split("=");

            if (list.count() == 2)
            {
                QString key = list.at(0);
                QString value = list.at(1);
                if (value.isEmpty())
                {
                    qDebug() << TIMES << "ini node no value" << key;
                    ok = false;
                    break;
                }
            }
        }
        if (!ok)
        {
            return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}