#pragma once
#include <QObject>
#include <QSettings>
#include <QFile>
#include <QTime>

#define TIMES qPrintable(QTime::currentTime().toString("HH:mm:ss zz"))

class AppConfig
{
public:
    static QString ConfigFile;
    static bool Style;
    static bool WriteFlag;
    static QString CameraPath_1;
    static QString CameraPath_2;
    static QString CameraPath_3;
    static QString CameraPath_4;
    static QString CameraPath_5;
    static QString CameraPath_6;
    static QString SavePicturePath;
    static QString SaveResultPath;
    static double Mileage;
    static int MileageDown;

    static void readConfig();
    static void writeConfig();

    static bool checkIniFile(const QString &iniFile);
};

