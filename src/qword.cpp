#include "qword.h"
#include <QDateTime>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <ActiveQt/QAxObject>
#include <ActiveQt/QAxWidget>
#include<QTextStream>
#include <ActiveQt/QAxBase>
#include <QTextFormat>
#include <QDebug>
#pragma execution_character_set("utf-8")
QWord::QWord(QObject* parent)
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);//解决非主线程无法调用word问题
	m_word = new QAxObject(parent);
	m_documents = NULL;
	m_document = NULL;
	m_bOpened = false;

}

QWord::~QWord()
{
	//自己注释
	//close();
}

bool QWord::createWord(QString reportname)		//创建一个新的word
{
	QString ReportName = reportname;
	QString defaultFileName = QString("%1").arg(ReportName);
	//m_saveName = QFileDialog::getSaveFileName(0, tr("Report Information"), defaultFileName, tr("*.doc"));
    m_saveName = defaultFileName;
	//CGlobalAppData *pAppData = CAppDataSingleton::instance();
	QString SavePath = /*pAppData->m_strAppDirPath + */"/ReportWord" + m_saveName;
	QFile file(m_saveName);
    if (file.exists())
    {
        qDebug() << "文件已存在，尝试删除后重新创建: " << m_saveName;
        if (!file.remove())
        {
            m_strError = tr("error:无法删除已存在的文件!");
            qDebug() << m_strError;
            return false;
        }
    }
	if (!m_saveName.isEmpty())
	{
		bool ok = m_word->setControl("kwps.Application");

		if (!ok)
		{
			// 用wps打开
			ok = m_word->setControl("Word.Application");
			if (!ok)
			{
				//			//m_strError = QString::fromLocal8Bit("错误：获取word组件失败，请确定是否安装了word!");
				return false;
			}
			//			m_strError = tr("abnormal:failed to get the word component,please make sure the word is installed!");

			//			return false;
		}
		m_word->setProperty("Visible", false);
		m_word->setProperty("DisplayAlerts", false);//不显示任何警告信息。如果为true那么在关闭是会出现类似“文件已修改，是否保存”的提示
		m_documents = m_word->querySubObject("Documents");  // 获取所有的工作文档
		m_documents->dynamicCall("Add (void)"); // 新建一个文档页
		m_document = m_word->querySubObject("ActiveDocument");//获取当前激活的文档 	
		return true;
	}
	else
	{
		m_strError = tr("abnormal2:the file name is empty!");
		//m_strError = QString::fromLocal8Bit("错误：文件名为空");		
		return false;
	}
}
//bool QWord::createNewWord(const QString& filePath)
//{
//	m_saveName = filePath;
//	QFile file(m_saveName);
//	if (file.exists())
//	{
//		file.remove(m_saveName);
//		//m_strError = tr("error:the file already exists!");
//		//m_strError = QString::fromLocal8Bit("错误：目标文件已存在!");			
//		//return false;
//	}
//	if (!m_saveName.isEmpty())
//	{
//		bool ok = m_word->setControl("Word.Application");
//		if (!ok)
//		{
//			// 用wps打开
//			ok = m_word->setControl("kwps.Application");
//			if (!ok)
//			{
//				m_strError = tr("abnormal:failed to get the word component,please make sure the word is installed!");
//				//m_strError = QString::fromLocal8Bit("错误：获取word组件失败，请确定是否安装了word!\n");
//				return false;
//			}
//		}
//
//
//
//		m_word->setProperty("Visible", false);
//		m_word->setProperty("DisplayAlerts", false);//不显示任何警告信息。如果为true那么在关闭是会出现类似“文件已修改，是否保存”的提示
//		m_documents = m_word->querySubObject("Documents");
//		if (!m_documents)
//		{
//			m_strError = tr("abnormal:failed to get the documents!");
//			//m_strError = QString::fromLocal8Bit("获取文档失败！\n");
//			return false;
//		}
//		m_documents->dynamicCall("Add (void)");
//		m_document = m_word->querySubObject("ActiveDocument");//获取当前激活的文档 	
//		return true;
//	}
//	else
//	{
//		m_strError = tr("abnormal:the file name is empty!");
//		//m_strError = QString::fromLocal8Bit("错误：文件名为空");		
//		return false;
//	}
//}

bool QWord::createNewWord(const QString& filePath)
{
	m_saveName = filePath;
	QFile file(m_saveName);
	if (file.exists())
	{
		qDebug() << "文件已存在，尝试删除后重新创建: " << m_saveName;
		if (!file.remove())
		{
			m_strError = tr("error:无法删除已存在的文件!");
			qDebug() << m_strError;
			return false;
		}
	}

	if (!m_saveName.isEmpty())
	{
		bool ok = m_word->setControl("Word.Application");
		if (!ok)
		{
			// 用wps打开
			ok = m_word->setControl("kwps.Application");
			if (!ok)
			{
				m_strError = tr("abnormal1:获取Word组件失败，请确定是否安装了Word或WPS!\n");
				qDebug() << m_strError;
				return false;
			}
		}

		bool visibleSetOk = m_word->setProperty("Visible", false);
		if (!visibleSetOk)
		{
			m_strError = tr("abnormal2:设置Word应用程序可见性属性失败！");
			qDebug() << m_strError;
			return false;
		}

		bool alertsSetOk = m_word->setProperty("DisplayAlerts", false);
		if (!alertsSetOk)
		{
			m_strError = tr("abnormal3:设置Word应用程序显示警告属性失败！");
			qDebug() << m_strError;
			return false;
		}

		m_documents = m_word->querySubObject("Documents");
		if (!m_documents)
		{
			m_strError = tr("abnormal4:获取文档集合失败！\n");
			qDebug() << m_strError;
			return false;
		}

		QAxObject* newDoc = m_documents->dynamicCall("Add (void)").value<QAxObject*>();
		if (!newDoc)
		{
			// 进一步检查dynamicCall返回值为空的情况，并输出详细错误信息
			m_strError = tr("abnormal5:添加新文档操作失败，可能是与Word应用程序通信问题或Word应用程序状态异常！");
			qDebug() << m_strError;
			return false;
		}
		m_document = newDoc;

		return true;
	}
	else
	{
		m_strError = tr("abnormal:文件名为为");
		qDebug() << m_strError;
		return false;
	}
}


bool QWord::openword(bool bVisable)
{
	m_word = new QAxObject();
	bool bFlag = m_word->setControl("word.Application");
	if (!bFlag)
	{
		bFlag = m_word->setControl("kwps.Application");
	}
	if (!bFlag)
	{
		return false;
	}
	m_word->setProperty("Visible", bVisable);
	QAxObject* documents = m_word->querySubObject("Documents");
	if (!documents)
	{
		return false;
	}
	documents->dynamicCall("Add(QString)", m_strFilePath);
	m_bOpened = true;
	return m_bOpened;
}

bool QWord::open(const QString& strFilePath, bool bVisable)
{
	m_strFilePath = strFilePath;
	//close();
	return openword(bVisable);
}
bool QWord::isOpen()
{
	return m_bOpened;
}
void QWord::save()
{
	if (m_document)
		m_document->dynamicCall("Save()");
	else
		return;
}


//自己注释
//void QWord::close()				//关闭 退出 析构时候也会自动调用一次
//{
//	if (!m_saveName.isEmpty())		//如果不为空  则为新建   
//	{
//		saveAs();
//		m_saveName = "";
//	}
//	//if(m_document)
//	//	m_document->dynamicCall("Close (boolean)",false);
//	//if(m_word)	
//	//	m_word->dynamicCall("Quit (void)");	
//	if (m_documents)
//		delete m_documents;
//	if (m_word)
//		delete m_word;
//	m_document = NULL;
//	m_documents = NULL;
//	m_word = NULL;
//}

//void QWord::saveAs()
//{
//	if (m_document)
//		m_document->dynamicCall("SaveAs(const QString&)", QDir::toNativeSeparators(m_saveName));
//	else
//		return;
//}


void QWord::setStrErrorInfo(const QString& errorInfo) {
	m_strErrorInfo = errorInfo;
}

QString QWord::getStrErrorInfo() {
	return m_strErrorInfo;
}


//void QWord::saveAs(QString& path)
//{
//	if (m_document) {
//		try {
//			// 调用Word的SaveAs方法保存文件，将路径转换为本地格式
//			m_document->dynamicCall("SaveAs(const QString&)", QDir::toNativeSeparators(path));
//		}
//		catch (const std::exception& e) {
//			// 捕获更具体的异常类型，输出详细错误信息
//			qDebug() << "保存文件时出现异常: " << e.what();
//			setStrErrorInfo(QString::fromStdString(e.what()));
//			return;
//		}
//		catch (...) {
//			qDebug() << "保存文件时出现未知异常";
//			setStrErrorInfo("保存文件时出现未知异常");
//			return;
//		}
//	}
//	else {
//		return;
//	}
//}

void QWord::saveAs(QString& path)
{
	if (m_document)
	{
		// 检查文件是否已被其他程序占用
		QFileInfo fileInfo(path);
		if (fileInfo.exists() && fileInfo.isFile() && fileInfo.isWritable())
		{
			qDebug() << "文件已存在且可写，尝试直接保存: " << path;
			try
			{
				m_document->dynamicCall("SaveAs(const QString&)", QDir::toNativeSeparators(path));
			}
			catch (const std::exception& e)
			{
				// 捕获更具体的异常类型，输出详细错误信息
				qDebug() << "保存文件时出现异常: " << e.what();
				setStrErrorInfo(QString::fromStdString(e.what()));
				return;
			}
			catch (...)
			{
				qDebug() << "保存文件时出现未知异常";
				setStrErrorInfo("保存文件时出现未知异常");
				return;
			}
		}
		else
		{
			qDebug() << "文件不存在或不可写，创建新文件保存: " << path;
			try
			{
				m_document->dynamicCall("SaveAs(const QString&)", QDir::toNativeSeparators(path));
			}
			catch (const std::exception& e)
			{
				// 捕获更具体的异常类型，输出详细错误信息
				qDebug() << "保存文件时出现异常: " << e.what();
				setStrErrorInfo(QString::fromStdString(e.what()));
				return;
			}
			catch (...)
			{
				qDebug() << "保存文件时出现未知异常";
				setStrErrorInfo("保存文件时出现未知异常");
				return;
			}
		}
	}
	else
	{
		return;
	}
}

void QWord::close() {
    saveAs(m_saveName);
	if (m_document) {
		// 关闭Word文档
		m_document->dynamicCall("Close");
		// 释放COM组件资源
		delete m_document;
		m_document = nullptr;
	}
}


void QWord::setPageOrientation(int flag)	//设置页面1 横向   还是 0竖向
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QString page;
	switch (flag)
	{
	case 0:
		page = "wdOrientPortrait";
		break;
	case 1:
		page = "wdOrientLandscape";
		break;
	}
	selection->querySubObject("PageSetUp")->setProperty("Orientation", page);
}
void QWord::setWordPageView(int flag)
{
	QAxObject* viewPage = m_word->querySubObject("ActiveWindow");
	if (NULL == viewPage)
	{
		return;
	}
	QString view;
	switch (flag)
	{
	case 1:
		view = "wdNormalView";
		break;
	case 2:
		view = "wdOutlineView";
		break;
	case 3:
		view = "wdPrintView";
		break;
	case 4:
		view = "wdPrintPreview";
		break;
	case 5:
		view = "wdMasterView";
		break;
	case 6:
		view = "wdWebView";
		break;
	case 7:
		view = "wdReadingView";
		break;
	case 8:
		view = "wdConflictView";
		break;
	}
	viewPage->querySubObject("View")->setProperty("Type", view);
}
void QWord::insertMoveDown()				//插入回车
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->dynamicCall("TypeParagraph(void)");
}

void QWord::InsertBreak()				//插入分页符
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}

	//    value = selection->dynamicCall("PageBreakBefore (int)").toBool();

	selection->dynamicCall("InsertBreak  (int)", 7);
}

void QWord::insertText(const QString& text)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->dynamicCall("TypeText(const QString&)", text);
}

QString QWord::GetText()
{
	QAxObject* selection = m_word->querySubObject("Selection");
	QString str;
	if (NULL != selection)
	{
		str = selection->dynamicCall("GetText(void)").toString();
	}

	return str;
}
//设置选中位置文字居中 0 ,居左 1,居右 2
void QWord::setParagraphAlignment(int flag)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	if (flag == 0)
	{
		selection->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphCenter");
	}
	else if (flag == 1)
	{
		selection->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphJustify");
	}
	else if (flag == 2)
	{
		selection->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphRight");
	}
}


//首行缩进
void QWord::setFirstLineIndent(int indent)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->querySubObject("ParagraphFormat")->setProperty("FirstLineIndent", indent);
}

//设置行距  1为1.5倍行距
void QWord::setLineSpacing(int flag)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	if (flag == 1)
	{
		QAxObject* paragraphFormat = selection->querySubObject("ParagraphFormat");
		if (paragraphFormat)
		{
			paragraphFormat->setProperty("LineSpacing", 18);
		}
	}
}


void QWord::setRowAlignment(/*int tableIndex*/QAxObject* table, int row, int flag)
{
	//	QAxObject* tables = m_document->querySubObject("Tables");
	//    if(NULL== tables)
	//	{
	//		return;
	//	}
	//	QAxObject* table = tables->querySubObject("Item(int)",tableIndex);
	//    if(NULL== table )
	//	{
	//		return;
	//	}
	QAxObject* Row = table->querySubObject("Rows(int)", row);
	if (NULL == Row)
	{
		return;
	}
	QAxObject* range = Row->querySubObject("Range");
	if (NULL == range)
	{
		return;
	}
	Row->querySubObject("Alignment(int)", flag);
	if (flag == 0)
	{
		range->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphCenter"); // 水平居中
		range->querySubObject("Cells")->setProperty("VerticalAlignment", "wdCellAlignVerticalCenter");//垂直居中
	}
	else if (flag == 1)
	{
		range->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphJustify");
	}
	else if (flag == 2)
	{
		range->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphRight");
	}
	else if (flag == 3)
	{
		range->querySubObject("ParagraphFormat")->setProperty("Alignment", "wdAlignParagraphLeft");
	}
}
void QWord::setFontSize(int fontsize)		//设置字体大小
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->querySubObject("Font")->setProperty("Size", fontsize);
}



void QWord::setFontBold(bool flag)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->querySubObject("Font")->setProperty("Bold", flag);
}
void QWord::setFontName(QString& fontName)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->querySubObject("Font")->setProperty("Name", fontName);
}
void QWord::setSelectionRange(int start, int end)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	selection->dynamicCall("SetRange(int, int)", start, end);	//第1个字符后开始，到第9个字符结束范围
}
void QWord::getUsedRange(int* topLeftRow, int* topLeftColumn, int* bottomRightRow, int* bottomRightColumn)
{
	QAxObject* range = m_document->querySubObject("Range");
	if (NULL == range)
	{
		return;
	}
	*topLeftRow = range->property("Row").toInt();
	if (NULL == topLeftRow)
	{
		return;
	}
	*topLeftColumn = range->property("Column").toInt();
	if (NULL == topLeftColumn)
	{
		return;
	}
	QAxObject* rows = range->querySubObject("Rows");
	if (NULL == rows)
	{
		return;
	}
	*bottomRightRow = *topLeftRow + rows->property("Count").toInt() - 1;
	if (NULL == bottomRightRow)
	{
		return;
	}
	QAxObject* columns = range->querySubObject("Columns");
	if (NULL == columns)
	{
		return;
	}
	*bottomRightColumn = *topLeftColumn + columns->property("Count").toInt() - 1;
	if (NULL == bottomRightColumn)
	{
		return;
	}
}
void QWord::insertHyperlink(const QString& filePath, const QString& displayText) {
    if (!m_document) return;
    QAxObject* content1 = m_document->querySubObject("Content");
    if (!content1) return;

    QAxObject* hyperlinks = m_document->querySubObject("Hyperlinks");
    if (!hyperlinks) return;

    hyperlinks->dynamicCall("Add( QString, QVariant, QVariant, QString)",filePath, QVariant(), QVariant(), displayText);
}
void QWord::insertPic(QString picPath)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	selection->querySubObject("ParagraphFormat")->dynamicCall("Alignment", "wdAlignParagraphCenter");
	QVariant tmp = selection->asVariant();
	QList<QVariant>qList;
	qList << QVariant(picPath);
	qList << QVariant(false);
	qList << QVariant(true);
	qList << tmp;
	QAxObject* Inlineshapes = m_document->querySubObject("InlineShapes");
	Inlineshapes->dynamicCall("AddPicture(const QString&, QVariant, QVariant ,QVariant)", qList);
}


QAxObject* QWord::intsertTable(int row, int column)
{
	QAxObject* tables = m_document->querySubObject("Tables");
	if (NULL == tables)
	{
		return NULL;
	}
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return NULL;
	}
	QAxObject* range = selection->querySubObject("Range");
	if (NULL == range)
	{
		return NULL;
	}
	QVariantList params;
	params.append(range->asVariant());
	params.append(row);
	params.append(column);
	tables->querySubObject("Add(QAxObject*, int, int, QVariant&, QVariant&)", params);
	QAxObject* table = selection->querySubObject("Tables(int)", 1);
	if (NULL == table)
	{
		return NULL;
	}
	table->setProperty("Style", "网格型");
	QAxObject* Borders = table->querySubObject("Borders");
	if (NULL == Borders)
	{
		return NULL;
	}
	Borders->setProperty("InsideLineStyle", 1);
	Borders->setProperty("OutsideLineStyle", 1);
	/*QString doc = Borders->generateDocumentation();
	QFile outFile("D:\\360Downloads\\Picutres\\Borders.html");
	outFile.open(QIODevice::WriteOnly|QIODevice::Append);
	QTextStream ts(&outFile);
	ts<<doc<<endl;*/

	return table;
}
void QWord::intsertTable(int tableIndex, int row, int column)
{
	QAxObject* tables = m_document->querySubObject("Tables");
	if (NULL == tables)
	{
		return;
	}
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* range = selection->querySubObject("Range");
	if (NULL == range)
	{
		return;
	}
	QVariantList params;
	params.append(range->asVariant());
	params.append(row);
	params.append(column);
	tables->querySubObject("Add(QAxObject*, int, int, QVariant&, QVariant&)", params);
	QAxObject* table = selection->querySubObject("Tables(int)", tableIndex);
	if (NULL == table)
	{
		return;
	}
	table->setProperty("Style", "网格型");
	QAxObject* Borders = table->querySubObject("Borders");
	if (NULL == Borders)
	{
		return;
	}
	Borders->setProperty("InsideLineStyle", 1);
	Borders->setProperty("OutsideLineStyle", 1);
	/*QString doc = Borders->generateDocumentation();
	QFile outFile("D:\\360Downloads\\Picutres\\Borders.html");
	outFile.open(QIODevice::WriteOnly|QIODevice::Append);
	QTextStream ts(&outFile);
	ts<<doc<<endl;*/
}

void QWord::setColumnWidth(int column, int width)		//设置列宽
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	table->querySubObject("Columns(int)", column)->setProperty("Width", width);

}
void QWord::setCellString(/*int nTable*/QAxObject* table, int row, int column, const QString& text)
{
	// 	QAxObject* pTables =m_document->querySubObject("Tables");
	//    if(NULL==pTables)
	//	{
	//		return;
	//	}
	//	QAxObject* table=pTables->querySubObject("Item(int)",nTable);
	if (table)
	{
		table->querySubObject("Cell(int,int)", row, column)->querySubObject("Range")
			->dynamicCall("SetText(QString)", text);
	}
}
void QWord::MergeCells(/*int tableIndex*/QAxObject* table, int nStartRow, int nStartCol, int nEndRow, int nEndCol)//合并单元格
{
	//	QAxObject* tables = m_document->querySubObject("Tables");
	//    if(NULL==tables)
	//	{
	//		return;
	//	}
	//	QAxObject* table = tables->querySubObject("Item(int)",tableIndex);
	//    if(NULL== table)
	//	{
	//		return;
	//	}
	if (table)
	{
		QAxObject* StartCell = table->querySubObject("Cell(int, int)", nStartRow, nStartCol);
		QAxObject* EndCell = table->querySubObject("Cell(int, int)", nEndRow, nEndCol);
		if (NULL == StartCell)
		{
			return;
		}
		if (NULL == EndCell)
		{
			return;
		}
		StartCell->querySubObject("Merge(QAxObject *)", EndCell->asVariant());
	}

}

//第二种方法调用
// void QWord::MergeCells(int tableIndex, int nStartRow,int nStartCol,int nEndRow,int nEndCol)//合并单元格
// {
// 	QAxObject* tables = m_document->querySubObject("Tables");	
// 	QAxObject* table = tables->querySubObject("Item(int)",tableIndex);
// 	QAxObject* StartCell =table->querySubObject("Cell(int, int)",nStartRow,nStartCol);
// 	QAxObject* EndCell = table->querySubObject("Cell(int, int)",nEndRow,nEndCol);
// 	StartCell->dynamicCall("Merge(LPDISPATCH)",EndCell->asVariant());
// }

void QWord::setColumnHeight(int nTable, int column, int height)
{
	QAxObject* pTables = m_document->querySubObject("Tables");
	if (NULL == pTables)
	{
		return;
	}
	QAxObject* table = pTables->querySubObject("Item(int)", nTable);
	if (table)
	{
		table->querySubObject("Columns(int)", column)->setProperty("Hight", height);
	}
}
void QWord::setRowHeight(int nTable, int Row, int height)
{
	QAxObject* pTables = m_document->querySubObject("Tables");
	if (NULL == pTables)
	{
		return;
	}
	QAxObject* table = pTables->querySubObject("Item(int)", nTable);

	if (table)
	{
		table->querySubObject("Rows(int)", Row)->setProperty("Hight", height);
	}
}

void QWord::setColumnHeight(int column, int height)		//设置列高
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (table)
	{
		table->querySubObject("Columns(int)", column)->setProperty("Hight", height);
	}

}


void QWord::setCellString(int row, int column, const QString& text)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	QAxObject* rows = table->querySubObject("Rows");
	if (rows)
	{
		return;
	}
	int Count = rows->dynamicCall("Count").toInt();
	table->querySubObject("Cell(int, int)", row, column)->querySubObject("Range")->dynamicCall("SetText(QString)", text);
}

void QWord::setCellFontBold(int row, int column, bool isBold)	//设置内容粗体  isBold控制是否粗体
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	table->querySubObject("Cell(int, int)", row, column)->querySubObject("Range")->dynamicCall("SetBold(int)", isBold);
}
void QWord::setCellFontSize(int row, int column, int size)		//设置文字大小
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	table->querySubObject("Cell(int, int)", row, column)->querySubObject("Range")->querySubObject("Font")->setProperty("Size", size);
}
QVariant QWord::getCellValue(int row, int column)					//获取单元格内容 此处对于Excel来说列和行从1开始最少
{
	QAxObject* selection = m_word->querySubObject("Selection");
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL != selection && NULL != table)
		return table->querySubObject("Cell(int, int)", row, column)->querySubObject("Range")->property("Text");
}
int QWord::getTableCount()
{
	QAxObject* tables = m_document->querySubObject("Tables");
	int val;
	if (NULL != tables)
	{
		val = tables->property("Count").toInt();
	}
	return val;
}
void QWord::moveForEnd()
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QVariantList params;
	params.append(6);
	params.append(0);
	selection->dynamicCall("EndOf(QVariant&, QVariant&)", params).toInt();
}
void QWord::insertCellPic(int row, int column, const QString& picPath)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	QAxObject* range = table->querySubObject("Cell(int, int)", row, column)->querySubObject("Range");
	if (NULL == range)
	{
		return;
	}
	range->querySubObject("InlineShapes")->dynamicCall("AddPicture(const QString&)", picPath);
}
void QWord::setTableAutoFitBehavior(int flag)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	if (0 <= flag && flag <= 2)
		table->dynamicCall("AutoFitBehavior(WdAutoFitBehavior)", flag);
}
void QWord::deleteSelectColumn(int column)
{
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* table = selection->querySubObject("Tables(1)");
	if (NULL == table)
	{
		return;
	}
	QAxObject* columns = table->querySubObject("Columns(int)", column);
	if (NULL == columns)
	{
		return;
	}
	columns->dynamicCall("Delete()");
}

void QWord::setOptionCheckSpell(bool flags)
{
	QAxObject* opetions = m_word->querySubObject("Options");
	if (!opetions)
		return;
	opetions->setProperty("CheckGrammarAsYouType", flags);
	opetions->setProperty("CheckGrammarWithSpelling", flags);
	opetions->setProperty("ContextualSpeller", flags);
	opetions->setProperty("CheckSpellingAsYouType", flags);
}

void QWord::addTableRow(int tableIndex, int nRow, int rowCount)
{
	QAxObject* tables = m_document->querySubObject("Tables");
	if (NULL == tables)
	{
		return;
	}
	QAxObject* table = tables->querySubObject("Item(int)", tableIndex);
	if (NULL == table)
	{
		return;
	}
	QAxObject* rows = table->querySubObject("Rows");
	if (NULL == rows)
	{
		return;
	}
	int Count = rows->dynamicCall("Count").toInt();
	if (0 < nRow && nRow <= Count)
	{
		for (int i = 0; i < rowCount; ++i)
		{
			QString sPos = QString("Item(%1)").arg(nRow + i);
			QAxObject* row = rows->querySubObject(sPos.toStdString().c_str());
			QAxObject* row1 = rows->querySubObject("Last");
			QAxObject* row2 = rows->querySubObject("Row(int)", nRow + i);
			QAxObject* row3 = rows->querySubObject("Item(int)", nRow + i);
			if (NULL != row)
			{
				QVariant param = row->asVariant();
				/*rows->dynamicCall("Add(Variant)",param);*/
				QAxObject* NewRow = rows->querySubObject("Add(Variant)", param);
			}

		}
	}
}
////创建表格
void QWord::insertTable(int tableIndex, int row, int column)
{

	QAxObject* tables = m_document->querySubObject("Tables");
	if (NULL == tables)
	{
		return;
	}
	QAxObject* table = tables->querySubObject("Item(int)", tableIndex);
	if (NULL == table)
	{
		return;
	}
	//QAxObject* rows =table->querySubObject("Rows");
	QAxObject* selection = m_word->querySubObject("Selection");
	if (NULL == selection)
	{
		return;
	}
	QAxObject* range = selection->querySubObject("Range");
	if (NULL == range)
	{
		return;
	}
	QVariantList params;
	params.append(range->asVariant());
	params.append(row);
	params.append(column);
	tables->querySubObject("Add(QAxObject*, int, int, QVariant&, QVariant&)", params);
	table = selection->querySubObject("Tables(int)", 1);
	table->setProperty("Style", "网格型");

	QAxObject* Borders = table->querySubObject("Borders");
	if (NULL == Borders)
	{
		return;
	}
	Borders->setProperty("InsideLineStyle", 1);
	Borders->setProperty("OutsideLineStyle", 1);
}
////设置表格列宽
void QWord::setColumnWidth(int nTable, int column, int width)
{
	QAxObject* pTables = m_document->querySubObject("Tables");
	if (NULL == pTables)
	{
		return;
	}
	QAxObject* table = pTables->querySubObject("Item(int)", nTable);
	if (table)
	{
		table->querySubObject("Columns(int)", column)->setProperty("width", width);
	}

}


//在表格中插入图片
void QWord::insertCellPic(/*int nTable*/QAxObject* table, int row, int column, const QString& picPath)
{
	if (table)
	{
		QDir dir(picPath);
		QString strTemp = dir.absolutePath();
		strTemp = QDir::toNativeSeparators(strTemp);

		QAxObject* range = table->querySubObject("Cell(int, int)", row, column)
			->querySubObject("Range");
		range->querySubObject("InlineShapes")
			->dynamicCall("AddPicture(const QString&)", strTemp);
	}
}
//QAxObject* range = table->querySubObject("Cell(int,int )", row, column)->querySubObject("Range");
//if (NULL == range)
//{
//	return;
//}
//bool result = range->querySubObject("InlineShapes")->dynamicCall("AddPicture(const QString&)", picPath).toBool();
//if (result)
//{
//	qDebug() << "插入成功";
//}
//else
//{
//	qDebug() << "插入失败";
//}
//设置内容粗体
void QWord::setCellFontBold(int nTable, int row, int column, bool isBold)
{
	QAxObject* pTables = m_document->querySubObject("Tables");
	if (NULL == pTables)
	{
		return;
	}
	QAxObject* table = pTables->querySubObject("Item(int)", nTable);
	if (NULL == table)
	{
		return;
	}
	table->querySubObject("Cell(int,int )", row, column)->querySubObject("Range")->dynamicCall("SetBold(int)", isBold);


}
//设置文字大小
void QWord::setCellFontSize(int nTable, int row, int column, int size)
{
	QAxObject* pTables = m_document->querySubObject("Tables");
	if (NULL == pTables)
	{
		return;
	}
	QAxObject* table = pTables->querySubObject("Item(int)", nTable);
	if (NULL == table)
	{
		return;
	}
	table->querySubObject("Cell(int,int)", row, column)->querySubObject("Range")->querySubObject("Font")->setProperty("Size", size);

}

void QWord::setVisible(bool isVisible)
{
	if (m_word != NULL)
	{
		m_word->setProperty("Visible", isVisible);
	}
}
