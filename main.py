from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys

# from mainwindow import Ui_MainWindow
from MyWindow import MyWindow
# from test import Ui_MainWindow

def main1():
    app = QApplication(sys.argv)
    # player = QMediaPlayer()
    vw = MyWindow()
    vw.show()
    # player.setVideoOutput(vw.videoFrame)
    # player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
    # player.play()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main1()
