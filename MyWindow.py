from PyQt5 import QtGui
from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from PyQt5.uic import loadUi

import cv2


class MyWindow(QMainWindow):
    sendImg = pyqtSignal(int)

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)

        loadUi("BaseWindow.ui", self)

        # self.player = QMediaPlayer()
        # self.videoPlayer = QVideoWidget(self.videoFrame)
        # self.videoPlayer.setFixedSize(
        #     self.videoFrame.width(), self.videoFrame.height())
        # self.videoPlayer.show()
        # self.player.setVideoOutput(self.videoPlayer)

        self.selectVideo.clicked.connect(self.getVideo)
        # self.player.positionChanged.connect(self.test)

        self.showBlock.setFixedSize(
            self.showBlock.width(), self.showBlock.height())
        # self.showBlock.setStyleSheet("QLabel{background:white;}"
        #                              "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
        #                              )

        self.image = []
        self.sendImg.connect(self.showImg)

        self.exit = False

    def msg(self):
        directory = QFileDialog.getOpenFileNames(
            self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        cap = cv2.VideoCapture(directory[0])

        self.videoLength = cap.get(7)
        self.videoFps = cap.get(5)

        cap.release()
        cv2.destroyAllWindows()
        print(directory)  # ('D:/data/pythonProject/MyWindow.py', 'All Files (*)')

        qMC = QMediaContent(QUrl(directory[0]))
        self.player.setMedia(qMC)
        self.videoPath.setText(directory[0])

        # print(self.player.MediaStatus())
        # print(self.player.position())

        self.player.play()

    def getVideo(self):
        directory = QFileDialog.getOpenFileNames(
            self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        cap = cv2.VideoCapture(directory[0])

        self.videoLength = cap.get(7)
        self.videoFps = cap.get(5)
        frameid = 0
        self.videoLength = cap.get(7)
        self.videoFps = cap.get(5)
        blockTime = int(1./self.videoFps*1000)
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None or self.exit:
                break
            self.image.append(frame)
            self.sendImg.emit(frameid)
            frameid = frameid+1

            cv2.waitKey(blockTime)

        cap.release()
        cv2.destroyAllWindows()

    def showImg(self, index):
        shrink = cv2.cvtColor(self.image[index], cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  shrink.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)

        self.showBlock.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.showBlock.show()

    def test(self):
        print(self.player.position(), "/", self.videoLength)
