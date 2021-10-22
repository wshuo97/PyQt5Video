from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from PyQt5.uic import loadUi

import cv2


class MyWindow(QMainWindow):
    # new signal
    sendImg = pyqtSignal(int)
    loadVid = pyqtSignal()

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        loadUi("BaseWindow.ui", self)

        # window settings
        self.showBlock.setFixedSize(
            self.showBlock.width(), self.showBlock.height())

        # signal and slot
        self.selectVideo.clicked.connect(self.getVideo)
        self.selectDet.clicked.connect(self.getDet)
        self.vidControl.clicked.connect(self.setPlayState)

        self.sendImg.connect(self.showImg)
        self.loadVid.connect(self.vidPlay)

        # others
        self.images = [1 for _ in range(10)]
        self.srcImages = []
        self.ablePlay = False
        self.exit = False
        self.stFrameIndex = 0
        self.frameid = 0

    def getVideo(self):
        directory = QFileDialog.getOpenFileNames(
            self, "Select File", "./", "All Files (*);;Text Files (*.txt)")[0]
        self.vidPath = directory[0]

        cap = cv2.VideoCapture(self.vidPath)
        # frameid = 0
        self.videoLength = int(cap.get(7))
        self.videoFps = int(cap.get(5))
        self.blockTime = int(1./self.videoFps*1000)
        self.size = (self.showBlock.width(), self.showBlock.height())
        # while cap.isOpened():

        #     # while not self.ablePlay:
        #     #     continue

        #     _, frame = cap.read()
        #     if frame is None:
        #         break
        #     self.images.append(frame)
        #     # self.srcImages.append(frame)
        #     # self.images.append(cv2.resize(frame, self.size))
        #     print(frameid)
        #     # self.sendImg.emit(frameid)
        #     frameid = frameid+1

        #     # cv2.waitKey(blockTime)

        cap.release()
        cv2.destroyAllWindows()

        self.videoPath.setText(self.vidPath)

        # self.loadVid.emit()

    def getDet(self):
        directory = QFileDialog.getOpenFileNames(
            self, "Select File", "./", "All Files (*);;Text Files (*.txt)")[0]
        self.detPath = directory[0]
        self.detectorPath.setText(self.vidPath)

    def setPlayState(self):
        self.ablePlay = not self.ablePlay
        if self.ablePlay:
            self.vidControl.setText("暂停")
            self.loadVid.emit()
        else:
            self.vidControl.setText("播放")

    def showImg(self, index):
        # img = self.images[index]
        img = cv2.resize(self.images[index], self.size)
        shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  shrink.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)

        self.showBlock.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.showBlock.show()

    def vidPlay(self):
        cap = cv2.VideoCapture(self.vidPath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameid)  # 设置要获取的帧号
        # self.frameid = 0
        while(cap.isOpened()):
            _, frame = cap.read()
            ret = self.ablePlay
            if self.exit:
                break
            if ret == True:
                self.images[self.frameid % 10] = frame
                self.sendImg.emit(self.frameid % 10)
            else:
                break
            self.frameid = self.frameid+1
            cv2.waitKey(self.blockTime)

        cap.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self, "Wraning!", "Really to quit?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.exit = True
            event.accept()
        else:
            event.ignore()
