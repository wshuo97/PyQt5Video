from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from DaT import DaT

import os
import cv2
import numpy as np
# import _thread
from utils.plots import colors, plot_one_box


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
        # self.selectDet.addItems(['YOLOv5s', 'YOLOv5l', 'YOLOv5x'])

        # signal and slot
        self.selectVideo.clicked.connect(self.getVideo)
        self.selectDet.clicked.connect(self.getDet)
        self.vidControl.clicked.connect(self.setPlayState)
        self.startTracking.clicked.connect(self.setTrackState)
        self.startDetecting.clicked.connect(self.setDetectState)

        self.sendImg.connect(self.showImg)
        self.loadVid.connect(self.vidPlay)

        # class
        # _thread.start_new_thread(self.showImgThread, (self,))

        # others
        self.DaT = DaT()
        self.images = [1 for _ in range(10)]
        self.inputImg = [1 for _ in range(10)]
        self.srcImages = []
        self.ablePlay = False
        self.exit = False
        self.stFrameIndex = 0
        self.frameid = 0
        self.isDetecting = False
        self.isTracking = False
        self.hasTracker = False
        self.hasDetector = False
        self.couldShowImg = False
        self.detectResults = []
        self.trackResults = []

    def getVideo(self):
        directory = QFileDialog.getOpenFileNames(
            self, "Select File", "./", "All Files (*);;Text Files (*.txt)")[0]
        if len(directory) == 0:
            return
        self.vidPath = directory[0]
        cap = cv2.VideoCapture(self.vidPath)
        # frameid = 0
        self.videoLength = int(cap.get(7))
        self.videoFps = int(cap.get(5))
        self.blockTime = int(1./self.videoFps*1000)
        self.size = (self.showBlock.width(), self.showBlock.height())
        cap.release()
        cv2.destroyAllWindows()
        self.videoPath.setText(self.vidPath)

    def getDet(self):
        directory = QFileDialog.getOpenFileNames(
            self, "Select File", "./", "All Files (*);;Text Files (*.txt)")[0]
        if len(directory) == 0:
            return
        self.detWeights = directory[0]
        self.DaT.newDetector(self.detWeights)
        self.detectorPath.setText(self.detWeights)

    def setPlayState(self):
        self.ablePlay = not self.ablePlay
        if self.ablePlay:
            self.vidControl.setText("暂停")
            self.loadVid.emit()
        else:
            self.vidControl.setText("播放")

    def setDetectState(self):
        self.isDetecting = not self.isDetecting
        if self.isDetecting:
            self.startDetecting.setText("停止检测")
        else:
            self.startDetecting.setText("开始检测")

    def setTrackState(self):
        # if not self.isDetecting and not self.isTracking:
        #     QtWidgets.QMessageBox.question(self, "Wraning!", "No Detector!")
        if not self.hasTracker:
            self.hasTracker = True
            self.DaT.newTracker()
        self.isTracking = not self.isTracking
        if self.isTracking:
            self.startTracking.setText("停止跟踪")
        else:
            self.startTracking.setText("开始跟踪")

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameid)
        while(cap.isOpened()):
            _, frame = cap.read()
            ret = self.ablePlay
            if self.exit:
                break
            if ret == True:
                resFrame = frame
                if self.isDetecting and not self.isTracking:
                    (bbox_xywh, cls_ids, cls_conf, detRes) = self.detecting(frame)
                    resFrame = self.drawBboxes(
                        detRes, frame, self.frameid, "detect")
                elif self.isTracking:
                    (bbox_xywh, cls_ids, cls_conf, _) = self.detecting(frame)
                    trackRes = self.tracking(
                        bbox_xywh, cls_ids, cls_conf, frame)
                    # print(trackRes)
                    resFrame = self.drawBboxes(trackRes, frame, self.frameid)
                self.images[self.frameid % 10] = resFrame
                # self.couldShowImg = True
                self.sendImg.emit(self.frameid % 10)
            else:
                break
            self.frameid = self.frameid+1
            # if self.isTracking:
            cv2.waitKey(self.blockTime)

        cap.release()
        cv2.destroyAllWindows()

    def drawBboxes(self, trackRes, im0s, frameid, flag="track"):
        opt = self.DaT.opt
        im0 = im0s.copy()
        # print(trackRes)
        cnames = ["person", "vehicle"]
        for _, det in enumerate(trackRes):
            if len(det):
                for *xyxy, idx in reversed(det):
                    if flag != "track":
                        labelid = cnames[int(idx)]
                    else:
                        labelid = f'{idx}'
                    self.trackResults.append((frameid - 1, cnames[_],
                                              xyxy, idx))
                    plot_one_box(xyxy,
                                 im0,
                                 label=labelid,
                                 color=colors(idx % 255, True),
                                 line_thickness=opt.line_thickness)
        return im0

    def write_results(self, filename, results):
        save_format = "{frame},{cname},{id},{x1},{y1},{x2},{y2}\n"
        with open(filename, 'w') as f:
            for frame_id, cls_name, xyxy, track_id in results:
                x1, y1, x2, y2 = xyxy
                line = save_format.format(frame=frame_id,
                                          id=track_id,
                                          x1=x1,
                                          y1=y1,
                                          x2=x2,
                                          y2=y2,
                                          cname=cls_name)
                f.write(line)

    def detecting(self, frame):
        return self.DaT.detect(frame)

    def tracking(self, bbox_xywh, cls_ids, cls_conf, frame):
        return self.DaT.track(bbox_xywh, cls_ids, cls_conf, frame)

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self, "警告!", "确定关闭软件?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.exit = True
            if len(self.trackResults) != 0:
                self.write_results("./log/results.txt", self.trackResults)
            event.accept()
        else:
            event.ignore()
