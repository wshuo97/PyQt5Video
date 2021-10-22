from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from PyQt5.uic import loadUi
# from test import Ui_MainWindow


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)

        self.player = QMediaPlayer()
        loadUi("BaseWindow.ui", self)

        # QVideoWidget()
        self.videoPlayer = QVideoWidget(self.videoFrame)
        self.player.setVideoOutput(self.videoPlayer)
        self.selectVideo.clicked.connect(self.msg)
        self.videoFrame.show()

    def msg(self, Filepath):
        # directory = QFileDialog.getOpenFileUrl(
        #     self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        directory = QFileDialog.getOpenFileUrl()[0]
        # self.videoPath.setText(directory[0])
        print(directory) # ('D:/data/pythonProject/MyWindow.py', 'All Files (*)')
        qMC = QMediaContent(directory)
        self.player.setMedia(qMC)
        self.player.play()
        # self.videoPlayer.show()
