from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
import sys

import warnings 
warnings.filterwarnings("ignore")

from MyWindow import MyWindow

def main1():
    app = QApplication(sys.argv)
    vw = MyWindow()
    vw.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main1()
