from PyQt5.QtCore import (
                    Qt, QPointF, QRectF, QSizeF, QObject, QThread, pyqtSignal, pyqtSlot,
                    QSize
                )
from PyQt5.QtWidgets import (
                    QApplication, QWidget, QMainWindow, QMenu, QDialog, QLineEdit,
                    QFormLayout, QVBoxLayout, QDialogButtonBox, QSpinBox, QMessageBox
                )
from PyQt5.QtGui import (
                    QPainter, QColor, QPolygonF, QIntValidator, QIcon, QKeySequence,
                    QFont, QFontMetrics
                )
import math
import numpy as np
import sys
from board import Board, Move, MoveTree
from minimax import Minimax
from pypref import SinglePreferences
import os
import threading
from ml import ModelPredictor
from bisect import bisect_left
import traceback


class PredictorFactory(object):
    
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = PredictorFactory()
        return cls._instance
    
    def __init__(self):
        super().__init__()
        self.modelFilesDict = {
                3: "model3.h5",
                5: "model5.h5",
                7: "model7.h5",
            }
        self.modelFilesSizes = sorted(self.modelFilesDict.keys())
        
    def predictor(self,size):
        k = min(bisect_left(self.modelFilesSizes,size),len(self.modelFilesSizes)-1)
        size = self.modelFilesSizes[k]
        return ModelPredictor(size,self.modelFilesDict[size])
                                            

class BoardWidget(QWidget):
        
    def __init__(self, presHexMainWindow, board):
        super().__init__(presHexMainWindow)
        self.presHexMainWindow = presHexMainWindow
        self.preferences = presHexMainWindow.preferences
        self.setMinimumSize(256, 256/math.sqrt(3))
        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus()
        self.board = board
        self.history = []
        self.historyPointer = 0
        self.moveTree = MoveTree(board)
        self.moveTreeNode = self.moveTree.node
        self.modelPredictor = PredictorFactory.instance().predictor(board.size)
        self.minimax = Minimax(heuristic = self.modelPredictor.predict if self.modelPredictor else None)
        self.minimaxWorker = None
        
    def close(self):
        if self.modelPredictor:
            self.modelPredictor.close()
        self.moveTree.save()
        return super().close()
        
    def working(self):
        return self.minimaxWorker is not None
      
    def sizeHint(self):
        return QSize(1024,1024/math.sqrt(3))
        
    def oruv(self):
        o = QPointF(self.width(),self.height())/2
        r = min(self.width()*0.9/math.sqrt(3),self.height()*0.9)/(math.sqrt(3)*(self.board.size+1))
        u = QPointF(3/2,math.sqrt(3)/2)*r
        v = QPointF(3/2,-math.sqrt(3)/2)*r
        return o,r,u,v

    def center(self,o,u,v,i,j):
        return o + u * (i - (self.board.size - 1) / 2) + v * (j - (self.board.size - 1) / 2)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        o,r,u,v = self.oruv()
         
        def center(i,j):
            return self.center(o,u,v,i,j)

        def color(cell):
            if cell > 0:
                return QColor(0,0,0)
            elif cell < 0:
                return QColor(255,255,255)
            else:
                return QColor(128,128,128)

        painter.setPen(QColor(0,0,0))
        painter.setBrush(color(1))
        painter.drawConvexPolygon(center(-1,-1),o,center(-1,self.board.size))
        painter.drawConvexPolygon(center(self.board.size,self.board.size),o,center(self.board.size,-1))
        painter.setBrush(color(-1))
        painter.drawConvexPolygon(center(-1,-1),o,center(self.board.size,-1))
        painter.drawConvexPolygon(center(self.board.size,self.board.size),o,center(-1,self.board.size))
        
        
        font = QFont("Monospace")
        font.setWeight(QFont.Black)
        font.setPixelSize(r/(2*math.sqrt(3))*1.25)
        fontMetrics = QFontMetrics(font)
        painter.setFont(font)
        painter.setPen(QColor(0,0,0))
        for i in range(self.board.size):
            s = Move.columnName(i)
            br = fontMetrics.boundingRect(s)
            c = (2*center(i,-1) + center(i,0))/3
            painter.drawText(c-(br.topLeft()+br.bottomRight())/2,s)
            c = (2*center(i,self.board.size) + center(i,self.board.size-1))/3
            painter.drawText(c-(br.topLeft()+br.bottomRight())/2,s)
            
        painter.setPen(QColor(255,255,255))
        for j in range(self.board.size):
            s = Move.rowName(j)
            br = fontMetrics.boundingRect(s)
            c = (2*center(-1,j) + center(0,j))/3
            painter.drawText(c-(br.topLeft()+br.bottomRight())/2,s)
            c = (2*center(self.board.size,j) + center(self.board.size-1,j))/3
            painter.drawText(c-(br.topLeft()+br.bottomRight())/2,s)

        def hexagon(c):
            return QPolygonF(c + r * QPointF(math.cos(k*2*math.pi/6),math.sin(k*2*math.pi/6)) for k in range(6))
        
        self.centers = np.zeros((self.board.size,)*2,dtype = QPointF)
        self.hexagons = np.zeros((self.board.size,)*2,dtype = QPolygonF)
        
        pathSet = set(self.board.connectedPathPos() or self.board.connectedPathNeg() or [])
        
        for i in range(self.board.size):
            for j in range(self.board.size):
                c = center(i,j)
                h = hexagon(c)
                painter.setPen(QColor(0,0,0))
                painter.setBrush(color(0))
                painter.drawConvexPolygon(h)
                move = Move(i,j)
                if self.lastMove() == move or move in pathSet:
                    if move in pathSet:
                        painter.setBrush(QColor(0,0,255,16))
                    else:
                        painter.setBrush(QColor(255,0,0,16))
                    painter.drawConvexPolygon(h)
                if self.board.cell(move) != 0:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(color(self.board.cell(move)))
                    f = 1 + 0.1 * (self.lastMove() == move or move in pathSet)
                    painter.drawEllipse(QRectF(c-QPointF(1,1)*r*(math.sqrt(5)-1)/2*f,QSizeF(1,1)*r*(math.sqrt(5)-1)*f))
                    

    def move(self,move):
        self.board.move(move)
        self.history = self.history[:self.historyPointer]
        self.history.append(move)
        self.historyPointer += 1
        self.moveTreeNode = self.moveTreeNode.child(move)
        self.update()
        
    def moveBack(self):
        if self.historyPointer > 0:
            self.historyPointer -= 1
            self.board.unmove(self.history[self.historyPointer])
            self.moveTreeNode = self.moveTreeNode.parent
            self.update()

    def moveForward(self):
        if self.historyPointer < len(self.history):
            self.moveTreeNode = self.moveTreeNode.child(self.history[self.historyPointer])
            self.board.move(self.history[self.historyPointer])
            self.historyPointer += 1
            self.update()
            
    def lastMove(self):
        if self.historyPointer > 0: 
            return self.history[self.historyPointer-1]

    def mousePressEvent(self,event):
        if not self.working():
            o,r,u,v = self.oruv()
            p = event.localPos()
            
            det = u.x() * v.y() - u.y() * v.x()
            u_ = QPointF(v.y(),-v.x())/det
            v_ = QPointF(-u.y(),u.x())/det
            
            i__ = ((self.board.size - 1) / 2 + (p - o).x() * u_.x() + (p - o).y() * u_.y())
            j__ = ((self.board.size - 1) / 2 + (p - o).x() * v_.x() + (p - o).y() * v_.y())
            i = None
            j = None
            d = np.inf
            for i_ in range(math.ceil(i__ - 2/3),math.floor(i__ + 2/3) + 1):
                for j_ in range(math.ceil(j__ - 2/3),math.floor(j__ + 2/3) + 1):
                    c_ = self.center(o,u,v,i_,j_)
                    d_ = math.sqrt((c_.x()-p.x())**2 + (c_.y()-p.y())**2)
                    if d_ < d:
                        i, j, d = i_, j_, d_
            try:
                self.move(Move(i,j))
                return
            except Board.InvalidMove:
                pass           
        super().mousePressEvent(event)
        
    def wheelEvent(self,event):
        if not self.working():
            if event.angleDelta().y() > 0:
                try:
                    self.moveBack()
                except Board.InvalidMove:
                    pass
            elif event.angleDelta().y() < 0:
                try:
                    self.moveForward()
                except Board.InvalidMove:
                    pass
            else:
                super().wheelEvent(event)
        else:
            super().wheelEvent(event)
            
    
    class MinimaxWorker(QObject):
        
        finished = pyqtSignal()
        status = pyqtSignal(str)
        exception = pyqtSignal(Exception)
        
        def __init__(self,minimax,size,margin):
            super().__init__()
            self.minimax = minimax
            self.size = size
            self.margin = margin
            self.thread = QThread()
            self.thread.started.connect(self.run)
            self.moveToThread(self.thread)
            self.finished.connect(self.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.finished.connect(self.thread.quit)
            self.aborted_ = False
            self.stopped_ = False
        
        def start(self):
            self.thread.start()
            
        def wait(self):
            self.thread.wait()
        
        @pyqtSlot()
        def run(self):
            threading.current_thread().setName("Minimax")
            try:
                self.minimax.expand(self.size, self.margin, status = self.statusEmit, aborted = self.abortedOrStopped)
            except Exception as e:
                traceback.print_exc()
                self.abort()
                self.exception.emit(e)
            finally:
                self.finished.emit()
        
        def abort(self):
            self.aborted_ = True
            
        def aborted(self):
            return self.aborted_
        
        def stop(self):
            self.stopped_ = True
            
        def stopped(self):
            return self.stopped_
        
        def abortedOrStopped(self):
            return self.aborted() or self.stopped()
        
        def statusEmit(self,s):
            self.status.emit(s)
    
    def startMinimax(self):
        if self.minimaxWorker is None:
            self.minimax.setRootBoard(self.board)
            self.minimaxWorker = self.MinimaxWorker(self.minimax,self.preferences.minimaxSize,self.preferences.minimaxMargin)
            self.minimaxWorker.status.connect(self.presHexMainWindow.status)
            self.minimaxWorker.finished.connect(self.finishedMinimax)
            self.minimaxWorker.exception.connect(self.exceptionMinimax)
            self.minimaxWorker.start()
            self.presHexMainWindow.workingStatusChanged(True)
            QApplication.instance().setOverrideCursor(Qt.WaitCursor)
            
    def stopMinimax(self):
        if self.minimaxWorker:
            self.minimaxWorker.stop()
       
    def abortMinimax(self):     
        if self.minimaxWorker:
            self.minimaxWorker.abort()
        
    def finishedMinimax(self):
        if self.minimaxWorker:
            self.minimaxWorker.wait()
            if not self.minimaxWorker.aborted():        
                move = self.minimax.selectedMove()
                if move:
                    self.move(move)
            self.minimaxWorker = None
            self.presHexMainWindow.workingStatusChanged(False)
            QApplication.instance().restoreOverrideCursor()
        
    def exceptionMinimax(self,exception):
        QMessageBox.critical(self,"Error",str(exception))
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.minimaxWorker:
                self.stopMinimax()
            else:
                self.startMinimax()
        elif event.key() == Qt.Key_Escape:
            if self.minimaxWorker:
                self.minimaxWorker.moveWhenFinished = False
                self.minimaxWorker.abort()
        elif event.key() == Qt.Key_F7:
            self.presHexMainWindow.reset()
                
    def pruneMinimax(self):
        try:
            QApplication.instance().setOverrideCursor(Qt.WaitCursor)
            self.minimax.prune()
        finally:
            QApplication.instance().restoreOverrideCursor()         

    def purgeMinimax(self):
        try:
            QApplication.instance().setOverrideCursor(Qt.WaitCursor)
            self.minimax.purge()
        finally:
            QApplication.instance().restoreOverrideCursor()         
            

class PresHexPreferences(SinglePreferences):
    
    def __init__(self):
        super().__init__(directory = os.path.join(os.path.expanduser("~"),".config/PresHex"))
        self.boardSize = self.preferences.get("boardSize",7)
        self.minimaxSize = self.preferences.get("minimaxSize",2**12)
        self.minimaxMargin = self.preferences.get("minimaxMargin",2**11)
        
    def dict(self):
        return {
                "boardSize": self.boardSize,
                "minimaxSize": self.minimaxSize,
                "minimaxMargin": self.minimaxMargin,
            }
        
    def save(self):
        self.set_preferences(self.dict())

    def __repr__(self):
        return str(self.dict())

           
class PreferencesDialog(QDialog):
    
    def __init__(self,presHexMainWindow):
        super().__init__(presHexMainWindow)
        self.preferences = presHexMainWindow.preferences
        self.boardSizeSpinBox = QSpinBox(self)
        self.boardSizeSpinBox.setMinimum(1)
        self.boardSizeSpinBox.setMaximum(50)
        self.boardSizeSpinBox.setValue(self.preferences.boardSize)
        self.minimaxSizeSpinBox = QSpinBox(self)
        self.minimaxSizeSpinBox.setMinimum(0)
        self.minimaxSizeSpinBox.setMaximum(2**30)
        self.minimaxSizeSpinBox.setValue(self.preferences.minimaxSize)
        self.minimaxMarginSpinBox = QSpinBox(self)
        self.minimaxMarginSpinBox.setMinimum(0)
        self.minimaxMarginSpinBox.setMaximum(2**20)
        self.minimaxMarginSpinBox.setValue(self.preferences.minimaxMargin)
        form = QWidget()
        formLayout = QFormLayout()
        formLayout.addRow(self.tr("&Board size"),self.boardSizeSpinBox)
        formLayout.addRow(self.tr("&Minimax size"),self.minimaxSizeSpinBox)
        formLayout.addRow(self.tr("&Minimax margin"),self.minimaxMarginSpinBox)
        form.setLayout(formLayout)
        layout = QVBoxLayout()
        layout.addWidget(form)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
        self.setWindowTitle("Preferences")
        self.setFixedWidth(256)
        self.setModal(True)

    def show(self):
        super().show()
        self.setFixedSize(self.size())

    def accept(self):
        self.preferences.boardSize = self.boardSizeSpinBox.value()
        self.preferences.minimaxSize = self.minimaxSizeSpinBox.value()
        self.preferences.minimaxMargin = self.minimaxMarginSpinBox.value()
        self.preferences.save()
        super().accept()
        
    def workingStatusChanged(self, working):
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(not working)

class PresHexMainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.preferences = PresHexPreferences()
        self.preferencesDialog = PreferencesDialog(self)
        fileMenu = QMenu("&File", self)
        fileMenu.addAction("&Preferences", self.preferencesDialogExec, QKeySequence(Qt.Key_F2))
        fileMenu.addAction("&Reset", self.reset, QKeySequence(Qt.Key_F7))
        fileMenu.addAction("&Exit", self.exit)
        self.menuBar().addMenu(fileMenu)
        thinkMenu = QMenu("&Think", self)
        self.startThinkingAction = thinkMenu.addAction("&Start", self.startThinking, QKeySequence(Qt.Key_F1))
        self.stopThinkingAction = thinkMenu.addAction("&Stop", self.stopThinking, QKeySequence(Qt.Key_F1))
        self.stopThinkingAction.setEnabled(False)
        self.pruneMinimaxAction = thinkMenu.addAction("&Prune memory", self.pruneMinimax)
        self.purgeMinimaxAction = thinkMenu.addAction("&Purge memory", self.purgeMinimax)
        self.menuBar().addMenu(thinkMenu)
        self.statusBar().show()
        self.updateBoardSize()
       
    def boardWidget(self):
        centralWidget = self.centralWidget()
        if isinstance(centralWidget,BoardWidget):
            return centralWidget
        
    def reset(self):
        self.updateBoardSize(force = True)
        
    def updateBoardSize(self, force = False):
        boardWidget = self.boardWidget()
        if boardWidget:
            if not force and boardWidget.board.size == self.preferences.boardSize:
                return
            else:
                boardWidget.close()
                boardWidget.deleteLater()
        self.setCentralWidget(BoardWidget(self,Board(size = self.preferences.boardSize)))
        
        
    def workingStatusChanged(self, working):
        self.preferencesDialog.workingStatusChanged(working)
        self.startThinkingAction.setEnabled(not working)
        self.stopThinkingAction.setEnabled(working)
        self.pruneMinimaxAction.setEnabled(not working)
        self.purgeMinimaxAction.setEnabled(not working)
        
    def working(self):
        boardWidget = self.boardWidget()
        if boardWidget:
            return boardWidget.working()
        else:
            return False
        
    def preferencesDialogExec(self):
        self.preferencesDialog.show()
        self.preferencesDialog.exec_()
        if self.preferencesDialog.result() == self.preferencesDialog.Accepted:
            self.updateBoardSize()
            
    def startThinking(self):
        boardWidget = self.boardWidget()
        if boardWidget:
            boardWidget.startMinimax()
    
    def stopThinking(self):
        boardWidget = self.boardWidget()
        if boardWidget:
            boardWidget.stopMinimax()
            
    def pruneMinimax(self):
        boardWidget = self.boardWidget()
        if boardWidget:
            boardWidget.pruneMinimax()
        
    def purgeMinimax(self):
        boardWidget = self.boardWidget()
        if boardWidget:
            boardWidget.purgeMinimax()

    def status(self,s):
         self.statusBar().showMessage(s)
         print(s,file = sys.stderr)
                
    def exit(self):
        self.close()
        
    def closeEvent(self, event):
        boardWidget = self.boardWidget()
        if boardWidget:
            boardWidget.close()
        super().closeEvent(event)


def presHexIcon():
    if getattr(sys, "frozen", False):
        return QIcon(os.path.join(sys._MEIPASS,"preshex.png"))
    else:
        return QIcon(os.path.join(os.path.dirname(os.path.realpath(__file__)),"icons/preshex.png"))
           
class PresHexApplication(QApplication):
    
    def __init__(self):
        super().__init__(sys.argv)
        self.setApplicationName("PresHex")
        self.setWindowIcon(presHexIcon())

def gui():
    app = PresHexApplication()
    window = PresHexMainWindow()
    window.show()
    app.exec_()
        
if __name__ == '__main__':
    gui()
        
        
        
