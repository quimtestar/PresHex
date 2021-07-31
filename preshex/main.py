from board import Board, Move
from gui import PresHexMainWindow, PresHexApplication
from minimax import Minimax
import random
from PyQt5.QtWidgets import (
                    QApplication, QMainWindow
                )
import sys
import math
from ml import Predictor

def gui():
    app = PresHexApplication()
    window = PresHexMainWindow()
    window.show()
    app.exec_()

def minimax():
    board = Board(size = 7)
    minimax = Minimax(board,Predictor(board.size).predict)
    minimax.expand(250000,1000)

if __name__ == '__main__':
    gui()
    #minimax()

    
    
    