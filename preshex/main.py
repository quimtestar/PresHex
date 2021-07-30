from board import Board, Move
from gui import PresHexMainWindow, PresHexApplication
from minimax import Minimax
import random
from PyQt5.QtWidgets import (
                    QApplication, QMainWindow
                )
import sys
import math

def gui():
    app = PresHexApplication()
    window = PresHexMainWindow()
    window.show()
    app.exec_()

def minimax():
    board = Board(size = 4)
    minimax = Minimax(board,Board.pendingDistanceHeuristic)
    minimax.expand(2**22,2**10)

if __name__ == '__main__':
    gui()
    #minimax()

    
    
    