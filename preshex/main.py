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
    board.move(Move(1,3))
    board.move(Move(0,1))
    board.move(Move(2,0))
    board.move(Move(3,2))
    minimax = Minimax(board)
    minimax.expandToSize(100000,1000)

if __name__ == '__main__':
    gui()

    
    
    