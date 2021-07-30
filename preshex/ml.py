import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from minimax import Minimax
from board import Board


def initialize(boardSize):
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2, target_shape = (boardSize,)*2+(1,)))
    model.add(Conv2D(filters = 1,kernel_size = (3,3),activation="tanh"))
    model.add(Flatten())
    model.add(Dense(units = 1))
    model.compile(loss = "mean_absolute_error", optimizer = "SGD")
    model.summary()
    
    def generator(boardSize, batchSize = 64):
        def cellsGenerator():
            board = Board(boardSize)
            while True:
                yield board.cells, board.pendingDistanceHeuristic()
                moves = list(board.possibleMoves())
                if moves:
                    board.move(random.choice(moves))
                else:
                    break
        cellsBatch = np.zeros((batchSize,) + (boardSize,)*2)
        valueBatch = np.zeros(batchSize)
        i = 0
        while True:
            for cells,value in cellsGenerator():
                cellsBatch[i] = cells
                valueBatch[i] = value
                i += 1
                if i >= batchSize:
                    yield cellsBatch, valueBatch
                    i = 0

    while True:    
        history = model.fit_generator(
                generator(boardSize),
                steps_per_epoch = 128,
                epochs = 1,
                verbose = 1)
        model.save("model.h5")
    
    
    
    
    
if __name__ == '__main__':
    initialize(3)

    
    
    
    
    
    
    
    
    

