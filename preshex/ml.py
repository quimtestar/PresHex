import numpy as np
import random
from multiprocessing import Pool
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from minimax import Minimax
from board import Board

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
tf.get_logger().setLevel('INFO')


def generator(boardSize, batchSize = 256):
    def cellsGenerator():
        board = Board(boardSize)
        while True:
            yield board.cells, board.pendingDistanceHeuristic()
            moves = list(board.possibleMoves())
            if moves:
                board.move(random.choice(moves))
            else:
                break
    inputBatch = np.zeros((batchSize,) + (boardSize,)*2 + (3,))
    inputBatch[:,0,:,1] = 1
    inputBatch[:,-1,:,1] = 1
    inputBatch[:,:,0,-1] = -1
    inputBatch[:,:,-1,-1] = -1
    
    outputBatch = np.zeros((batchSize,1))
    i = 0
    while True:
        for cells,value in cellsGenerator():
            inputBatch[i,:,:,0] = cells
            outputBatch[i] = value
            i += 1
            if i >= batchSize:
                p = np.random.permutation(i)
                yield inputBatch[p], outputBatch[p]
                i = 0

def makeDataSegment(boardSize, size, segment):
    x = []
    y = []
    for i,(inputBatch,outputBatch) in enumerate(generator(boardSize,256)):
        x.append(inputBatch)
        y.append(outputBatch)
        print(f"s: {segment}  i: {i}")
        if i * 256 >= size:
            break
    return np.concatenate(x), np.concatenate(y)
                
def makeData(boardSize, size, processes = 16):
    with Pool(processes) as pool:
        x = []
        y = []
        for result in [pool.apply_async(makeDataSegment, args=(boardSize, size//processes, segment)) for segment in range(processes)]:
            x_, y_ = result.get()
            x.append(x_)
            y.append(y_)
        return np.concatenate(x), np.concatenate(y)
        
        

def initialize(boardSize):
    """
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2+(3,), target_shape = (boardSize,)*2+(3,)))
    model.add(Conv2D(filters = 16,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Flatten())
    model.add(Dense(units = 16, activation = "tanh"))
    model.add(Dense(units = 1, activation = "tanh"))
    model.compile(loss = "mean_absolute_error", optimizer = "SGD")
    """
    model = load_model("model.h5")
    model.summary()
    

    x, y = makeData(boardSize, 2**20)

    while True:
        history = model.fit(
                x,
                y,
                batch_size = 64,
                shuffle = True,
                epochs = 1,
                verbose = 2,
                validation_split = 0.0625)
        
        model.save("model.h5")
    
    
def test(boardSize):
    model = load_model("model.h5")
      
    x, y = makeData(boardSize, 2**12)
    ev = model.evaluate(x,y, verbose = 2)
    print(ev)     
            
    
    
    
    
    
if __name__ == '__main__':
    initialize(7)
    #test(7)

    
    
    
    
    
    
    
    
    

