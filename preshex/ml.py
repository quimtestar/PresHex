import numpy as np
import random
import multiprocessing
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from minimax import Minimax
from board import Board, Move
import bisect
from collections import deque
import sys
from threading import Thread, Condition, RLock, Lock

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
tf.get_logger().setLevel('INFO')


def heuristicDataGenerator(boardSize, batchSize = 256):
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

def makeHeuristicDataSegment(boardSize, size, segment):
    x = []
    y = []
    for i,(inputBatch,outputBatch) in enumerate(heuristicDataGenerator(boardSize,256)):
        x.append(inputBatch)
        y.append(outputBatch)
        print(f"s: {segment}  i: {i}")
        if i * 256 >= size:
            break
    return np.concatenate(x), np.concatenate(y)
                
def makeHeuristicData(boardSize, size, processes = multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes) as pool:
        x = []
        y = []
        for result in [pool.apply_async(makeHeuristicDataSegment, args=(boardSize, size//processes, segment)) for segment in range(processes)]:
            x_, y_ = result.get()
            x.append(x_)
            y.append(y_)
        return np.concatenate(x), np.concatenate(y)
        
def generateModel(boardSize):
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2+(3,), target_shape = (boardSize,)*2+(3,)))
    model.add(Conv2D(filters = 16,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Flatten())
    model.add(Dense(units = 16, activation = "tanh"))
    model.add(Dense(units = 1, activation = "tanh"))
    model.compile(loss = "mean_absolute_error", optimizer = "SGD")
    return model
        

def heuristicTrain(boardSize):
    model = generateModel(boardSize)
    #model = load_model("model.h5")
    model.summary()

    x, y = makeHeuristicData(boardSize, 2**18)

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
    
    
def heuristicTest(boardSize):
    model = load_model("model.h5")
    x, y = makeData(boardSize, 2**12)
    ev = model.evaluate(x,y, verbose = 2)
    print(ev)     
            

class Predictor(object):
    
    def __init__(self,modelFile,size):
        self.modelFile = modelFile
        self.input = np.zeros((1,) + (size,)*2 + (3,))
        self.input[:,0,:,1] = 1
        self.input[:,-1,:,1] = 1
        self.input[:,:,0,-1] = -1
        self.input[:,:,-1,-1] = -1
        self.thread = Thread(target = self.run, name = "Predictor", daemon = True)
        self.condition = Condition()
        self.board = None
        self.stop = False
        self.thread.start()
        
    def close(self):
        with self.condition:
            self.stop = True
            self.condition.notifyAll()
        self.thread.join()

    def run(self):
        with self.condition:
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))
            tf.get_logger().setLevel('INFO')            
            model = load_model(self.modelFile)
            while True:
                while not self.stop and self.board is None:
                    self.condition.wait()
                if self.stop:
                    break
                self.input[:,:,:,0] = self.board.cells
                self.prediction = model.predict(self.input)[0,0]
                self.board = None
                self.condition.notifyAll()
        
    def predict(self,board):
        with self.condition:
            while self.board is not None:
                self.condition.wait()
            self.board = board
            self.prediction = None
            self.condition.notifyAll()
            while self.prediction is None:
                self.condition.wait()
            return self.prediction
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()    
    
def minimaxTrainOld(boardSize):
    cells = []
    values = []
    for i in range(10):
        minimax = Minimax(Board(boardSize), heuristic = Predictor("model.h5",boardSize).predict)
        minimax.expand(100000,1000,uniformDepthFactor = 1,uniformDepthRandomization = 0.1)
        for board,value in minimax.collectLeafValues():
            cells.append(board.cells)
            values.append(value)
        del minimax
    input = np.zeros((len(cells),) + (boardSize,)*2 + (3,))
    output = np.zeros((len(values),) + (1,))
    input[:,0,:,1] = 1
    input[:,-1,:,1] = 1
    input[:,:,0,-1] = -1
    input[:,:,-1,-1] = -1
    input[:,:,:,0] = cells
    output[:,0] = values
    del cells, values
    x,y = input, output
    model = load_model("model.h5")
    model.summary()
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
    
    
def terminalSmallMinimax(boardSize,predictor):
    board = Board(boardSize)
    game = []
    while board:
        game.append(board)
        boards = []
        accumulatedPredict = []
        s = 0
        for move in board.possibleMoves():
            boards.append(board.moveOnCopy(move))
            s += 1 + predictor.predict(board) * board.turn
            accumulatedPredict.append(s)
        if boards:
            board = boards[bisect.bisect(accumulatedPredict,s*random.random())]
        else:
            board = None
    minimax = Minimax(heuristic = predictor.predict)
    for board in game[::-1]:
        minimax.setRootBoard(board)
        fully = minimax.expand(25000,1000,statusInterval = 60 * multiprocessing.cpu_count())
        if not fully:
            break
    return minimax
    
def minimaxTrain(boardSize):
    with Predictor("model.h5",boardSize) as predictor:
        lock = RLock()
        cells = []
        values = []
        class TerminalSmallMinimaxThread(Thread):
            def __init__(self):
                super().__init__(name = "terminalSmallMinimax")
            
            def run(self):
                while True:
                    minimax = terminalSmallMinimax(boardSize,predictor)
                    with lock:
                        for board,value in minimax.collectLeafValues():
                            cells.append(board.cells)
                            values.append(value)
                        print(f"----> data size:{len(cells)}",file = sys.stderr)
                        if len(cells) >= 1000000:
                            break
        threads = []
        for i in range(multiprocessing.cpu_count()):
            thread = TerminalSmallMinimaxThread()
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    input = np.zeros((len(cells),) + (boardSize,)*2 + (3,))
    output = np.zeros((len(values),) + (1,))
    input[:,0,:,1] = 1
    input[:,-1,:,1] = 1
    input[:,:,0,-1] = -1
    input[:,:,-1,-1] = -1
    input[:,:,:,0] = cells
    output[:,0] = values
    del cells, values
    x,y = input, output
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    model = load_model("model.h5")
    model.summary()
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
        
    
if __name__ == '__main__':
    #heuristicTrain(7)
    #heuristicTest(7)
    minimaxTrain(7)

    
    
    
    
    
    
    
    
    

