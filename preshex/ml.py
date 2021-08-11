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
import hashlib
import time

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
    
    def __init__(self,modelFile,boardSize,batchSize = multiprocessing.cpu_count(),k = 64):
        self.modelFile = modelFile
        self.k = k
        self.input = np.zeros((batchSize,) + (boardSize,)*2 + (3,))
        self.input[:,0,:,1] = 1
        self.input[:,-1,:,1] = 1
        self.input[:,:,0,-1] = -1
        self.input[:,:,-1,-1] = -1
        self.n = 0
        self.pending = np.zeros(batchSize, dtype = bool)
        self.ready = np.zeros(batchSize, dtype = bool)
        self.thread = Thread(target = self.run, name = "Predictor", daemon = True)
        self.condition = Condition()
        self.stop = False
        self.thread.start()
        
    def close(self):
        with self.condition:
            self.stop = True
            self.condition.notifyAll()
        self.thread.join()

    def postPredict(self,x):
        return x/(1+np.abs(x)**self.k)**(1/self.k)

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
                while not self.stop and (self.n <= 0 or self.ready.any()):
                    self.condition.wait()
                if self.stop:
                    break
                self.prediction = self.postPredict(model.predict(self.input[:self.n]))
                self.ready[:self.n] = True
                self.n = 0
                self.condition.notifyAll()
        
    def predict(self,board):
        with self.condition:
            while self.n >= len(self.input) or self.ready[self.n]:
                self.condition.wait()
            i = self.n
            self.input[i,:,:,0] = board.cells
            self.n += 1
            self.condition.notifyAll()
            while not self.ready[i]:
                self.condition.wait()
            self.ready[i] = False
            self.condition.notifyAll()
            return self.prediction[i,0]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()    
        
def terminalSmallMinimax(boardSize, predictor = None,  target = None, initialSize = 0, deltaSize = 2048,):
    board = Board(boardSize)
    game = []
    while board:
        game.append(board)
        boards = []
        accumulatedPredict = []
        s = 0
        for move in board.possibleMoves():
            boards.append(board.moveOnCopy(move))
            if predictor:
                s += 1 + predictor.predict(board) * board.turn
            accumulatedPredict.append(s)
        if boards:
            if s > 0:
                board = boards[bisect.bisect(accumulatedPredict,s*random.random())]
            else:
                board = random.choice(boards)
        else:
            board = None
    minimax = Minimax(heuristic = predictor.predict) if predictor else Minimax()
    size = initialSize
    for board in game[::-1]:
        minimax.setRootBoard(board)
        reached = minimax.expand(size,1000,target = target,statusInterval = 60 * multiprocessing.cpu_count())
        if not reached:
            break
        size += deltaSize
    return minimax

def hashCells(cells):
    hash = hashlib.md5()
    hash.update(cells.tobytes())
    digest = hash.digest()[:(sys.maxsize.bit_length()+1)//8]
    return int.from_bytes(digest, byteorder = "big", signed = True)
    
def generateMinimaxTrainDataPredictor(boardSize, predictor = None, target = None, size = 2**22, deltaSize = 2048, terminal = False):
    lock = RLock()
    cells = []
    values = []
    class TerminalSmallMinimaxThread(Thread):
        def __init__(self):
            super().__init__(name = "terminalSmallMinimax")
        
        def run(self):
            while True:
                minimax = terminalSmallMinimax(boardSize,predictor,target = target,deltaSize = deltaSize)
                with lock:
                    for board,value in minimax.collectLeafValues(terminal = terminal):
                        cells.append(board.cells)
                        values.append(value)
                    print(f"----> data size:{len(cells)}",file = sys.stderr)
                    if len(cells) >= size:
                        break
    threads = []
    for i in range(multiprocessing.cpu_count()):
        thread = TerminalSmallMinimaxThread()
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return np.array(cells), np.array(values)


def generateMinimaxTrainData(boardSize,  modelFile = None, target = None, size = 2**22, deltaSize = 2048, terminal = False):
    if modelFile:
        with Predictor(modelFile,boardSize) as predictor:
            return generateMinimaxTrainDataPredictor(boardSize, predictor, target = target, size = size, deltaSize = deltaSize, terminal = terminal)
    else:
        return generateMinimaxTrainDataPredictor(boardSize, size = size, target = target, deltaSize = deltaSize, terminal = terminal)

def saveMinimaxTrainData(boardSize, dataFile, modelFile = None, target = None, size = 2**22, deltaSize = 2048, terminal = False):
    cells, values = generateMinimaxTrainData(boardSize,modelFile,target,size,deltaSize,terminal)
    np.savez(dataFile,cells = cells,values = values)

    
def formatMinimaxTrainData(cells,values,validation = 1/16):
    hashes = np.apply_along_axis(lambda w:hashCells(w.reshape(cells.shape[1:])),1,cells.reshape((cells.shape[0],np.product(cells.shape[1:]))))
    validation = hashes/(sys.maxsize+1) < validation * 2 - 1
    train = np.logical_not(validation)
    input = np.zeros(cells.shape + (3,))
    output = np.zeros(values.shape + (1,))
    input[:,0,:,1] = 1
    input[:,-1,:,1] = 1
    input[:,:,0,-1] = -1
    input[:,:,-1,-1] = -1
    input[:,:,:,0] = cells
    output[:,0] = values
    x,y = input[train], output[train]
    x_val, y_val = input[validation], output[validation]
    return x,y,x_val,y_val
 
def dataMinError(dataFile):
    data = np.load(dataFile)
    cells = data["cells"]
    values = data["values"]
    means = np.zeros(len(cells))
    ucells, icells = np.unique(cells,axis=0, return_inverse = True)
    for j in range(len(ucells)):
        i = icells == j
        means[i] = np.mean(values[i])
    return np.mean(np.abs(values-means))
    
        
def minimaxTrain(dataFile,modelFile,fraction = 1, validation = 1/16):
    data = np.load(dataFile)
    cells = data["cells"]
    values = data["values"]
    pick = np.random.random(len(cells)) < fraction
    cells = cells[pick]
    values = values[pick]
    x, y, x_val, y_val = formatMinimaxTrainData(cells,values,validation)
    print(f"training: {len(x)}\tvalidation: {len(x_val)}\tratio: {len(x_val)/len(x)}")
    del cells, values
    model = load_model(modelFile)
    model.summary()
    if len(x_val) > 0:
        lastValLoss = model.evaluate(x_val,y_val,verbose = 2)
        print(f"Initial valLoss: {lastValLoss}")
        round = 0
        while True:
            t0 = time.time()
            history = model.fit(
                    x,
                    y,
                    batch_size = 64,
                    shuffle = True,
                    epochs = 1,
                    verbose = 0,
                    validation_data = (x_val, y_val))
            loss = history.history["loss"][-1]
            valLoss = history.history['val_loss'][-1]
            print(f"round: {round}\tdtime:{time.time()-t0}\tloss: {loss}\tvalLoss: {valLoss}\t{'(*)' if valLoss < lastValLoss else ''}")
            if valLoss < lastValLoss:
                model.save(modelFile)
                lastValLoss = valLoss
            round += 1
    else:
        lastLoss = model.evaluate(x,y,verbose = 2)
        print(f"Initial loss: {lastLoss}")
        round = 0
        while True:
            t0 = time.time()
            history = model.fit(
                    x,
                    y,
                    batch_size = 64,
                    shuffle = True,
                    epochs = 1,
                    verbose = 0)        
            loss = history.history['loss'][-1]
            print(f"round: {round}\tdtime:{time.time()-t0}\tloss: {loss}\t{'(*)' if loss < lastLoss else ''}")
            if loss < lastLoss:
                model.save(modelFile)
                lastLoss = loss
        
    
def modelDesign(boardSize):
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2+(3,), target_shape = (boardSize,)*2+(3,)))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Flatten())
    model.add(Dense(units = 16, activation = "tanh"))
    model.add(Dense(units = 1, activation = "tanh"))
    model.compile(loss = "mean_squared_error", optimizer = "SGD")
#    model = load_model("model_new4.h5")
    model.summary()
    modelOrig = load_model("model7.h5")
    for i,layer in enumerate(model.layers):
        if i in (3,6,9):
            w = layer.get_weights()
            w[0][:] = 0
            w[0][1,1] = np.identity(64)
            w[1][:] = 0
            layer.set_weights(w)
        else:
            layer.set_weights(modelOrig.layers[i - (i > 3) - (i > 6) - (i > 9)].get_weights())
    data = np.load("data7.npz")
    cells = data["cells"]
    values = data["values"]
    x, y, x_val, y_val = formatMinimaxTrainData(cells,values)
    del cells, values
    lastValLoss = model.evaluate(x_val,y_val,verbose = 2)
    print(f"Initial validation loss: {lastValLoss}")
    model.save("model7_new.h5")

def modelDesign3(boardSize,modelFile):
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2+(3,), target_shape = (boardSize,)*2+(3,)))
    model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = "tanh"))
    model.add(Flatten())
    model.add(Dense(units = 64, activation = "tanh"))
    model.add(Dense(units = 1, activation = "tanh"))
    model.compile(loss = "mean_squared_error", optimizer = "SGD")
    model.summary()
    model.save(modelFile)

def modelAlter(modelFile):
    model = load_model(modelFile)
    pass
    model.save(modelFile)
    
def checkAccuracy(boardSize,modelFile):
    minimax = terminalSmallMinimax(boardSize, initialSize = 2**20, deltaSize = 0)
    terminals = list(filter(lambda n:n.bestLeaf()[0].board.winner,minimax.nodes.values()))
    input = np.zeros((len(terminals),) + (boardSize,)*2 + (3,))
    input[:,0,:,1] = 1
    input[:,-1,:,1] = 1
    input[:,:,0,-1] = -1
    input[:,:,-1,-1] = -1
    input[:,:,:,0] = list(map(lambda n:n.board.cells,terminals))
    model = load_model(modelFile)
    def postPredict(x, k = 64):
        return x/(1+np.abs(x)**k)**(1/k)
    predictions = postPredict(model.predict(input))[:,0]
    values = predictions * list(map(lambda n:n.bestLeaf()[0].board.winner,terminals))
    errors = 1 - values
    error = np.mean(errors**2)
    pass

if __name__ == '__main__':
    #heuristicTrain(7)
    #heuristicTest(7)
    #minimaxTrain("data7.npz","model7.h5",fraction = 1)
    #modelDesign(7)
    #saveMinimaxTrainData(7,"data7.npz","model7.h5",deltaSize = 2**13)
    #modelDesign3(3,"model3_new.h5")
    #print(dataMinError("data3.npz"))
    #minimaxTrain("data3.npz","model3_new.h5",validation=0)
    #modelAlter("model7_sq.h5")
    checkAccuracy(7,"model7.h5")

    
    
    
    
    
    
    
    
    

