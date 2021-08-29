import numpy as np
import random
import multiprocessing
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Reshape
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from minimax import Minimax
from board import Board, Move, MoveTree
import bisect
from collections import deque
import sys
from threading import Thread, Condition, RLock, Lock
import hashlib
import time
import math
from abc import ABC, abstractmethod
import traceback

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
tf.get_logger().setLevel('INFO')


def generateModel(boardSize,filters,extra):
    model = Sequential()
    model.add(Reshape(input_shape = (boardSize,)*2+(3,), target_shape = (boardSize,)*2+(3,)))
    for i in range((boardSize-1)//2):
        model.add(Conv2D(filters = filters,kernel_size = (3,3),activation = "tanh"))
        for j in range(extra):
            model.add(Conv2D(filters = filters,kernel_size = (3,3),padding = "same", activation = "tanh"))
    model.add(Flatten())
    model.add(Dense(units = 2**(math.floor(math.log(filters,2))//2+1), activation = "tanh"))
    model.add(Dense(units = 1, activation = "tanh"))
    model.compile(loss = "mean_squared_error", optimizer = "SGD")
    return model

class Predictor(ABC):

    def __init__(self,boardSize):
        self.boardSize = boardSize

    @abstractmethod
    def predict(self,board):
        pass
 
    def fastMove(self, board, e = 1):
        movedBoards = []
        acc = []
        infinites = []
        s = 0
        for move in board.possibleMoves():
            movedBoard = board.moveOnCopy(move)
            movedBoards.append(movedBoard)
            v = self.predict(movedBoard) * board.turn
            if math.isclose(v,1):
                infinites.append(movedBoard)
                s += math.inf
            else:
                s += math.pow((1 + v)/(1 - v),e)
            acc.append(s)
        if infinites:
            return random.choice(infinites)
        elif s > 0:
            return movedBoards[bisect.bisect(acc,s*random.random())]
        elif movedBoards:
            return random.choice(movedBoards)
                    
        
    def successRatio(self,other,turn,n = 1000, e = 1):
        predictorSelf = self
        
        class State(object):
            
            def __init__(self,turn):
                self.lock = RLock()
                self.turn = turn
                self.count = 0
                self.wins = 0
                
            def score(self,win):
                with self.lock:
                    self.count += 1
                    self.wins += win
                    if self.count % 10 == 0:
                        self.trace()
            
            def trace(self):
                with self.lock:
                    print(f"-----> turn:{turn}\tcount:{self.count}\twins:{self.wins}\tratio:{self.ratio()}",file = sys.stderr)
            
            def ratio(self):
                with self.lock:
                    return self.wins/self.count
                    
                
        state = State(turn)  
        class SuccessRatioThread(Thread):
            def __init__(self):
                super().__init__(name = "successRatio")
            
            def run(self):
                while True:
                    board = Board(predictorSelf.boardSize)
                    while not board.winner:
                        for predictor in [predictorSelf,other][::turn]:
                            if board.winner:
                                break
                            board = predictor.fastMove(board, e)
                    state.score(board.winner * turn > 0)
                    if state.count >= n:
                        break

        threads = []
        for i in range(multiprocessing.cpu_count()):
            thread = SuccessRatioThread()
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        state.trace()
        return state.ratio()

    def successRatioAttack(self,other, n = 1000, e = 1):
        return self.successRatio(other,1,n,e)
    
    def successRatioDefend(self,other, n = 1000, e = 1):
        return self.successRatio(other,-1,n,e)
    
    def successRatioAverage(self,other,n = 1000, e = 1):
        
        class successRatioThread(Thread):
            
            def __init__(self,name,function):
                super().__init__(name = name)
                self.function = function
                self.result = None

            def run(self):
                self.result = self.function(other,n,e)
        
        attackThread = successRatioThread("successRatioAttack", self.successRatioAttack)
        attackThread.start()
        defendThread = successRatioThread("successRatioDefend", self.successRatioDefend)
        defendThread.start()
        attackThread.join()
        defendThread.join()
        
        return (attackThread.result + defendThread.result)/2

class VoidPredictor(Predictor):
    
    def predict(self,board):
        return board.winner
    
class ModelPredictor(Predictor):
    
    def __init__(self,boardSize,*modelFiles,batchSize = multiprocessing.cpu_count(),k = 64):
        super().__init__(boardSize)
        self.filler = np.fromfunction(lambda i,j:np.logical_xor(i>j,i+j<boardSize).astype(int)-np.logical_xor(i<j,i+j<boardSize).astype(int),(boardSize,)*2,dtype=int)
        self.modelFiles = modelFiles
        self.batchSize = batchSize
        self.k = k
        self.input = np.zeros((len(modelFiles),batchSize,) + (boardSize,)*2 + (3,))
        self.input[:,:,0,:,1] = 1
        self.input[:,:,-1,:,1] = 1
        self.input[:,:,:,0,-1] = -1
        self.input[:,:,:,-1,-1] = -1
        self.prediction = np.zeros((len(modelFiles),batchSize,1))
        self.n = [0] * len(modelFiles)
        self.pending = np.zeros((len(modelFiles),batchSize), dtype = bool)
        self.ready = np.zeros((len(modelFiles),batchSize), dtype = bool)
        self.thread = Thread(target = self.run, name = "ModelPredictor", daemon = True)
        self.condition = Condition()
        self.exception = None
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
            try:
                models = list(map(load_model,self.modelFiles))
                while True:
                    while not self.stop and (all(map(lambda n_:n_<=0,self.n)) or self.ready.any()):
                        self.condition.wait()
                    if self.stop:
                        break
                    for m,model in enumerate(models):
                        if self.n[m] > 0:
                            self.prediction[m,:self.n[m]] = self.postPredict(model.predict(self.input[m,:self.n[m]]))
                            self.ready[m,:self.n[m]] = True
                            self.n[m] = 0
                            self.condition.notifyAll()
            except Exception as e:
                traceback.print_exc()
                self.exception = e
                self.condition.notifyAll()                
        
    def predict(self,board,m = 0):
        with self.condition:
            while self.exception is None and (self.n[m] >= self.batchSize or self.ready[m,self.n[m]]):
                self.condition.wait()
            if self.exception:
                raise self.exception
            i = self.n[m]
            if board.size == self.boardSize:
                self.input[m,i,:,:,0] = board.cells
            elif board.size < self.boardSize:
                self.input[m,i,:,:,0] = self.filler
                k0 = (self.boardSize - board.size + 1)//2
                k1 = (self.boardSize + board.size + 1)//2
                self.input[m,i,k0:k1,k0:k1,0] = board.cells
            elif board.size > self.boardSize:
                k0 = (board.size - self.boardSize + 1)//2
                k1 = (board.size + self.boardSize + 1)//2
                self.input[m,i,:,:,0] = board.cells[k0:k1,k0:k1]
            else:
                self.input[m,i,:,:,0] = 0
            self.n[m] += 1
            self.condition.notifyAll()
            while self.exception is None and not self.ready[m,i]:
                self.condition.wait()
            if self.exception:
                raise self.exception
            self.ready[m,i] = False
            self.condition.notifyAll()
            return self.prediction[m,i,0]

    def singleModelPredictor(self,m):
        class SingleModelPredictor(Predictor):
            
            def __init__(self,modelPredictor,m):
                super().__init__(modelPredictor.boardSize)
                self.modelPredictor = modelPredictor
                self.m = m
                
            def predict(self,board):
                return self.modelPredictor.predict(board,self.m)
        return SingleModelPredictor(self,m)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
def terminalSmallMinimaxes(moveTree, predictor = None,  target = None, initialSize = 0, deltaSize = 2048, selectionExponent = 1):
    if predictor is None:
        predictor = VoidPredictor(moveTree.board.size)
    minimax = Minimax(heuristic = predictor.predict) if predictor else Minimax()
    keepOn = False
    for board,node in moveTree.postOrder():
        if not node.children:
            if keepOn:
                yield minimax
            keepOn = True
            game = []
            while board:
                game.append(board)
                board = predictor.fastMove(board, selectionExponent)
            size = initialSize
            for board in game[::-1]:
                minimax.setRootBoard(board)
                reached = minimax.expand(size,1024,target = target,statusInterval = 10 * multiprocessing.cpu_count())
                if not reached:
                    keepOn = False
                    yield minimax
                    break
                size += deltaSize
        else:
            if keepOn:
                minimax.setRootBoard(board)
                reached = minimax.expand(size,1024,target = target,statusInterval = 10 * multiprocessing.cpu_count())
                if not reached:
                    keepOn = False
                    yield minimax
                size += deltaSize
    if keepOn:
        minimax.selectionExponent = selectionExponent
        minimax.expand(size,1024,target = None,statusInterval = 10 * multiprocessing.cpu_count())        
        yield minimax


def generateMinimaxTrainDataPredictor(boardSize, predictor = None, useMoveTrees = False, targetFrom = None, targetAlpha = 0, size = 2**22, deltaSize = 2048, selectionExponent = 1, terminal = False):
    lock = RLock()
    if useMoveTrees:
        moveTreesIterator = iter(MoveTree.loadAll(boardSize,delete = True)) 
    cells = []
    values = []
    class TerminalSmallMinimaxThread(Thread):
        def __init__(self):
            super().__init__(name = "terminalSmallMinimax")
        
        def run(self):
            while True:
                if useMoveTrees:
                    with lock:
                        try:
                            moveTree = next(moveTreesIterator)
                        except StopIteration:
                            moveTree = MoveTree(Board(boardSize))
                else:
                    moveTree = MoveTree(Board(boardSize))
                if targetFrom is None:
                    target = None
                elif targetAlpha == 0:
                    target = targetFrom + random.random() * (1 - targetFrom)
                else:
                    x = random.random()
                    target = 1/targetAlpha * math.log(x * math.exp(targetAlpha) + (1 - x) * math.exp(targetAlpha * targetFrom))
                for minimax in terminalSmallMinimaxes(moveTree,predictor,target = target,deltaSize = deltaSize, selectionExponent = selectionExponent):
                    with lock:
                        for board,value in minimax.collectLeafValues(terminal = terminal):
                            cells.append(board.cells)
                            values.append(value)
                        print(f"----> data size:{len(cells)}",file = sys.stderr)
                with lock:
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


def generateMinimaxTrainData(boardSize,  modelFile = None, useMoveTrees = False, targetFrom = None, targetAlpha = 0, size = 2**22, deltaSize = 2048, selectionExponent = 1, terminal = False):
    if modelFile:
        with ModelPredictor(boardSize,modelFile) as predictor:
            return generateMinimaxTrainDataPredictor(boardSize, predictor, useMoveTrees = useMoveTrees, targetFrom = targetFrom, targetAlpha = targetAlpha, size = size, deltaSize = deltaSize, selectionExponent = selectionExponent, terminal = terminal)
    else:
        return generateMinimaxTrainDataPredictor(boardSize, useMoveTrees = useMoveTrees, targetFrom = targetFrom, targetAlpha = targetAlpha, size = size, deltaSize = deltaSize, selectionExponent = selectionExponent, terminal = terminal)

def saveMinimaxTrainData(boardSize, dataFile, modelFile = None, useMoveTrees = False, targetFrom = None, targetAlpha = 0, size = 2**22, deltaSize = 2048, selectionExponent = 1, terminal = False):
    cells, values = generateMinimaxTrainData(boardSize,modelFile,useMoveTrees,targetFrom,targetAlpha,size,deltaSize,selectionExponent, terminal)
    np.savez(dataFile,cells = cells,values = values)

def saveRootMinimaxTrainData(boardSize, dataFile, modelFile, size = 2**22, randomization = 1):
    with ModelPredictor(boardSize,modelFile) as predictor:
        minimax = Minimax(Board(boardSize), heuristic = predictor.predict)
        minimax.expand(size,1024,statusInterval = 10,uniformDepthFactor = np.inf, uniformDepthRandomization = randomization)
        cells = []
        values = []
        for board,value in minimax.collectLeafValues():
            cells.append(board.cells)
            values.append(value)
        np.savez(dataFile,cells = cells,values = values)        

def hashCells(cells):
    hash = hashlib.md5()
    hash.update(cells.tobytes())
    digest = hash.digest()[:(sys.maxsize.bit_length()+1)//8]
    return int.from_bytes(digest, byteorder = "big", signed = True)

def hashCellsArray(cellsArray):
    class HashCellsThread(Thread):
        def __init__(self,cellsPortion):
            super().__init__(name = "hashCells")
            self.cellsPortion = cellsPortion
        def run(self):
            self.hashes = np.apply_along_axis(lambda w:hashCells(w.reshape(self.cellsPortion.shape[1:])),1,self.cellsPortion.reshape((self.cellsPortion.shape[0],np.product(self.cellsPortion.shape[1:]))))            
    threads = []
    n = multiprocessing.cpu_count()
    p = math.ceil(len(cellsArray)/n)
    for i in range(n):
        thread = HashCellsThread(cellsArray[i*p:(i+1)*p])
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return np.concatenate([t.hashes for t in threads])
    
def formatMinimaxTrainData(cells,values,validation = 1/16):
    hashes = hashCellsArray(cells)
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

def modelWeightsFromBase(modelFileBase,modelFile):
    modelBase = load_model(modelFileBase)
    model = load_model(modelFile)
    for i,layer in enumerate(model.layers):
        """
        if i in (4,8,12):
            w = layer.get_weights()
            w[0][:] = 0
            w[0][1,1] = np.identity(w[0].shape[2])
            w[1][:] = 0
            layer.set_weights(w)
        else:
            layer.set_weights(modelBase.layers[i - (i > 4) - (i > 8) - (i > 12)].get_weights())
        """
        w0 = modelBase.layers[i].get_weights()
        w = layer.get_weights()
        for a0,a in zip(w0,w):
            #a[:] = 0
            s = tuple(slice(0,n) for n in a0.shape)
            a[s] = a0
        layer.set_weights(w)
    model.save(modelFile)

def modelWeightsFromPrevious(modelFilePrevious,modelFile):
    modelPrevious = load_model(modelFilePrevious)
    model = load_model(modelFile)
    for i,layer in enumerate(model.layers):
        if i not in (5,6):
            layer.set_weights(modelPrevious.layers[i - (i >= 5) * 2].get_weights())
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

def generateSaveModel(boardSize,filters,extra,modelFile):
    model = generateModel(boardSize,filters,extra)
    model.summary()
    model.save(modelFile)

def successRatio(predictor, other, n = 1000, e = 3):
    ratio = predictor.successRatioAverage(other,n,e)
    print(f"ratio: {ratio}")

def successRatioAgainstVoid(boardSize, modelFile, n = 1000, e = 3):
    with ModelPredictor(boardSize,modelFile) as predictor:
        successRatio(predictor,VoidPredictor(boardSize),n,e)

def successRatioAgainstOther(boardSize, modelFile, modelFileOther, n = 1000, e = 3):
    with ModelPredictor(boardSize,modelFile,modelFileOther) as predictor:
        successRatio(predictor.singleModelPredictor(0),predictor.singleModelPredictor(1),n,e)

if __name__ == '__main__':
    #heuristicTrain(7)
    #heuristicTest(7)
    #modelAlter("model7_lr.h5")
    #minimaxTrain("data7.npz","model7.h5",fraction = 1)
    #minimaxTrain("data7_01.npz","model7_01.h5",fraction = 1)
    #modelDesign(7)
    #saveMinimaxTrainData(7,"data7.npz","model7.h5",targetFrom = 0.5, targetAlpha = math.log(2)/0.40, size = 2**23, deltaSize = 2**12)
    #saveMinimaxTrainData(7,"data7_01.npz","model7_01.h5",targetFrom = None, targetAlpha = math.log(2)/0.10, deltaSize = 2**14)
    #saveRootMinimaxTrainData(7,"data7_root.npz","model7.h5", size = 2**22, randomization = 1)
    #modelDesign3(3,"model3_new.h5")
    #print(dataMinError("data3.npz"))
    #minimaxTrain("data3.npz","model3_new.h5",validation=0)
    #checkAccuracy(7,"model7.h5")
    #generateSaveModel(7,"model7_01.h5")
    #successRatioAgainstVoid(7,"model7.h5")
    #successRatioAgainstOther(7,"model7.h5","model7_old.h5")
    
    #saveMinimaxTrainData(3,"data3.npz",size = 2**16, deltaSize = 2**16, terminal = True)
    #generateSaveModel(3,16,0,"model3.h5")
    #minimaxTrain("data3.npz","model3.h5",validation = 0)
    #successRatioAgainstVoid(3,"model3.h5")

    #saveMinimaxTrainData(5,"data5.npz", deltaSize = 2**14, terminal = True)
    #saveMinimaxTrainData(5,"data5.npz","model5_new.h5",targetFrom = 0.5, targetAlpha = math.log(2)/0.2, size = 2**22, deltaSize = 2**14)
    #generateSaveModel(5,16,0,"model5.h5")
    #minimaxTrain("data5.npz","model5.h5")
    #successRatioAgainstVoid(5,"model5.h5")
    #successRatioAgainstOther(5,"model5.h5","model5_old.h5")
    #generateSaveModel(5,24,1,"model5_new.h5")
    #modelWeightsFromBase("model5.h5","model5_new.h5")
    #minimaxTrain("data5.npz","model5_new.h5")
    #modelAlter("model5_new.h5")
    #successRatioAgainstOther(5,"model5_new.h5","model5.h5")
    
    
    #generateSaveModel(7,24,2,"model7_.h5")
    #modelWeightsFromPrevious("model5.h5","model7_.h5")
    saveMinimaxTrainData(7,"data7.npz","model7.h5",targetFrom = 0.5, targetAlpha = math.log(2)/1, size = 2**22, deltaSize = 2**12)
    #minimaxTrain("data7.npz","model7.h5")
    #successRatioAgainstOther(7,"model7.h5","model7_old.h5")
    #successRatioAgainstVoid(7,"model7.h5")
    #generateSaveModel(7,48,2,"model7_new.h5")
    #modelWeightsFromBase("model7.h5","model7_new.h5")
    #minimaxTrain("data7.npz","model7_new.h5")
    #successRatioAgainstOther(7,"model7_new.h5","model7.h5")
    #generateSaveModel(7,32,3,"model7_new2.h5")
    #modelWeightsFromBase("model7.h5","model7_new2.h5")
    #minimaxTrain("data7.npz","model7_new2.h5")
    #generateSaveModel(7,40,2,"model7_new3.h5")
    #modelWeightsFromBase("model7.h5","model7_new3.h5")
    #minimaxTrain("data7.npz","model7_new3.h5")

    #successRatioAgainstOther(7,"model7_new.h5","model7.h5")
    #successRatioAgainstOther(7,"model7_new2.h5","model7.h5")
    #successRatioAgainstOther(7,"model7_new3.h5","model7.h5")
    #successRatioAgainstOther(7,"model7_new2.h5","model7_new.h5")
    #successRatioAgainstOther(7,"model7_new3.h5","model7_new.h5")
    #successRatioAgainstOther(7,"model7_new3.h5","model7_new2.h5")

    
    
    
    
    
    
    

