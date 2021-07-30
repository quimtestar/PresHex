import numpy as np
from collections import deque
import sys
import itertools


class Move(object):
    
    def __init__(self,i,j):
        self.i = i
        self.j = j
        
    def __repr__(self):
        return f"{self.columnName(self.i)}{self.rowName(self.j)}"
    
    def __hash__(self):
        return hash((self.i, self.j))
    
    def __eq__(self,other):
        return isinstance(other, Move) and (self.i, self.j) == (other.i, other.j)

    def initial(self,turn):
        if turn > 0:
            return self.i <= 0 
        elif turn < 0:
            return self.j <= 0
        else:
            return False
        
    def final(self,turn,size):
        if turn > 0:
            return self.i >= size
        elif turn < 0:
            return self.j >= size
        else:
            return False 
    
    def adjacent(self,other):
        return -1 <= (self.i - other.i)*(self.j - other.j) <= 0 and 0 < max(abs(self.i - other.i), abs(self.j - other.j)) <= 1
    
    def adjacents(self):
        def g():
            for i in range(self.i-1,self.i+2):
                for j in range(self.j-1,self.j+2):
                    yield Move(i,j)
        return [m for m in g() if self.adjacent(m)]
    
    @staticmethod
    def columnName(i):
        base = ord('z') - ord('a') + 1
        digits = []
        i += 1
        while i > 0:
            digits.append((i - 1) % base + 1)
            i = (i - 1)//base
        digits.reverse()
        return "".join([chr(ord('a') + x - 1) for x in digits])
        
    @staticmethod
    def columnByName(s):
        i = 0
        for c in s:
            i *= ord('z') - ord('a') + 1
            i += ord(c) - ord('a') + 1
        i -= 1
        return i

    @staticmethod
    def rowName(j):
        return str(j)
        
    @staticmethod
    def rowByName(s):
        return int(s)

class Board(object):
    
    def __init__(self,size):
        self.size = size
        self.turn = 1
        self.cells = np.zeros((size,)*2,dtype=np.byte)
        self.floodPos = np.zeros((size,)*2,dtype=bool) 
        self.floodNeg = np.zeros((size,)*2,dtype=bool) 
        self.winner = 0
        self._pendingDistancePos = None
        self._pendingDistanceNeg = None

    def copy(self):
        board = Board(self.size)
        board.turn = self.turn
        board.cells[:] = self.cells
        board.floodPos[:] = self.floodPos
        board.floodNeg[:] = self.floodNeg
        board.winner = self.winner
        return board

    def __hash__(self):
        return hash((self.size,self.turn,self.cells.tobytes()))
    
    def __eq__(self,other):
        return isinstance(other, Board) and (self.size, self.turn) == (other.size, other.turn) and np.array_equal(self.cells,other.cells)

 
    def inBoard(self,move):
        return 0 <= move.i < self.size and 0 <= move.j < self.size
        
    def cell(self,move):
        if self.inBoard(move):
            return self.cells[move.i,move.j]
        else:
            return np.sign((self.size-2*move.i-1)**2-(self.size-2*move.j-1)**2)
        
    def validMove(self, move):
        return self.inBoard(move) and self.cell(move) == 0 and self.winner == 0
    
    class InvalidMove(Exception):
        pass
    
    def move(self,move):
        if not self.validMove(move):
            raise self.InvalidMove(move)
        self.cells[move.i,move.j] = self.turn
        def updateFlood(flood,initial):
            stack = [ move ]
            connected = np.zeros(flood.shape,dtype=bool)
            connection = False
            while stack:
                m = stack.pop()
                if self.inBoard(m) and self.cells[m.i,m.j] == self.turn and not connected[m.i,m.j]:
                    connected[m.i,m.j] = True
                    if initial(m) or flood[m.i,m.j]:
                        connection = True
                    if not flood[m.i,m.j]:
                        stack.extend(m.adjacents())
            if connection:
                flood |= connected
        if self.turn > 0:
            updateFlood(self.floodPos,lambda m:m.initial(1))
            if self.connectedPos():
                self.winner = self.turn
        elif self.turn < 0:
            updateFlood(self.floodNeg,lambda m:m.initial(-1))
            if self.connectedNeg():
                self.winner = self.turn
        self.turn *= -1
        self._pendingDistancePos = None
        self._pendingDistanceNeg = None
        
    def moveOnCopy(self, move):
        board = self.copy()
        board.move(move)
        return board
        
    def validUnmove(self,move):
        return self.inBoard(move) and self.cells[move.i,move.j] == -self.turn
        
    def unmove(self,move):
        if not self.validUnmove(move):
            raise self.InvalidMove(move)
        self.cells[move.i,move.j] = 0
        self.turn *= -1
        def updateFlood(flood,initial):
            stack = list(initial)
            connected = np.zeros(flood.shape,dtype=bool)
            while stack:
                m = stack.pop()
                if self.inBoard(m) and self.cells[m.i,m.j] == self.turn and flood[m.i,m.j] and not connected[m.i,m.j]:
                    connected[m.i,m.j] = True
                    stack.extend(m.adjacents())
            flood[:] = connected
        if self.turn > 0:
            updateFlood(self.floodPos,[ Move(0,j) for j in range(self.size) ])
        elif self.turn < 0:
            updateFlood(self.floodNeg,[ Move(i,0) for i in range(self.size) ])
        self.winner = 0
        self._pendingDistancePos = None
        self._pendingDistanceNeg = None
        
    def possibleMoves(self):
        return filter(lambda m:self.validMove(m),(Move(i,j) for i,j in zip(*np.where(self.cells==0))))
    
    def connectedPos(self):
        return self.floodPos[-1].any()
    
    def findPath(self,flood,index0,index1):
        previousMoves = np.full(flood.shape,None,dtype = object)
        unvisited = set(map(lambda w:Move(*w),zip(*np.where(flood))))
        distances = np.full(flood.shape,np.product(flood.shape),dtype = int)
        distances[index0] = 0
        while unvisited:
            move = min(unvisited, key = lambda m:distances[m.i,m.j])
            for m in filter(lambda m: self.inBoard(m) and m in unvisited, move.adjacents()):
                if distances[move.i,move.j] + 1 < distances[m.i,m.j]:
                    distances[m.i,m.j] = distances[move.i,move.j] + 1
                    previousMoves[m.i,m.j] = move
            unvisited.remove(move)
        path = deque()
        move = np.fromfunction(np.vectorize(Move),flood.shape,dtype=int)[index1][np.argmin(distances[index1])] 
        while move:
            path.appendleft(move)
            move = previousMoves[move.i,move.j]
        return path
        
    
    def connectedPathPos(self):
        if self.connectedPos():
            return self.findPath(self.floodPos,([0]*self.size,range(self.size)),([-1]*self.size,range(self.size)))
        
    def connectedNeg(self):
        return self.floodNeg[:,-1].any()
    
    def connectedPathNeg(self):
        if self.connectedNeg():
            return self.findPath(self.floodNeg,(range(self.size),[0]*self.size),(range(self.size),[-1]*self.size))
        
    def computePendingDistance(self,flood,turn,index0,index1):
        distances = np.full(flood.shape,np.inf,dtype = float)
        for m in map(lambda w:Move(*w),zip(*index0)):
            v = self.cell(m) * turn
            if v == 0:
                d = 1
            elif v > 0:
                d = 0
            else:
                d = np.inf
            if d < distances[m.i,m.j]:
                distances[m.i,m.j] = d
        unvisited = set(map(lambda w:Move(*w),itertools.product(*(range(self.size),)*2))) 
        while unvisited:
            move = min(unvisited, key = lambda m:distances[m.i,m.j])
            for m in filter(lambda m: self.inBoard(m) and m in unvisited, move.adjacents()):
                v = self.cell(m) * turn
                if v == 0:
                    d = 1
                elif v > 0:
                    d = 0
                else:
                    d = np.inf
                if distances[move.i,move.j] + d < distances[m.i,m.j]:
                    distances[m.i,m.j] = distances[move.i,move.j] + d
            unvisited.remove(move)
        return np.min(distances[index1])
    
    def computePendingDistancePos(self):
        return self.computePendingDistance(self.floodPos,1,([0]*self.size,range(self.size)),([-1]*self.size,range(self.size)))
        
    def computePendingDistanceNeg(self):
        return self.computePendingDistance(self.floodNeg,-1,(range(self.size),[0]*self.size),(range(self.size),[-1]*self.size))
    
    def pendingDistancePos(self):
        if self._pendingDistancePos is None:
            self._pendingDistancePos = self.computePendingDistancePos()
        return self._pendingDistancePos

    def pendingDistanceNeg(self):
        if self._pendingDistanceNeg is None:
            self._pendingDistanceNeg = self.computePendingDistanceNeg()
        return self._pendingDistanceNeg
                
    def pendingDistanceHeuristic(self):
        if self.winner == 0:
            n2 = self.pendingDistanceNeg()**2
            p2 = self.pendingDistancePos()**2
            ts = self.turn/2 * (self.pendingDistanceNeg() + self.pendingDistancePos())
            td = self.turn/2 * (self.pendingDistanceNeg() - self.pendingDistancePos())
            return (n2 - p2 + ts) / (n2 + p2 + td)
        else:
            return self.winner
                
    def __str__(self):
        return str({"board": self.cells, "turn": self.turn})
    
    def trace(self, indent = 0, file = sys.stdout):
        print(" " * indent + "-" * 3 * (2 * self.size - 1), file = file)
        for r in range(self.size - 1, -self.size, -1):
            print(" " * indent, end="", file = file)
            for s in range(2 * self.size - 1):
                if (s - r) % 2 == 0 and (s + r) % 2 == 0:
                    m = Move((s - r) // 2, (s + r) // 2)
                    if self.inBoard(m):
                        if self.cell(m) > 0:
                            print(" x ", end="", file = file)
                        elif self.cell(m) < 0:
                            print(" o ", end="", file = file)
                        else:
                            print(" _ ", end="", file = file)
                    else:
                        print("   ", end="", file = file)
                else:
                    print("   ", end="", file = file)
            print(file = file)
        print(" " * indent + "-" * 3 * (2 * self.size - 1), file = file)

