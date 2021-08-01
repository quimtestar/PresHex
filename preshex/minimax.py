import sys
import functools
from collections import deque
import random
import math
import time
import numpy as np

class Minimax(object):

    class Node(object):
        
        def __init__(self, minimax, board):
            self.minimax = minimax
            self.board = board
            self.valueFactor = board.turn
            self.parentBoards = set()
            self.successors = None
            self._ownValue = None
            self._bestLeaf = None
            self._bestMovesAndSuccessors = None
            self._sortedSuccessors = None
        
        def computeOwnValue(self):
            if self.board.winner:
                return self.board.winner
            else:
                return self.minimax.heuristic(self.board)
        
        def ownValue(self):
            if self._ownValue is None:
                self._ownValue = self.computeOwnValue()
            return self._ownValue

        def bestLeaf(self):
            if self._bestLeaf is None:
                self._bestLeaf = self.computeBestLeaf()
            return self._bestLeaf
        
        def computeBestMovesAndSuccessors(self):
            items = self.successors
            best = []
            while items:
                pivot = self.successorSortKey(items[0][1])
                best = []
                items_ = []
                for w in items:
                    k = self.successorSortKey(w[1])
                    if k >= pivot:
                        best.append(w)
                        if k > pivot:
                            items_.append(w)
                items = items_
            if self.successors and not best:    #XXX debugging
                print(f"successors: {self.successors}, best:{best}")  
            return best
        
        def bestMovesAndSuccessors(self):
            if self._bestMovesAndSuccessors is None:
                self._bestMovesAndSuccessors = self.computeBestMovesAndSuccessors()
            return self._bestMovesAndSuccessors
        
        def bestMoveAndSuccessor(self):
            try:
                return random.choice(self.bestMovesAndSuccessors())
            except IndexError:
                return None, None

        def bestMove(self):
            return self.bestMoveAndSuccessor()[0]
        
        def bestSuccessor(self):
            s = self.bestMoveAndSuccessor()[1]
            if s == None:   #XXX debugging
                pass
            return s
        
        def computeBestLeaf(self):
            if self.successors:
                s = self.bestSuccessor().bestLeaf()
                return (s[0], s[1] + 1)
            else:
                return (self, 0)

        def leafValue(self):
            return self.bestLeaf()[0].ownValue()
        
        def leafDistance(self):
            return self.bestLeaf()[1]
        
        def meanSuccessorLeafValue(self):
            if self.successors:
                return np.mean(np.fromiter(map(lambda w: w[1].leafValue(),self.successors),dtype=float))
            else:
                return 0

        def successorSortKey(self, s):
            v = s.leafValue() * self.valueFactor
            return (v, -s.leafDistance() * v, -len(s.bestMovesAndSuccessors())) 
 
        def makeSuccessors(self):
            assert self.successors is None
            self.successors = list(map(lambda move: (move, self.minimax.node(self.board.moveOnCopy(move))),self.board.possibleMoves()))
            for m,s in self.successors:
                s.parentBoards.add(self.board)

        def expandLeaf(self, uniformDepthFactor = None):
            if uniformDepthFactor:
                return self.expandLeafUniformDepth(factor = uniformDepthFactor)
            else:
                return self.expandLeafClassic()

        def expandLeafClassic(self):
            if self.successors is None:
                self.makeSuccessors()
                self.clearParents()
                return self
            elif self.successors:
                return self.bestSuccessor().expandLeaf()

        def computeSortedSuccessors(self):
            return sorted([s for m,s in self.successors], key = self.successorSortKey, reverse = True)

        def sortedSuccessors(self):
            if self._sortedSuccessors is None:
                self._sortedSuccessors = self.computeSortedSuccessors()
            return self._sortedSuccessors

        def expandLeafUniformDepth(self, factor):
            if self.successors is None:
                if self.minimax.numBoardsByMoves(self.board.moves + 1) < factor * self.minimax.size() / self.minimax.moveDepth():
                    self.makeSuccessors()
                    self.clearParents()
                    return self
            elif self.successors:
                for s in self.sortedSuccessors():
                    l = s.expandLeafUniformDepth(factor)
                    if l:
                        return l

        def reset(self):
            self._bestMovesAndSuccessors = None
            self._sortedSuccessors = None
            if self._bestLeaf:
                for m,s in self.bestMovesAndSuccessors():
                    if self._bestLeaf[0] == s.bestLeaf()[0]:
                        return False
                self._bestLeaf = None
                return True
            else:
                return False

        def clearParents(self):
            stack = [ self.board ]
            while stack:
                node = self.minimax.nodes.get(stack.pop())
                if node:
                    if node.reset():
                        stack.extend(node.parentBoards)
            
        def bestChain(self):
            m,s = self.bestMoveAndSuccessor()
            if m and s:
                return [(m, s)] + s.bestChain()
            else:
                return []
            
        def trace(self, indent = 0, rank = 0, file = sys.stdout):
            self.board.trace(indent, file = file)
            print(" " * indent + f"valueFactor: {self.valueFactor} ownValue: {self.ownValue()} leafValue: {self.leafValue()} leafDistance: {self.leafDistance()} rank: {rank} board hash: {hash(self.board)}", file = file)
            if self.successors:
                for r,s in enumerate(sorted((s for m,s in self.successors), key = self.successorSortKey, reverse = True)):
                    s.trace(indent = indent + 8, rank = r, file = file)
            
    
    def __init__(self,board = None, heuristic = lambda board:0):
        self.nodes = {}
        self.boardsByMoves = []
        self.heuristic = heuristic
        self.pruneDeque = deque()
        self.root = None
        if board:
            self.setRootBoard(board)
        
    def setRootBoard(self, board):
        oldRoot = self.root
        self.root = self.node(board.copy())
        if oldRoot and self.root != oldRoot:
            self.pruneDeque.append(oldRoot)
      
    def node(self,board):
        n = self.nodes.get(board)
        if n is None:
            n = self.nodes[board] = self.Node(self,board)
            if board.moves >= len(self.boardsByMoves):
                self.boardsByMoves.extend(set() for i in range(board.moves + 1 - len(self.boardsByMoves)))
            self.boardsByMoves[board.moves].add(board)
        return n
        
    def expandLeaf(self, uniformDepthFactor = None):
        if self.root:
            n = self.root.expandLeaf(uniformDepthFactor)
            if n is None and uniformDepthFactor:
                return self.root.expandLeaf() 
            else:
                return n

    def bestChain(self):
        if self.root:
            return self.root.bestChain()
        else:
            return []
        
    def bestChainStr(self):
        chain = self.bestChain()
        return f"{[m for m,s in chain]}: {chain[-1][1].ownValue():.2f}" if chain else ""

    def statusText(self):
        return f"({self.size()}) {self.bestChainStr()}"

    def expand(self, size, margin, status = lambda s: print(s,file = sys.stderr), aborted = lambda:False, statusInterval = 1, uniformDepthFactor = None):
        fully = False
        t0 = time.time()
        while self.size() >= size:
            if self.prune(margin) <= 0:
                break
        while self.size() < size + margin:
            while self.size() >= size:
                if self.prune(margin) <= 0:
                    break
            size0 = self.size()
            status(self.statusText())
            while self.size() < size0 + margin:
                if aborted():
                    break
                if time.time() - t0 >= statusInterval:
                    status(self.statusText())
                    t0 = time.time()
                if not self.expandLeaf(uniformDepthFactor):
                    fully = True
                    break
            else:
                continue
            break
        while self.size() >= size:
            if self.prune(margin) <= 0:
                break
        status(self.statusText())
        return fully

    def prune(self, amount = 0):
        pruned = 0
        while self.pruneDeque and (amount <= 0 or pruned < amount):
            node = self.pruneDeque.popleft()
            if self.root != node and len(node.parentBoards) == 0:
                node_ = self.nodes.pop(node.board, None)
                if node_:
                    pruned += 1
                    self.boardsByMoves[node_.board.moves].discard(node_.board)
                    while self.boardsByMoves and not self.boardsByMoves[-1]:
                        self.boardsByMoves.pop()
                    if node_.successors:
                        for m,s in node_.successors:
                            s.parentBoards.remove(node_.board)
                            self.pruneDeque.appendleft(s)
        return pruned
    
    def purge(self):
        self.nodes = {}
        self.boardsByMoves = []
        self.pruneDeque = deque()
        self.root = None

    def size(self):
        return len(self.nodes)
    
    def bestMove(self):
        if self.root:
            return self.root.bestMove()
        
    def leafValue(self):
        if self.root:
            return self.root.leafValue()
        
    def leafDistance(self):
        if self.root:
            return self.root.leafDistance()
        
    def collectLeafValues(self):
        return map(lambda w: (w[0],w[1].leafValue()), self.nodes.items())

    def moveDepth(self):
        return len(self.boardsByMoves)
    
    def numBoardsByMoves(self,moves):
        return len(self.boardsByMoves[moves]) if moves < self.moveDepth() else 0

    def trace(self, file = sys.stdout):
        if self.root:
            return self.root.trace(file = file)
        
