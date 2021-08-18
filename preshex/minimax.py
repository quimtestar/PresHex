import sys
import functools
from collections import deque
import random
import math
import time
import numpy as np
import utils
import bisect

class Minimax(object):

    class Node(object):
        
        def __init__(self, minimax, board):
            self.minimax = minimax
            self.board = board
            self.valueFactor = board.turn
            self.parentBoards = set()
            self.successors = None
            self._ownValue = None
            self._selectedMoveAndSuccessor = None
            self._bestMovesAndSuccessors = None
            self._bestLeaf = None
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
        
        def computeSelectedMoveAndSuccessor(self):
            if self.successors:
                s = 0
                acc = []
                infinites = []
                for move,node in self.successors:
                    v = self.successorSortKey(node)
                    if math.isclose(v,1):
                        infinites.append((move,node))
                        s += math.inf
                    else:
                        s += math.pow((1 + v)/(1 - v),self.minimax.selectionExponent)
                    acc.append(s)
                if infinites:
                    return random.choice(infinites)
                elif s > 0:
                    return self.successors[bisect.bisect(acc,s*random.random())]
                else:
                    return random.choice(self.successors)
            else:
                return None, None
            
        def selectedMoveAndSuccessor(self):
            if self._selectedMoveAndSuccessor is None:
                self._selectedMoveAndSuccessor = self.computeSelectedMoveAndSuccessor()
            return self._selectedMoveAndSuccessor
        
        def selectedMove(self):
            return self.selectedMoveAndSuccessor()[0]
        
        def selectedSuccessor(self):
            return self.selectedMoveAndSuccessor()[1]

        def computeBestMovesAndSuccessors(self):
            if self.successors:
                maxKey = max(map(lambda s:self.successorSortKey(s[1]),self.successors))
                return list(filter(lambda s:self.successorSortKey(s[1]) >= maxKey,self.successors))
            else:
                return []
        
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
            return self.bestMoveAndSuccessor()[1]

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

        def makeSuccessors(self):
            assert self.successors is None
            self.successors = list(map(lambda move: (move, self.minimax.node(self.board.moveOnCopy(move))),self.board.possibleMoves()))
            for m,s in self.successors:
                s.parentBoards.add(self.board)

        def expandLeaf(self):
            if self.successors is None:
                self.makeSuccessors()
                self.clearParents()
                return self
            elif self.successors:
                return self.selectedSuccessor().expandLeaf()

        def successorSortKey(self, s):
            return s.leafValue() * self.valueFactor

        def computeSortedSuccessors(self):
            return sorted([s for m,s in self.successors], key = self.successorSortKey, reverse = True)

        def sortedSuccessors(self):
            if self._sortedSuccessors is None:
                self._sortedSuccessors = self.computeSortedSuccessors()
            return self._sortedSuccessors

        def reset(self):
            self._selectedMoveAndSuccessor = None
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
            
        def selectedChain(self):
            m,s = self.selectedMoveAndSuccessor()
            if m and s:
                return [(m, s)] + s.selectedChain()
            else:
                return []
            
        def trace(self, indent = 0, rank = 0, file = sys.stdout):
            self.board.trace(indent, file = file)
            print(" " * indent + f"valueFactor: {self.valueFactor} ownValue: {self.ownValue()} leafValue: {self.leafValue()} leafDistance: {self.leafDistance()} rank: {rank} board hash: {hash(self.board)}", file = file)
            if self.successors:
                for r,s in enumerate(sorted((s for m,s in self.successors), key = self.successorSortKey, reverse = True)):
                    s.trace(indent = indent + 8, rank = r, file = file)
            
    
    def __init__(self,board = None, heuristic = None, selectionExponent = 3):
        self.nodes = {}
        self.boardsByMoves = []
        self.heuristic = heuristic if heuristic is not None else lambda board:0
        self.selectionExponent = selectionExponent
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
        
    def expandLeaf(self):
        if self.root:
            return self.root.expandLeaf() 

    def selectedChain(self):
        if self.root:
            return self.root.selectedChain()
        else:
            return []
        
    def selectedChainStr(self):
        chain = self.selectedChain()
        return f"{[m for m,s in chain]}: {chain[-1][1].ownValue():.2f}" if chain else ""

    def statusText(self):
        return f"({self.size()}) {self.selectedChainStr()}"

    def expand(self, size, margin, target = None, status = lambda s: print(s,file = sys.stderr), aborted = lambda:False, statusInterval = 1):
        reached = False
        t0 = time.time()
        while self.size() >= size:
            if self.prune(margin) <= 0:
                break
        while self.size() < size + margin:
            while self.size() >= size:
                if self.prune(margin) <= 0:
                    break
            size0 = self.size()
            if time.time() - t0 >= statusInterval:
                status(self.statusText())
                t0 = time.time()
            while self.size() < size0 + margin:
                if aborted():
                    break
                if time.time() - t0 >= statusInterval:
                    status(self.statusText())
                    t0 = time.time()
                expanded = self.expandLeaf()
                if (not expanded) or (target is not None and abs(self.leafValue()) > target):
                    reached = True
                    break
            else:
                continue
            break
        while self.size() >= size:
            if self.prune(margin) <= 0:
                break
        status(self.statusText())
        return reached

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
    
    def selectedMove(self):
        if self.root:
            return self.root.selectedMove()
        
    def leafValue(self):
        if self.root:
            return self.root.leafValue()
        
    def valueFactor(self):
        if self.root:
            return self.root.valueFactor
        
    def leafDistance(self):
        if self.root:
            return self.root.leafDistance()
        
    def collectLeafValues(self, terminal = False):
        nodeItems = self.nodes.items()
        if terminal:
            nodeItems =  filter(lambda w:w[1].bestLeaf()[0].board.winner, nodeItems)
        return map(lambda w: (w[0],w[1].leafValue()), nodeItems)

    def moveDepth(self):
        return len(self.boardsByMoves)
    
    def numBoardsByMoves(self,moves):
        return len(self.boardsByMoves[moves]) if moves < self.moveDepth() else 0

    def trace(self, file = sys.stdout):
        if self.root:
            return self.root.trace(file = file)
        
