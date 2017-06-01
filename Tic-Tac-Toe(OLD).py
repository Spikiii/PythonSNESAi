from __future__ import print_function
import numpy as np
import pandas as pd
import random
import math
import csv
import os
import sklearn
import copy

    
#Eytan will write some code and I will use the stuff he writes under the eytan package
class TicTacToe():
    def __init__(self):
        self.board =  [[0,0,0],[0,0,0],[0,0,0]]
        for k in range(len(self.board)):
            for i in range(len(self.board[k])):
                self.board[k][i] = 0
    def randomBoard(self):
        for k in range(len(self.board)):
            for i in range(len(self.board[k])):
                self.board[k][i] = random.choice((0,0,1,2))
    def isWin(self, player):
        for k in range(len(self.board)):
            if(self.board[k][0] == player and self.board[k][1] ==self.board[k][0] and self.board[k][2] == self.board[k][0]):
                return True
            
        for k in range(len(self.board[0])):
            if(self.board[0][k] == player and self.board[1][k] == self.board[k][0] and self.board[2][k] == self.board[0][k]):
                return True
        if(self.board[0][0] == player and self.board[1][1] == player and self.board[2][2] == player):
            return True
        if(self.board[0][2] == player and self.board[1][1] == player and self.board[2][0] == player):
            return True
        return False
    def displayBoard(self):
        print('-------------------------')
        for k in range(len(self.board)):
            for i in range(len(self.board[k])):
                print(self.board[k][i], end = ''),
            print()
        print('------------------------')
    def makeMove(self, inp, player):
        row = inp//3
        col = inp%3
        if(self.board[row][col] == 0):
            self.board[row][col] = player
            return True
        else:
            return False
    def getGameState(self):
        state = []
        for k in range(len(self.board)):
            for i in range(len(self.board[k])):
                state.append(float(self.board[k][i]))
        return state
    def isFull(self):
        i = 0
        while(i < len(self.board) and 0 in self.board[i]):
            i+=1
        if(i == len(self.board)):
            return False
        else:
            return True
class validMoveBot():
    def __init__(self, id):
        self.id = id
    def net(self, var):
        valids = []
        for k in range(len(var)):
            if(var[k] == 0):
                valids.append(k)
        return random.choice(valids)
        
class nickNet():
    def __init__(self, X, Y, layers, hiddenSize, id):
        self.X = X #input data
        self.Y = Y  #output data
        self.id = id
        self.inp = X[0].size
        self.output = 1
        self.layers = layers
        self.hidden = hiddenSize
        self.W1 = np.random.randn(self.inp, self.hidden)
        self.Wfinal = np.random.randn(self.hidden,self.output)
        self.Wmid = np.ones(hiddenSize, dtype = float)
        self. wAll = []
        for k in range(layers - 1):
            if (k%2 != 0):
                np.append(self.Wmid,self.Wfinal.T)
            else:
                np.append(self.Wmid,self.Wfinal)
        self.wAll.append(self.W1)
        self.wAll.append(self.Wmid)
        self.wAll.append(self.Wfinal)
        self.possibleOut = [0,1,2,3,4,5,6,7,8]
        self.outNorm = self.normilize(self.possibleOut)
        print(self.outNorm)
        #self.costPrime = grad(self.cost())
        #print(self.wAll)
    def __str__(self):
        return('NickNet')
    def normilize(self, data):
        if(max(data)-min(data) == 0):
            return data
        for k in range(len(data)):
            data[k] = float(data[k])
        madMax = max(data)
        minimum = min(data)
        for k in range(len(data)):
            data[k] = (data[k]-minimum)/(madMax-minimum)
        return data
    def roundClosest(self, data, inp):
        index = 0
        minimum = 1000
        for k in range(len(data)):
            if(abs(data[k] - inp)<minimum):
                minimum = abs(data[k] - inp)
                index = k
        return index
    def activate(self, num):
        '''num is a memory state'''
#        print('max and min of memory state')
#        print(np.max(num))
#        print(np.min(num))
        try:
            result = 1/(1+np.exp(-num))
        except:
            print('problem with num')
            print(np.max(num))
        return result
    def netRun(self):
        final = []
        for k in range(self.X.shape[0]):
            result = self.net(self.X[k])
            final.append(result)
        return final
    def net(self, var):
        for k in range(len(var)):
            if (var[k] == self.id):
                var[k] = 1
            elif(var[k] != self.id and var[k] !=0):
                 var[k] = -1
        var = self.normilize(var)
        result = np.dot(var, self.W1)
        result = self.activate(result)
        #print(var)
        for k in range(self.Wmid.shape[0]):
            result = np.dot(result, self.Wmid[k])
            result = self.activate(result)
        result = np.dot(result, self.Wfinal)
        result = self.activate(result)
        result = self.roundClosest(self.outNorm, result)
        return result
    def cost(self):
        tot = 0;
        out = self.netRun()#Make a method that runs every input
        for k in range(len(out)):
            tot+= (out[k]-self.Y[k])**2
        tot*=.5
        return tot
    def mutate(self):
        self.W1 += self.W1*.1*random.choice((-1,1))
        for k in range(self.Wmid.shape[0]):
            self.Wmid += self.Wmid*.1*random.choice((-1,1))
        self.Wfinal += self.Wfinal*.1*random.choice((-1,1))
        return self
        
class Trainer():
    def __init__(self, NN, tol):
        self.NN = NN
        self.tol = tol
    def optimize(self):
        while (sum(abs(self.NN.costPrime())) > self.tol):
            return 7
            
        
class VarEliminator():
    def __init__(self, X, Y, NN, Train):
        self.X = X
        self.Y = Y
        self.NN = NN
        self.Train = Train
        self.Result = NN.netRun(X)
    def extranious(self):
        k = 0
        while(k < self.X.shape[0]):
            tempX = self.X.delete(k)
            newNet = nickNet(tempX, self.Y, self.NN.layers, self.NN.hidden)
            Train = Trainer(newNet, 1)
            Train.train()
            test = Train.NN.netRun()
            if(sklearn.metrics.accuracy_score(self.Y, test) >= sklearn.metrics.accuracy_score(self.Y, self.Result) -.05):
                self.X = self.X.delete(k)
                k-=1
            k+=1
        return Train.NN
 

disp = True
def playGame(player1, player2):

    #print('newGameHasStarted')
    game = TicTacToe()
    game.randomBoard()
    player1ValidMoves = 0
    player2ValidMoves = 0
    while(not game.isWin(1) and not game.isWin(2) and not game.isFull()):
        gameState = game.getGameState()
        p1Move = player1.net(gameState)  #is the game state changed when this runs?
        print(p1Move)
        if (not game.makeMove(p1Move, 1)):
            print('player 1 made an invalid move')
            return -1000 + player1ValidMoves
        player1ValidMoves+=1
        p2Move = player2.net(gameState)
        if (not game.makeMove(p2Move, 2)):
            print('player 2 made an invalid move')
            return 1000-player1ValidMoves
        player2ValidMoves+=1
        game.displayBoard()

    if(game.isWin(1)):
        return 1
    elif(game.isWin(2)):
        return -1
    else:
        print('draw')
        return 0
                    
    # [up, down, left, right, a, b, x, y, rb, lb] [0 = off, 1 = on]
         
def main():
    teamsize = 100
    generations = 100
    X = np.array(([[0,0,0,0,0,0,0,0,0]]), dtype = float) #initial empty board
    Y = np.array([4], dtype = float)
    player1 = nickNet(X,Y,3,9, 1)
    #print(NN.costPrime())
    #NN.mutate()
                 
    """    
    #Eliminate all values which don't change much
    for k in range(data.size()):
        if(data[k].var() < 5):
            data.delete(k)
            k-=1
    Train = Trainer(NN, 1)
    NN = Train.optimize()
    """
    player2 = nickNet(X, Y, 3, 9, 2)
    team1 = []
    team2 = []
    team1.append(player1)
    team2.append(player2)
    for k in range(teamsize - 1):
        team1.append(copy.deepcopy(player1.mutate()))
        team2.append(copy.deepcopy(player2.mutate()))
    team1Wins = [0]*teamsize
    team2Wins = [0]*teamsize
    

    
    #each bot plays 1 game then mutates
    #1000 generations run
    for p in range(generations):
        for k in range(len(team1)):
            for i in range(len(team2)):
                result = playGame(team1[k],team2[i])
                team1Wins[k] += result
                team2Wins[i] -= result

                    
        
        best1Index=team1Wins.index(max(team1Wins))
        best2Index=team2Wins.index(max(team2Wins))
        #make a new list of bots
        best1 = copy.deepcopy(team1[best1Index]) #not sure if we need to copy
        best2 = copy.deepcopy(team2[best2Index])
        team1 = []
        team2 = []
        team1.append(best1)
        team2.append(best2)

        for i in range(teamsize - 1):
             team1.append(copy.deepcopy(best1.mutate()))
             team2.append(copy.deepcopy(best2.mutate()))#not sure if we need to copy
        print(team1[3]==team1[4])

def test():
    game = TicTacToe()
    bd = [0,0,0,0,0,1,1,2,2]
    random.shuffle(bd)
    X = np.array([bd], dtype = float) #initial empty board
    Y = np.array([4], dtype = float)
    player1 = nickNet(X,Y,1,3)
    mutants = [player1]
    for i in range(9):
        mutants.append(player1.mutate())
    for nn in mutants:
        print(nn.net(bd))
def test2():
    bd = [0,0,0,0,0,1,1,2,2]
    X = np.array([bd], dtype = float) #initial empty board
    Y = np.array([4], dtype = float)
    player1 = nickNet(X,Y,1,3,1)
    player2 = validMoveBot(2)
    team1 = []
    for k in range(100):
       team1.append(copy.deepcopy(player1.mutate()))
    scores = [0]*len(team1)
    for p in range(1000):
        for k in range(len(team1)):
            result = playGame(team1[k], player2)
            scores[k] +=result
        bestIndex = scores.index(max(scores))
        best = copy.deepcopy(team1[bestIndex])
        team1 = []
        team1.append(copy.deepcopy(best))
        for k in range(99):
            team1.append(copy.deepcopy(best.mutate()))
        
test2()
#test()
#main()
