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
    def isWin(self, player):
        for k in range(len(self.board)):
            if(self.board[k][0] == player and self.board[k][1] == player and self.board[k][2] == player):
                return True
            
        for k in range(len(self.board[0])):
            if(self.board[0][k] == player and self.board[1][k] == player and self.board[2][k] == player):
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
        while(i < len(self.board) and (0 not in self.board[i])):
            i+=1
        if(i == len(self.board)):
            return True
        else:
            return False
        
class nickNet():
    def __init__(self, X, Y, layers, hiddenSize, ID):
        self.X = X #input data
        self.Y = Y  #output data
        self.ID = ID
        self.inp = X[0].size  #9
        self.output = 1
        self.layers = layers #how deep the network is
        self.hidden = hiddenSize  #each layer has the same size
       # self.W1 = np.random.randn(self.inp, self.hidden)
       # self.Wfinal = np.random.randn(self.hidden,self.output)
       # self.WmidExample = np.random.randn(hiddenSize, hiddenSize)
       # self.Wmid = []
       # for k in range(layers - 1):
       #    Wmid.append(WmidExample)
        self.W = [np.random.randn(self.inp, self.hidden)]
        for k in range(layers-1):
            self.W.append(np.random.randn(hiddenSize, hiddenSize))
        self.W.append(np.random.randn(self.hidden,self.output))
       # self.possibleOut = [0,1,2,3,4,5,6,7,8]
       # self.outNorm = self.normilize(self.possibleOut)
        #self.costPrime = grad(self.cost())
    def __str__(self):
        return('NickNet')
    def normilize(self, data):
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
            print(np.min(num))
        return result
    def netRun(self):
        final = []
        for k in range(self.X.shape[0]):
            result = self.net(self.X[k]).item(0)  ##### added .item(0)
            final.append(result)
        return final
    def net(self, var):
        #transform board to be 1, -0 or 0. 1 means a space is the player id
        for k in range(len(var)):
            if(var[k] == self.ID):
                var[k] = 1
            elif(var[k] != 0):
                var[k] = -1
        result = var
        for w in self.W:
            result = self.activate(np.dot(result, w))
#        result = self.activate(np.dot(var, self.W1))
        #print(var)
#        for k in range(self.Wmid.shape[0]):
#            result = np.dot(result, self.Wmid[k])
#        result = np.dot(result, self.Wfinal).item(0)
#        result = self.activate(result)
        return result.item(0)
    def cost(self):
        tot = 0;
        out = self.netRun()#Make a method that runs every input
        for k in range(len(out)):
            tot+= (out[k]-self.Y[k])**2
        tot*=.5
        return tot
    def mutate(self):
        newNN = copy.deepcopy(self)
        for k in range(len(self.W)):
            newNN.W[k] +=self.W[k]*.02*random.choice((-1,1))
        #self.W1 += self.W1*.1*random.choice((-1,1))
        #for k in range(self.Wmid.shape[0]):
        #    self.Wmid += self.Wmid*.1*random.choice((-1,1))
        #self.Wfinal += self.Wfinal*.1*random.choice((-1,1))
        return newNN

class SmartMoveBot():
    def __init__(self, NN):
        self.brain = NN
        self.ID = self.brain.ID

    def move(self, board):
        valids =[]
        for k in range(len(board)):
            if(board[k] == 0):
                valids.append(k)
        evaluations = []
#        print('move options')
        for k in valids:
            bdCopy = copy.deepcopy(board)
            bdCopy[k] = self.ID
#            print(bdCopy)
            evaluations.append(self.brain.net(bdCopy))  # how does it like this board

#        print('options and valuations')
#        print(valids)
#        print(evaluations)
        #best move is the one which has the max valie in evaluaions
        bestMove = valids[evaluations.index(min(evaluations))]
        return bestMove
        
class validMoveBot():
    def __init__(self, ID):
        self.ID = ID
    def move(self, board):
        valids = []
        for k in range(len(board)):
            if(board[k] == 0):
                valids.append(k)

        return random.choice(valids)
disp = True
def playGame(player1, player2):

    #print('newGameHasStarted')
    game = TicTacToe()
    player1ValidMoves = 0
    player2ValidMoves = 0
    while(not game.isWin(1) and not game.isWin(2) and not game.isFull()):
        gameState = game.getGameState()
        p1Move = player1.move(gameState)  #is the game state changed when this runs?
        if (not game.makeMove(p1Move, 1)):
            print('player 1 made An invalid move')
            return -1000 + player1ValidMoves
        if(game.isFull()):
            break
        player1ValidMoves+=1
        gameState = game.getGameState()
        p2Move = player2.move(gameState)
        if (not game.makeMove(p2Move, 2)):
            print('player 2 made an invalid move %i', p2Move)
            return 1000-player1ValidMoves
        player2ValidMoves+=1
    #game.displayBoard()

    if(game.isWin(player1.ID)):
        return 1
    elif(game.isWin(player2.ID)):
        return -1
    else:
        #print(game.isFull())
        #print('draw')
        return 0
                    
    # [up, down, left, right, a, b, x, y, rb, lb] [0 = off, 1 = on]
         
def main():
    teamsize = 20
    generations = 1000
    X = np.array(([[0,0,0,0,0,0,0,0,0]]), dtype = float) #initial empty board
    Y = np.array([4], dtype = float)
    NN = nickNet(X,Y,1,7,1)
    player1 = SmartMoveBot(NN)
    saved_players = [player1]
    
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
    player2 = validMoveBot(2)
    team1 = []
    team1.append(player1)
    for k in range(teamsize - 1):
        player1.brain.mutate()
        team1.append(copy.deepcopy(player1))
    
    
    for p in range(generations):
        team1Wins = [0]*teamsize
        if p % 100 ==0:
            saved_players.append(team1[1])
            print(team1[0].brain.net([0,2,2,0,0,0,1,1,0]))
        for k in range(len(team1)):
            for i in range(100):
                result = playGame(team1[k],player2)
                team1Wins[k] += result
                    
        best1Index=team1Wins.index(min(team1Wins))
        team1Wins[best1Index] = 10000
        
        second1Index = team1Wins.index(min(team1Wins))
        #make a new list of bots
        best1 = copy.deepcopy(team1[best1Index]) #not sure if we need to copy
        best2 = copy.deepcopy(team1[second1Index])
        team1 = []

        team1.append(best1)
        team1.append(best2)

        for i in range(teamsize/2-1):
            team1.append(SmartMoveBot(best1.brain.mutate()))
            team1.append(SmartMoveBot(best2.brain.mutate()))
    #see how well the generations play against the random player
    for i in range(len(saved_players)):
        score = 0
        for j in range(30):
            score += playGame(saved_players[i], player2)
        print('generation %i, score %i' %(i * 100, score))
                   
    test(best1)

def test(bot):
    game = TicTacToe()
    game.displayBoard()
    while(not game.isWin(1) and not game.isWin(2) and not game.isFull()):
        botMove = bot.move(game.getGameState())
        game.makeMove(botMove, 1)
        print(bot.brain.net(game.getGameState()))
        game.displayBoard()
        print("player please make your move")
        inp = int(input())
        game.makeMove(inp, 2)
        print(bot.brain.net(game.getGameState()))
        game.displayBoard()
        
       
    print('1 win: %i, 2 win: %i, tie: %i' % (game.isWin(1), game.isWin(2), game.isFull()))
def test2():
    bd = [0,0,0,0,0,1,1,2,2]
    X = np.array([bd], dtype = float) #initial empty board
    Y = np.array([4], dtype = float)
    player1 = nickNet(X,Y, layers = 1, hiddensize = 3, ID=1)
    print(player1.Wfinal)
    print(player1.net(bd))
    player1.mutate()
    print(player1.Wfinal)
    print(player1.net(bd))
def test3():
    a = [0,1,2,3]
    print(0 not in a)
#test3()
#test2()
X = np.array(([[0,0,0,0,0,0,0,0,0]]), dtype = float) #initial empty board
Y = np.array([4], dtype = float)
NN = nickNet(X,Y,1,1,1)
player1 = SmartMoveBot(NN)
#test(player1)
main()
