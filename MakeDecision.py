import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math
import csv
import os
import sklearn
from autograd import grad
#Eytan will write some code and I will use the stuff he writes under the eytan package
class nickNet():
    def __init__(self, X, Y, layers, hiddenSize):
        self.X = X
        self.Y = Y
        self.input = X[0].size
        self.output = 1
        self.layers = layers
        self.hidden = hiddenSize
        self.W1 = np.random.randn(self.input, self.hidden)
        self.Wfinal = np.random.randn(self.hidden,self.output)
        self.Wmid = np.array((), dtype = float)
        self. wAll = []
        for k in range(layers - 1):
            if (k%2 != 0):
                np.append(self.Wmid,self.wfinal.T)
            else:
                np.append(self.Wmid,self.wfinal)
        self.wAll.append(self.W1)
        self.wAll.append(self.Wmid)
        self.wAll.append(self.Wfinal)
        self.costPrime = grad(self.cost())

    def activate(self, num):
        return 1/(1+np.exp(-num))
    def netRun(self):
        final = []
        for k in range(self.X.shape[0]):
            result = self.net(self.X[k])
            final.append(result)
        return final
    def net(self, var):
        result = np.dot(var, self.W1)
        result = self.activate(result)
        for k in range(self.Wmid.shape[0]):
            result = np.dot(result, self.wMid[k])
            result = self.activate(result)
        result = np.dot(result, self.Wfinal)
        result = self.activate(result[0])
        result = round(result)
        return result
    def cost(self):
        tot = 0;
        out = self.netRun()#Make a method that runs every input
        for k in range(len(out)):
            tot+= (out[k]-self.Y[k])**2
        tot*=.5
        return tot
    def mutate(self):
        self.W1 += self.W1*.01*random.choice((-1,1))
        for k in range(self.Wmid.shape[0]):
            self.Wmid[k] += self.Wmid*.01*random.choice((-1,1))
        self.Wfinal += self.Wfinal*.01*random.choice((-1,1))
        
class Trainer():
    def __init__(self, NN, tol):
        self.NN = NN
        self.tol = tol
    def optimize(self):
        while (sum(abs(self.NN.costPrime())) > tol):
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
#inputs = [u, d, l, r, a, b, x, y, rb, lb, upLeft, upright, upa, upb, upx, upy, uprb, uplb, downleft, downright, downa, downb, downx, downy, downrb, downlb. lefta, leftb, leftx, ly, leftrb, leftlb, righta, rightb, rightx,righty,rightlb,rightrb] 
#nums  = [0,1,2,3,4,5,||||6,7,8,9,10,11,12,13,14,15,16,17]

def playGame(NN):
    buttons = ['u','d','l', 'r','a','b','x','y','q','p']
    inputs = []
    for k in range(len(buttons)):
        inputs.append(buttons[k])
    for k in range(len(buttons)):
        for i in range(k+1,len(buttons)):
            inputs.append(buttons[k] +buttons[i])
        
    for k in range(len(buttons)):
        for i in range(k+1, len(buttons)):
            for j in range(i + 1, len(buttons)):
                inputs.append(buttons[k] + buttons[i] + buttons[j])
    ramCommand = pd.readcsv('isThereANewFrame.csv')
    command = ramCommand.values
    int timer = 0
    while(timer < 100):
        if(not np.array_equal(pd.readscv('isThereANewFrame.csv').value, command)):
            command = pd.readscv('isThereANewFrame.csv').values
            Mario = NN.net(command[0:command.size-1])
            command = inputs[Mario]
            write = [0,0,0,0,0,0,0,0,0,0]
            for k in range(len(Command)):
                if(command[k] == 'u'):
                    write[0] = 1
                elif(command[k] == 'd'):
                    write[1] = 1
                elif(command[k] == 'l'):
                    write[2] = 1
                elif(command[k] == 'r'):
                    write[3] = 1
                elif(command[k] == 'a'):
                    write[4] = 1
                elif(command[k] == 'b'):
                    write[5] = 1
                elif(command[k] == 'x'):
                    write[6] = 1
                elif(command[k] == 'y'):
                    write[7] = 1
                elif(command[k] == 'q'):
                    write[8] = 1
                elif(command[k] == 'p'):
                    write[9] = 1
            with open('heyEytan.csv', 'w') as csvfile:
                for k in write:
                    print ','.join(k)
        timer+=1
        return command[command.size-1]
                    
            
           
                   
               
           
           
           
    # [up, down, left, right, a, b, x, y, rb, lb] [0 = off, 1 = on]
         
def main():  
    X = np.array(([1, 4, 5], [3,6,3], [6,8,5], [4,8,9], [2,5,3]), dtype = float)
    Y = np.array([1,2,5,8,2], dtype = float)
    NN = nickNet(X,Y,1,3)
    print(NN.costPrime())
    NN.mutate()
                 
    
    data_df = pd.read_csv('ramValuesAndInputs.csv')
    data = data_df.values
    #Eliminate all values which don't change much
    for k in range(data.size()):
        if(data[k].var() < 5):
            data.delete(k)
            k-=1
    NN = nickNet(data, data[5], 2, 4)
    Train = Trainer(NN, 1)
    NN = Train.optimize()
    howFar = eytan.playGame(NN)
    marios = []
    marios.append(NN)
    indexBest = 0
    for k in range(9):
        marios.append(clone.deepclone(NN.mutate()))
    for k in range(1000):
        for i in range(10):
            run = eytan.playGame(marios[k])
            if(run > howFar):
                howFar = run
                indexBest = i
        Best = clone.deepclone(marios[indexBest])
        marios = []
        marios.append(Best)
        for i in range():
            marios.append(clone.deepclone(Best.mutate()))
    
main()