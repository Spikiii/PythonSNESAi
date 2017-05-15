import numpy as np
import pandas as od
import tensorflow as tf
import random
import math
import csv
import os
import sklearn
from autograd import grad
import eytan #as WhyIsThisGoddamnLuaPackageNotWorking 
#Eytan will write some code and I will use the stuff he writes under the eytan package
class nickNet():
    def __init__(self, X, Y, layers, hiddenSize):
        self.X = X
        self.Y = Y
        self.input = X[0].size()
        self.output = 1
        self.layers = layers
        self.hidden = hiddenSize
        self.W1 = np.random.randn(self.input, self.hidden)
        self.Wfinal = np.random.randn(self.hidden,self.output)
        self.Wmid = np.array((), dtpye = float)
        for k in range(layers - 1):
            if (k%2 != 0):
                np.append(Wmid,wfinal.T)
            else:
                np.append(Wmid,wfinal)
       
    def activate(self, num):
        return 1/(1+np.exp(-num))
    def netRun(self):
        final = []
        for k in range(X.size):
            result = np.dot(X[k],W1)
            result = self.activate(result)
            for k in rnage(Wmid.size()):
                result = np.dot(result, wMid(k))
                result = self.activate(result)
            result = np.dot(result, self.wfinal)
            result = activate(result)
            final.append(result)
        return final
    def net(self, var):
        return 7
    def cost(self):
        out = netRun(X)#Make a method that runs every input
        for k in range(len(out)):
            tot+= (out[k]-Y[k])**2
        tot*=.5
        return tot
    def costPrime(self):
        return grad(self.cost())
    def mutate():
        self.W1 += self.W1*.01*random.choice((-1,1))
        for k in range(Wmid.size()):
            Wmid[k] += self.Wmid*.01*random.choice((-1,1))
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
        while(k < X.size()):
            tempX = self.X.delete(k)
            newNet = nickNet(tempX, self.Y, self.NN.layers, self.NN.hidden)
            Train = Trainer(newNet, 1)
            Train.train()
            test = Train.NN.netRun()
            if(sklearn.metrics.accuracy_score(self.Y, test) >= sklearn.metrics.accuracy_score(self.Y, self.Result) -.05):
                self.X = self.X.delete(k)
                k-=1
            k+=1
def main():  
    data_df = pd.readcsv('ramValuesAndInputs.csv')
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